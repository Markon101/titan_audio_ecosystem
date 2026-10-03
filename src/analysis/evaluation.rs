//! Read-only fixed probes for opt-in frozen weight/world evaluation.
use super::super::{ChromaProjector, FixedProbeBank, SpectralProjector, TargetAudioLoader};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};
use std::path::PathBuf;

pub(crate) struct Evaluation {
    spec: SpectralProjector,
    chroma: ChromaProjector,
    development: FixedProbeBank,
    validation: FixedProbeBank,
    development_stereo: Vec<[f32; 3]>,
    validation_stereo: Vec<[f32; 3]>,
    pub manifest: Value,
}

fn geometry(audio: &Tensor) -> Result<[f32; 3]> {
    let x = super::super::stereo_geometry(audio)?;
    Ok([
        x.side_mid_log_ratio.to_scalar::<f32>()?,
        x.correlation.to_scalar::<f32>()?,
        x.level_log_ratio.to_scalar::<f32>()?,
    ])
}

impl Evaluation {
    pub(crate) fn new(loader: Option<&TargetAudioLoader>, device: &Device) -> Result<Self> {
        let loader = loader.ok_or_else(|| anyhow::anyhow!("frozen probes require a corpus"))?;
        anyhow::ensure!(
            loader.development_is_strict && loader.validation_is_strict,
            "frozen probes require strict development and validation families"
        );
        let spec = SpectralProjector::new(device).map_err(anyhow::Error::msg)?;
        let chroma = ChromaProjector::new(device)?;
        let dev = loader.development_chunks(device)?;
        let val = loader.validation_chunks(device)?;
        let stereo = |targets: &Tensor| -> Result<Vec<[f32; 3]>> {
            (0..targets.dim(0)?)
                .map(|index| {
                    geometry(
                        &targets
                            .narrow(0, index, 1)?
                            .reshape((2, super::super::CHUNK_SIZE))?,
                    )
                })
                .collect()
        };
        let identities = |indices: &[usize]| -> Vec<Value> {
            indices
                .iter()
                .take(4)
                .map(|&index| {
                    let file = &loader.files[index];
                    let name = file
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("");
                    let available = file
                        .output_frames
                        .saturating_sub(super::super::CHUNK_SIZE)
                        .max(1);
                    let frame = (super::super::stable_name_hash(name) as usize % available)
                        .min(available - 1);
                    json!({"file": name, "frame_48k": frame, "frames": super::super::CHUNK_SIZE})
                })
                .collect()
        };
        Ok(Self {
            development: FixedProbeBank::new(&dev, &spec, &chroma)?,
            validation: FixedProbeBank::new(&val, &spec, &chroma)?,
            development_stereo: stereo(&dev)?,
            validation_stereo: stereo(&val)?,
            manifest: json!({"development": identities(&loader.development_files),
                             "validation": identities(&loader.validation_files),
                             "strict": true, "probe_feedback": false}),
            spec,
            chroma,
        })
    }

    pub(crate) fn measure(&self, audio: &Tensor, target: &Tensor) -> Result<Value> {
        let spectrum = self.spec.log_mag(audio)?;
        let chroma = self.chroma.features(audio)?;
        let target_spectrum = self.spec.log_mag(target)?;
        let target_chroma = self.chroma.features(target)?;
        let dev = self.development.scores(audio, &spectrum, &chroma)?;
        let val = self.validation.scores(audio, &spectrum, &chroma)?;
        let out_geo = geometry(audio)?;
        let target_geo = geometry(target)?;
        let stereo_error = |probes: &[[f32; 3]]| -> Vec<f32> {
            (0..3)
                .map(|dim| {
                    probes
                        .iter()
                        .map(|probe| (probe[dim] - out_geo[dim]).powi(2))
                        .sum::<f32>()
                        / probes.len() as f32
                })
                .collect()
        };
        let target_error: Vec<f32> = (0..3)
            .map(|i| (target_geo[i] - out_geo[i]).powi(2))
            .collect();
        Ok(json!({
            "development_best_spectral": dev.0.to_scalar::<f32>()?,
            "development_mean_spectral": dev.1.to_scalar::<f32>()?,
            "development_mean_chroma": dev.2.to_scalar::<f32>()?,
            "validation_best_spectral": val.0.to_scalar::<f32>()?,
            "validation_mean_spectral": val.1.to_scalar::<f32>()?,
            "validation_mean_chroma": val.2.to_scalar::<f32>()?,
            "development_stereo_squared_errors": stereo_error(&self.development_stereo),
            "validation_stereo_squared_errors": stereo_error(&self.validation_stereo),
            "target_spectral_robust": super::super::robust_distance(&spectrum.sub(&target_spectrum)?, 0.03)?.to_scalar::<f32>()?,
            "target_chroma_mse": chroma.sub(&target_chroma)?.sqr()?.mean_all()?.to_scalar::<f32>()?,
            "target_stereo_squared_errors": target_error,
            "output_low_band_ratio": super::super::first_band_energy_ratio(&super::super::log_band_energy(&spectrum)?)?.to_scalar::<f32>()?,
            "target_low_band_ratio": super::super::first_band_energy_ratio(&super::super::log_band_energy(&target_spectrum)?)?.to_scalar::<f32>()?,
        }))
    }
}

pub(crate) fn score_audio_files(
    origin: &super::load::LoadedOrigin,
    paths: &[PathBuf],
    stride: usize,
) -> Result<Value> {
    let device = Device::Cpu;
    let loader = super::state::read_only_target_loader(
        &origin.paths.wav_dir,
        &origin.paths.corpus_manifest,
    )?;
    let evaluation = Evaluation::new(loader.as_ref(), &device)?;
    let dummy = Tensor::zeros((2, super::super::CHUNK_SIZE), DType::F32, &device)?;
    let mut reports = Vec::new();
    for path in paths {
        let identity = super::super::provenance::identify_file(path)?;
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.channels == 2
                && spec.sample_rate == 48_000
                && spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int,
            "audio-probe scoring requires stereo 48-kHz PCM16"
        );
        let frames = reader.duration() as usize;
        let chunks = frames / super::super::CHUNK_SIZE;
        anyhow::ensure!(chunks > 0, "audio probe contains no complete chunk");
        let mut scores = Vec::new();
        for offset in 1..=chunks {
            if offset != 1 && !offset.is_multiple_of(stride) && offset != chunks {
                continue;
            }
            let frame = (offset - 1) * super::super::CHUNK_SIZE;
            reader.seek(frame as u32)?;
            let interleaved = reader
                .samples::<i16>()
                .take(2 * super::super::CHUNK_SIZE)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            anyhow::ensure!(
                interleaved.len() == 2 * super::super::CHUNK_SIZE,
                "short audio-probe read"
            );
            let mut values = Vec::with_capacity(interleaved.len());
            values.extend(
                interleaved
                    .iter()
                    .step_by(2)
                    .map(|sample| *sample as f32 / 32768.0),
            );
            values.extend(
                interleaved
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .map(|sample| *sample as f32 / 32768.0),
            );
            let audio = Tensor::from_vec(values, (2, super::super::CHUNK_SIZE), &device)?;
            let mut value = evaluation.measure(&audio, &dummy)?;
            value.as_object_mut().unwrap().retain(|key, _| {
                key.starts_with("validation_")
                    || key.starts_with("development_")
                    || key == "output_low_band_ratio"
            });
            value["offset"] = json!(offset);
            value["frame"] = json!(frame);
            scores.push(value);
        }
        anyhow::ensure!(
            identity == super::super::provenance::identify_file(path)?,
            "audio probe changed during scoring"
        );
        reports.push(
            json!({"identity":identity, "frames":frames,"chunks":chunks,"sampled_metrics":scores}),
        );
    }
    Ok(
        json!({"schema":1,"model_forward_passes":0,"backward_passes":0,"optimizer_steps":0,
        "probe_manifest":evaluation.manifest,"audio_files":reports}),
    )
}
