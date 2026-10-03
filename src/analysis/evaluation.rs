//! Read-only fixed probes for opt-in frozen weight/world evaluation.
use super::super::{ChromaProjector, FixedProbeBank, SpectralProjector, TargetAudioLoader};
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde_json::{json, Value};

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
