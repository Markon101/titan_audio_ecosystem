use anyhow::{Context, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArtifactEntry {
    pub relative_path: String,
    pub media_type: String,
    pub sha256: String,
    pub bytes: u64,
    pub condition: Option<String>,
    pub start_offset: Option<usize>,
    pub end_offset: Option<usize>,
    pub processing: String,
}

pub(crate) fn prepare_output(path: &Path, overwrite: bool) -> Result<()> {
    if path.exists() {
        if !overwrite {
            anyhow::bail!(
                "analysis output already exists: {}; choose --analysis-dir or use --analysis-overwrite",
                path.display()
            );
        }
        let marker = path.join("analysis_report.json");
        if marker.exists() {
            anyhow::bail!(
                "refusing to overwrite a completed analysis report at {}; choose a new analysis tag",
                path.display()
            );
        }
    }
    std::fs::create_dir_all(path)?;
    Ok(())
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    ensure_parent(path)?;
    let bytes = serde_json::to_vec_pretty(value)?;
    let _: serde_json::Value =
        serde_json::from_slice(&bytes).context("analysis JSON contains an invalid value")?;
    atomic_bytes(path, &bytes)
}

pub(crate) fn write_text_atomic(path: &Path, text: &str) -> Result<()> {
    atomic_bytes(path, text.as_bytes())
}

pub(crate) fn write_step_csv_atomic(
    path: &Path,
    records: &[super::step::StepRecord],
) -> Result<()> {
    ensure_parent(path)?;
    let temporary = temporary_path(path);
    {
        let mut writer = csv::Writer::from_path(&temporary)?;
        writer.write_record([
            "schema_version",
            "rollout_offset",
            "absolute_step",
            "action",
            "model_proposal",
            "bandit_proposal",
            "force_macro",
            "target_file",
            "target_frame",
            "target_error_feedback",
            "movement",
            "micro_rms",
            "macro_rms",
            "recurrent_rms",
            "micro_near_bound_fraction",
            "macro_near_bound_fraction",
            "synergy",
            "sigma",
            "energy",
            "temperature",
            "radiation_probability",
            "radiation_realized",
            "shear_amplitude",
            "kick_amplitude",
            "morph_active_depth",
            "manifold_depth",
            "far_ring_gain",
            "episodic_slots",
            "motif_occupancy",
            "model_confidence_raw",
            "model_confidence_effective",
            "audio_raw_rms_mid",
            "audio_raw_peak",
            "audio_post_rms_mid",
            "audio_post_peak",
            "audio_post_stereo_correlation",
            "audio_post_width",
            "audio_post_clipping_fraction",
            "audio_post_nonfinite",
            "warnings",
        ])?;
        for record in records {
            writer.write_record([
                "1".to_string(),
                record.rollout_offset.to_string(),
                record.absolute_step.to_string(),
                record.action.clone(),
                record.model_proposal.clone(),
                record.bandit_proposal.clone(),
                record.force_macro.to_string(),
                record.target_file.clone().unwrap_or_default(),
                record
                    .target_frame
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .target_error_feedback
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record.movement.to_string(),
                record.micro_rms.to_string(),
                record.macro_rms.to_string(),
                record.recurrent_rms.to_string(),
                record.micro_near_bound_fraction.to_string(),
                record.macro_near_bound_fraction.to_string(),
                record.synergy.to_string(),
                record.sigma.to_string(),
                record.energy.to_string(),
                record.temperature.to_string(),
                record.radiation_probability.to_string(),
                record.radiation_realized.to_string(),
                record.shear_amplitude.to_string(),
                record.kick_amplitude.to_string(),
                record.morph_active_depth.to_string(),
                record.manifold_depth.to_string(),
                record.far_ring_gain.to_string(),
                record.episodic_slots.to_string(),
                record.motif_occupancy.to_string(),
                record.model_confidence_raw.to_string(),
                record.model_confidence_effective.to_string(),
                record.audio_raw.rms_mid.to_string(),
                record.audio_raw.peak.to_string(),
                record.audio_post_dsp.rms_mid.to_string(),
                record.audio_post_dsp.peak.to_string(),
                record
                    .audio_post_dsp
                    .stereo_correlation
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .audio_post_dsp
                    .correlation_aware_width
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record.audio_post_dsp.clipping_fraction.to_string(),
                record.audio_post_dsp.nonfinite.to_string(),
                record.warnings.join(" | "),
            ])?;
        }
        writer.flush()?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(())
}

pub(crate) fn write_wav_atomic(path: &Path, left: &[f32], right: &[f32]) -> Result<()> {
    ensure_parent(path)?;
    let temporary = temporary_path(path);
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: super::super::SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut writer = hound::WavWriter::create(&temporary, spec)?;
        for (&left, &right) in left.iter().zip(right) {
            let encode = |value: f32| (value.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            writer.write_sample(encode(left))?;
            writer.write_sample(encode(right))?;
        }
        writer.finalize()?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(())
}

pub(crate) fn write_spectrogram_png(path: &Path, left: &[f32], right: &[f32]) -> Result<()> {
    let frames = left.len().min(right.len());
    if frames < 256 {
        anyhow::bail!("spectrogram requires at least 256 frames");
    }
    let mono: Vec<f32> = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left + right) * 0.5)
        .collect();
    let fft_size = 1024usize.min(frames.next_power_of_two() / 2).max(256);
    let hop = (fft_size / 4).max(1);
    let time_bins = ((frames - fft_size) / hop + 1).min(512);
    let frequency_bins = (fft_size / 2).min(256);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut rgb = vec![0u8; time_bins * frequency_bins * 3];
    for time in 0..time_bins {
        let start = time * hop;
        let mut buffer: Vec<Complex<f32>> = mono[start..start + fft_size]
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let window =
                    0.5 - 0.5 * (super::super::TWO_PI * index as f32 / (fft_size - 1) as f32).cos();
                Complex::new(sample * window, 0.0)
            })
            .collect();
        fft.process(&mut buffer);
        for frequency in 0..frequency_bins {
            let source_bin = frequency * (fft_size / 2) / frequency_bins;
            let db = 20.0 * (buffer[source_bin].norm() / fft_size as f32 + 1e-8).log10();
            let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
            let (red, green, blue) = scientific_color(normalized);
            let row = frequency_bins - 1 - frequency;
            let index = (row * time_bins + time) * 3;
            rgb[index] = red;
            rgb[index + 1] = green;
            rgb[index + 2] = blue;
        }
    }
    write_png_rgb(path, time_bins, frequency_bins, &rgb)
}

pub(crate) fn write_state_atlas_png(
    path: &Path,
    values: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<()> {
    if values.len() != channels * height * width {
        anyhow::bail!("state atlas shape does not match {} values", values.len());
    }
    let columns = (channels as f32).sqrt().ceil() as usize;
    let rows = channels.div_ceil(columns);
    let atlas_width = columns * width;
    let atlas_height = rows * height;
    let mut rgb = vec![0u8; atlas_width * atlas_height * 3];
    for channel in 0..channels {
        let tile_row = channel / columns;
        let tile_column = channel % columns;
        for row in 0..height {
            for column in 0..width {
                let value = values[channel * height * width + row * width + column];
                let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
                let (red, green, blue) = diverging_color(normalized);
                let x = tile_column * width + column;
                let y = tile_row * height + row;
                let index = (y * atlas_width + x) * 3;
                rgb[index] = red;
                rgb[index + 1] = green;
                rgb[index + 2] = blue;
            }
        }
    }
    write_png_rgb(path, atlas_width, atlas_height, &rgb)
}

pub(crate) fn artifact_entry(
    root: &Path,
    path: &Path,
    media_type: &str,
    condition: Option<&str>,
    start_offset: Option<usize>,
    end_offset: Option<usize>,
    processing: &str,
) -> Result<ArtifactEntry> {
    let metadata = std::fs::metadata(path)?;
    Ok(ArtifactEntry {
        relative_path: path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string(),
        media_type: media_type.to_string(),
        sha256: super::super::provenance::sha256_file(path)?,
        bytes: metadata.len(),
        condition: condition.map(str::to_string),
        start_offset,
        end_offset,
        processing: processing.to_string(),
    })
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("analysis")
        .to_string();
    name.push_str(".tmp");
    path.with_file_name(name)
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure_parent(path)?;
    let temporary = temporary_path(path);
    {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        writer.write_all(bytes)?;
        writer.flush()?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn scientific_color(value: f32) -> (u8, u8, u8) {
    let value = value.clamp(0.0, 1.0);
    let red = (255.0 * (value * 1.45 - 0.30).clamp(0.0, 1.0)) as u8;
    let green = (255.0 * (1.0 - (value - 0.62).abs() * 2.2).clamp(0.0, 1.0)) as u8;
    let blue = (255.0 * (1.15 - value * 1.35).clamp(0.0, 1.0)) as u8;
    (red, green, blue)
}

fn diverging_color(value: f32) -> (u8, u8, u8) {
    let distance = (value - 0.5).abs() * 2.0;
    let neutral = (240.0 * (1.0 - distance)) as u8;
    if value < 0.5 {
        (neutral / 2, neutral, (155.0 + 100.0 * distance) as u8)
    } else {
        ((155.0 + 100.0 * distance) as u8, neutral, neutral / 2)
    }
}

fn write_png_rgb(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<()> {
    if rgb.len() != width * height * 3 || width == 0 || height == 0 {
        anyhow::bail!("invalid RGB image dimensions");
    }
    let mut scanlines = Vec::with_capacity(height * (1 + width * 3));
    for row in 0..height {
        scanlines.push(0);
        let start = row * width * 3;
        scanlines.extend_from_slice(&rgb[start..start + width * 3]);
    }
    let compressed = zlib_store(&scanlines);
    let mut png = Vec::new();
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&(width as u32).to_be_bytes());
    header.extend_from_slice(&(height as u32).to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"IDAT", &compressed);
    png_chunk(&mut png, b"IEND", &[]);
    atomic_bytes(path, &png)
}

fn png_chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(data);
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(kind);
    crc_data.extend_from_slice(data);
    output.extend_from_slice(&crc32(&crc_data).to_be_bytes());
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut output = vec![0x78, 0x01];
    let chunks = data.chunks(65_535);
    let total = chunks.len();
    for (index, chunk) in data.chunks(65_535).enumerate() {
        output.push(u8::from(index + 1 == total));
        let length = chunk.len() as u16;
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&(!length).to_le_bytes());
        output.extend_from_slice(chunk);
    }
    output.extend_from_slice(&adler32(data).to_be_bytes());
    output
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in data {
        a = (a + *byte as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zlib_store_has_standard_empty_adler() {
        let encoded = zlib_store(&[]);
        assert_eq!(&encoded[encoded.len() - 4..], &[0, 0, 0, 1]);
    }

    #[test]
    fn png_signature_and_chunks_are_emitted() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "titan-analysis-png-{}-{}.png",
            std::process::id(),
            super::super::super::unix_time_ms()
        ));
        write_png_rgb(&path, 1, 1, &[1, 2, 3])?;
        let bytes = std::fs::read(&path)?;
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        std::fs::remove_file(path)?;
        Ok(())
    }
}
