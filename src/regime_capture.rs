//! Opt-in measurement of a v10 trajectory. No runtime RNG, model parameter,
//! optimizer, controller, or world state is changed by this module.

use super::*;

const SKETCH_WIDTH: usize = 32;
const POOL_SIDE: usize = 4;

pub(super) struct RegimeCapture {
    path: String,
    writer: BufWriter<File>,
    fft: Arc<dyn rustfft::Fft<f32>>,
    run_id: String,
    stride: usize,
}

impl RegimeCapture {
    pub(super) fn new(base_dir: &str, run_id: &str, stride: usize) -> Result<Self> {
        let path = format!("{base_dir}/regime_capture_{run_id}.jsonl");
        let writer = BufWriter::new(File::create_new(&path)?);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(CHUNK_SIZE);
        Ok(Self {
            path,
            writer,
            fft,
            run_id: run_id.into(),
            stride,
        })
    }

    pub(super) fn path(&self) -> &str {
        &self.path
    }
    pub(super) fn should_sample(&self, absolute_step: u64) -> bool {
        absolute_step.is_multiple_of(self.stride as u64)
    }

    pub(super) fn internal_views(
        &self,
        model: &ComplexAudioEcosystem,
        fields: [&Tensor; 3],
        hidden: &Tensor,
        refined: &Tensor,
        control: SynthesisControl,
        phases: [f32; 4],
    ) -> Result<BTreeMap<String, Vec<f32>>> {
        let mut views = BTreeMap::<String, Vec<f32>>::new();
        for (name, field, side) in [
            ("field_fine", fields[0], GRID_H),
            ("field_meso", fields[1], MACRO_H),
            ("field_coarse", fields[2], msfield::COARSE_H),
        ] {
            views.insert(name.into(), spatial_view(&flatten_tensor(field)?, side)?);
        }
        views.insert("gru".into(), sketch(&flatten_tensor(hidden)?, 0x9315_7ae1)?);
        views.insert(
            "morphic_output".into(),
            sketch(&flatten_tensor(refined)?, 0x6b89_f391)?,
        );
        let mut morph = hidden.detach();
        for i in 0..model.morphic.depth() {
            let residual = model.morphic.layers[i]
                .forward(&model.morphic.norms[i].forward(&morph)?)?
                .affine(morphic_residual_gain(i), 0.0)?;
            morph = morph.add(&residual)?;
            if [0, 3, 7, 11, 15].contains(&i) {
                let layer = i + 1;
                views.insert(
                    format!("morphic_l{layer:02}"),
                    sketch(&flatten_tensor(&morph)?, 0x70bc_1234 ^ layer as u64)?,
                );
                views.insert(
                    format!("morphic_delta_l{layer:02}"),
                    sketch(&flatten_tensor(&residual)?, 0x97ab_d182 ^ layer as u64)?,
                );
            }
        }
        let mut decoder = sketch(&flatten_tensor(refined)?, 0x4024_cdef)?;
        decoder.extend([
            control.shear_mult,
            control.kick_mult,
            control.inharmonicity,
            control.spectral_tilt,
            control.width_mult,
        ]);
        decoder.extend(phases);
        views.insert("decoder_control".into(), decoder);
        Ok(views)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &mut self,
        step: u64,
        mut views: BTreeMap<String, Vec<f32>>,
        audio_l: &[f32],
        audio_r: &[f32],
        observation: &AudioObservation,
        stereo_corr: f32,
        motif: &MotifDiagnostics,
        ecology: &AdaptiveDynamics,
        confidence: f32,
        active_morph_depth: usize,
        optimizer_updates: u64,
    ) -> Result<()> {
        views.insert(
            "audio_behavior".into(),
            self.audio_view(audio_l, audio_r, observation, stereo_corr)?,
        );
        if views.values().flatten().any(|x| !x.is_finite()) {
            anyhow::bail!("regime capture encountered a non-finite descriptor");
        }
        let record = serde_json::json!({
            "schema": 1, "run_id": self.run_id, "global_step": step,
            "sample_stride": self.stride, "views": views,
            "health": ecology.activity_health, "stagnation": ecology.stagnation,
            "model_confidence": confidence, "motif_nearest_distance": motif.last_nearest_distance,
            "motif_candidates": motif.candidates, "motif_rejected_similarity": motif.rejected_similarity,
            "motif_rejected_quality": motif.rejected_quality,
            "active_morph_depth": active_morph_depth, "optimizer_updates": optimizer_updates,
        });
        serde_json::to_writer(&mut self.writer, &record)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    fn audio_view(
        &self,
        left: &[f32],
        right: &[f32],
        obs: &AudioObservation,
        corr: f32,
    ) -> Result<Vec<f32>> {
        let mut out = obs.values.to_vec();
        out.push(corr);
        let n = left.len().min(right.len()).min(CHUNK_SIZE);
        if n != CHUNK_SIZE {
            anyhow::bail!("regime audio sample has an incomplete chunk");
        }
        let mut spectrum: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let w = 0.5 - 0.5 * (TWO_PI * i as f32 / (n - 1) as f32).cos();
                Complex::new(0.5 * (left[i] + right[i]) * w, 0.0)
            })
            .collect();
        self.fft.process(&mut spectrum);
        let mut bands = [0.0f32; 8];
        let mut chroma = [0.0f32; 12];
        for (k, bin) in spectrum.iter().enumerate().take(n / 2).skip(2) {
            let hz = k as f32 * SAMPLE_RATE as f32 / n as f32;
            let mag = bin.norm();
            let band = ((hz.max(20.0) / 20.0).log2() * 8.0 / (20_000.0f32 / 20.0).log2())
                .floor()
                .clamp(0.0, 7.0) as usize;
            bands[band] += mag;
            if (55.0..=5000.0).contains(&hz) {
                let pitch = (69.0 + 12.0 * (hz / 440.0).log2()).round() as i32;
                chroma[pitch.rem_euclid(12) as usize] += mag;
            }
        }
        let band_sum = bands.iter().sum::<f32>().max(1e-8);
        let chroma_sum = chroma.iter().sum::<f32>().max(1e-8);
        out.extend(bands.map(|x| x / band_sum));
        out.extend(chroma.map(|x| x / chroma_sum));
        let envelope: Vec<f32> = (0..64)
            .map(|j| {
                let start = j * (n / 64);
                let e = (start..start + n / 64)
                    .map(|i| {
                        let m = 0.5 * (left[i] + right[i]);
                        m * m
                    })
                    .sum::<f32>();
                (e / (n / 64) as f32).sqrt()
            })
            .collect();
        let total = envelope.iter().sum::<f32>().max(1e-8);
        for k in 1..=8 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (t, &e) in envelope.iter().enumerate() {
                let angle = TWO_PI * k as f32 * t as f32 / 64.0;
                re += e * angle.cos();
                im += e * angle.sin();
            }
            out.push((re * re + im * im).sqrt() / total);
        }
        Ok(out)
    }

    pub(super) fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(Into::into)
    }
}

fn spatial_view(values: &[f32], side: usize) -> Result<Vec<f32>> {
    if values.len() != CA_CHANNELS * side * side || side % POOL_SIDE != 0 {
        anyhow::bail!("regime field dimensions do not match expected square scale");
    }
    let mut signed = [0.0f32; 2 * POOL_SIDE * POOL_SIDE];
    let mut energy = [0.0f32; 2 * POOL_SIDE * POOL_SIDE];
    let mut mean = 0.0f64;
    let mut square = 0.0f64;
    let mut rail = 0usize;
    let mut neighbor = 0.0f64;
    let block = side / POOL_SIDE;
    for c in 0..CA_CHANNELS {
        for y in 0..side {
            for x in 0..side {
                let offset = c * side * side + y * side + x;
                let v = values[offset];
                let b = (c / (CA_CHANNELS / 2)) * POOL_SIDE * POOL_SIDE
                    + (y / block) * POOL_SIDE
                    + x / block;
                signed[b] += v;
                energy[b] += v * v;
                mean += v as f64;
                square += (v * v) as f64;
                rail += usize::from(v.abs() > 0.9);
                if x > 0 {
                    neighbor += (v - values[offset - 1]).abs() as f64;
                }
                if y > 0 {
                    neighbor += (v - values[offset - side]).abs() as f64;
                }
            }
        }
    }
    let denom = (CA_CHANNELS / 2 * block * block) as f32;
    let mut out = Vec::with_capacity(68);
    out.extend(signed.map(|x| x / denom));
    out.extend(energy.map(|x| (x / denom).sqrt()));
    out.extend([
        (square / values.len() as f64).sqrt() as f32,
        (mean / values.len() as f64) as f32,
        rail as f32 / values.len() as f32,
        (neighbor / (CA_CHANNELS * 2 * side * (side - 1)) as f64) as f32,
    ]);
    Ok(out)
}

// A seeded signed Walsh-Hadamard sketch: the rows are orthogonal by construction.
fn sketch(values: &[f32], seed: u64) -> Result<Vec<f32>> {
    if values.len() != MEMORY_DIM || !values.len().is_power_of_two() {
        anyhow::bail!("regime sketch expects the fixed recurrent width");
    }
    let mut mixed = values.to_vec();
    let mut state = seed;
    for v in &mut mixed {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if state & 1 == 0 {
            *v = -*v;
        }
    }
    let mut stride = 1;
    while stride < mixed.len() {
        for start in (0..mixed.len()).step_by(2 * stride) {
            for i in 0..stride {
                let a = mixed[start + i];
                let b = mixed[start + i + stride];
                mixed[start + i] = a + b;
                mixed[start + i + stride] = a - b;
            }
        }
        stride *= 2;
    }
    let mut indices: Vec<usize> = (0..mixed.len()).collect();
    for i in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        indices.swap(i, (state as usize) % (i + 1));
    }
    Ok(indices
        .into_iter()
        .take(SKETCH_WIDTH)
        .map(|i| mixed[i] / (mixed.len() as f32).sqrt())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sketch_and_spatial_descriptor_are_reproducible_and_finite() -> Result<()> {
        let values: Vec<f32> = (0..MEMORY_DIM).map(|i| (i as f32 * 0.017).sin()).collect();
        assert_eq!(sketch(&values, 99)?, sketch(&values, 99)?);
        assert_ne!(sketch(&values, 99)?, sketch(&values, 100)?);
        let field = vec![0.25; CA_CHANNELS * msfield::COARSE_H * msfield::COARSE_W];
        let view = spatial_view(&field, msfield::COARSE_H)?;
        assert!(view.iter().all(|v| v.is_finite()));
        assert_eq!(view.len(), 68);
        Ok(())
    }
}
