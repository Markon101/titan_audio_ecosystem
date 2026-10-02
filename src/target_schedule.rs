//! Fixed, exogenous target episodes for the isolated v10 corpus experiment.
//! Normal target sampling never enters this module.

use super::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScheduledEpisode {
    start_step: u64,
    chunks: usize,
    slot: usize,
    source_frame: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScheduleDocument {
    schema: u32,
    seed: u64,
    slots: Vec<String>,
    episodes: Vec<ScheduledEpisode>,
}

pub(super) struct FixedTargetSchedule {
    path: String,
    sha256: String,
    document: ScheduleDocument,
}

impl FixedTargetSchedule {
    pub(super) fn load(path: &str, loader: &TargetAudioLoader) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let document: ScheduleDocument = serde_json::from_slice(&bytes)?;
        let schedule = Self {
            path: path.to_owned(),
            sha256: provenance::sha256_bytes(&bytes),
            document,
        };
        schedule.validate_loader(loader)?;
        Ok(schedule)
    }

    fn validate_loader(&self, loader: &TargetAudioLoader) -> Result<()> {
        let doc = &self.document;
        if doc.schema != 1 || doc.slots.len() != 6 || doc.episodes.is_empty() {
            anyhow::bail!("fixed target schedule requires schema 1, six slots, and episodes");
        }
        if loader.train_groups.len() != doc.slots.len()
            || !loader.development_is_strict
            || !loader.validation_is_strict
        {
            anyhow::bail!("fixed target schedule requires six single-file training families and strict held-out probes");
        }
        for (i, (name, family)) in doc.slots.iter().zip(&loader.train_groups).enumerate() {
            if name != &format!("slot_{i:02}.wav") || family.len() != 1 {
                anyhow::bail!("fixed target slot {i} is not a single canonical WAV alias");
            }
            if loader.files[family[0]]
                .path
                .file_name()
                .and_then(|name| name.to_str())
                != Some(name)
            {
                anyhow::bail!("fixed target slot {i} resolves to the wrong training WAV");
            }
        }
        let mut next_step = doc.episodes[0].start_step;
        for episode in &doc.episodes {
            if episode.start_step != next_step
                || episode.chunks == 0
                || episode.chunks > TARGET_EPISODE_CHUNKS
                || episode.slot >= doc.slots.len()
                || episode.source_frame % CHUNK_SIZE != 0
            {
                anyhow::bail!(
                    "fixed target episodes have a gap, overlap, invalid slot, or unaligned frame"
                );
            }
            let file = &loader.files[loader.train_groups[episode.slot][0]];
            let end_frame = episode
                .source_frame
                .checked_add(
                    episode
                        .chunks
                        .checked_mul(CHUNK_SIZE)
                        .ok_or_else(|| anyhow::anyhow!("target schedule frame overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("target schedule frame overflow"))?;
            if end_frame > file.output_frames {
                anyhow::bail!(
                    "fixed target episode exceeds slot {} WAV duration",
                    episode.slot
                );
            }
            next_step = next_step
                .checked_add(episode.chunks as u64)
                .ok_or_else(|| anyhow::anyhow!("target schedule step overflow"))?;
        }
        Ok(())
    }

    pub(super) fn validate_interval(&self, start_step: u64, chunks: usize) -> Result<()> {
        let first = self
            .document
            .episodes
            .first()
            .expect("validated episodes")
            .start_step;
        let last = self.document.episodes.last().expect("validated episodes");
        let end = last.start_step + last.chunks as u64;
        if start_step < first
            || start_step
                .checked_add(chunks as u64)
                .is_none_or(|step| step > end)
        {
            anyhow::bail!("fixed target schedule does not cover requested global steps");
        }
        Ok(())
    }

    pub(super) fn sample_at(
        &self,
        loader: &mut TargetAudioLoader,
        step: u64,
        k: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if k != 1 {
            anyhow::bail!("fixed target schedule requires TARGET_K=1");
        }
        let episode = self
            .document
            .episodes
            .iter()
            .find(|episode| {
                step >= episode.start_step && step < episode.start_step + episode.chunks as u64
            })
            .ok_or_else(|| anyhow::anyhow!("no fixed target episode covers step {step}"))?;
        let offset = (step - episode.start_step) as usize;
        let cursor = TargetCursor {
            file: loader.train_groups[episode.slot][0],
            output_frame: episode.source_frame + offset * CHUNK_SIZE,
            chunks_left: episode.chunks - offset,
        };
        let (left, right) = loader.decode_window(cursor)?;
        loader.active = None;
        loader.pending.clear();
        loader.prefetched.clear();
        loader.last_served = Some(cursor);
        let mut data = Vec::with_capacity(2 * CHUNK_SIZE);
        data.extend_from_slice(&left);
        data.extend_from_slice(&right);
        Ok(Tensor::from_vec(data, (1, 2, CHUNK_SIZE), device)?)
    }

    pub(super) fn verify_unchanged(&self) -> Result<()> {
        if provenance::sha256_file(&self.path)? != self.sha256 {
            anyhow::bail!("fixed target schedule changed during the run");
        }
        Ok(())
    }

    pub(super) fn path(&self) -> &str {
        &self.path
    }
    pub(super) fn sha256(&self) -> &str {
        &self.sha256
    }
    pub(super) fn seed(&self) -> u64 {
        self.document.seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_schedule_rejects_gaps_and_preserves_exact_frames() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "titan_fixed_target_{}_{}",
            std::process::id(),
            unix_time_ms()
        ));
        std::fs::create_dir(&dir)?;
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut files = Vec::new();
        for slot in 0..6 {
            let path = dir.join(format!("slot_{slot:02}.wav"));
            let mut wav = hound::WavWriter::create(&path, spec)?;
            for frame in 0..3 * CHUNK_SIZE {
                let value = (frame as i16).wrapping_add(slot as i16);
                wav.write_sample(value)?;
                wav.write_sample(value)?;
            }
            wav.finalize()?;
            files.push(TargetAudioLoader::index_wav(&path)?);
        }
        let mut loader = TargetAudioLoader {
            files,
            train_groups: (0..6).map(|i| vec![i]).collect(),
            development_files: vec![],
            validation_files: vec![],
            development_held_out_files: 0,
            held_out_files: 0,
            development_is_strict: true,
            validation_is_strict: true,
            manifest_path: String::new(),
            active: None,
            pending: vec![],
            prefetched: VecDeque::new(),
            last_served: None,
        };
        let mut document = ScheduleDocument {
            schema: 1,
            seed: 3,
            slots: (0..6).map(|i| format!("slot_{i:02}.wav")).collect(),
            episodes: vec![ScheduledEpisode {
                start_step: 100,
                chunks: 2,
                slot: 2,
                source_frame: 0,
            }],
        };
        let path = dir.join("schedule.json");
        std::fs::write(&path, serde_json::to_vec(&document)?)?;
        let schedule = FixedTargetSchedule::load(path.to_str().unwrap(), &loader)?;
        schedule.validate_interval(100, 2)?;
        assert!(schedule.validate_interval(100, 3).is_err());
        let first = schedule.sample_at(&mut loader, 100, 1, &Device::Cpu)?;
        assert_eq!(first.dims(), &[1, 2, CHUNK_SIZE]);
        assert_eq!(loader.episode_info().1, 0);
        loader.commit_selection(0);
        let second = schedule.sample_at(&mut loader, 101, 1, &Device::Cpu)?;
        assert_eq!(second.dims(), &[1, 2, CHUNK_SIZE]);
        assert_eq!(loader.episode_info().1, CHUNK_SIZE);
        assert!(schedule
            .sample_at(&mut loader, 102, 1, &Device::Cpu)
            .is_err());
        document.episodes.push(ScheduledEpisode {
            start_step: 103,
            chunks: 1,
            slot: 0,
            source_frame: 0,
        });
        std::fs::write(&path, serde_json::to_vec(&document)?)?;
        assert!(FixedTargetSchedule::load(path.to_str().unwrap(), &loader).is_err());
        drop(loader);
        for slot in 0..6 {
            std::fs::remove_file(dir.join(format!("slot_{slot:02}.wav")))?;
        }
        std::fs::remove_file(path)?;
        std::fs::remove_dir(dir)?;
        Ok(())
    }
}
