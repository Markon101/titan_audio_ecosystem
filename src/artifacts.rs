use anyhow::Result;
use rand::Rng;

pub fn artifact_path(base: &str, stem: &str, extension: &str, tag: Option<&str>) -> String {
    match tag {
        Some(tag) => format!("{}/{}_{}.{}", base, stem, tag, extension),
        None => format!("{}/{}.{}", base, stem, extension),
    }
}

pub fn model_companion_path(
    model_path: &str,
    companion_stem: &str,
    extension: &str,
) -> Option<String> {
    let path = std::path::Path::new(model_path);
    let model_stem = path.file_stem()?.to_str()?;
    let suffix = model_stem.strip_prefix("titan_model")?;
    let filename = format!("{}{}.{}", companion_stem, suffix, extension);
    Some(path.parent()?.join(filename).to_string_lossy().into_owned())
}

pub fn hashed_audio_path(base: &str, stem: &str, tag: Option<&str>, audio_hash: &str) -> String {
    let unique_tag = match tag {
        Some(tag) => format!("{}_{}", tag, audio_hash),
        None => audio_hash.to_string(),
    };
    artifact_path(base, stem, "wav", Some(&unique_tag))
}

pub fn unique_audio_hash(base: &str) -> Result<String> {
    let mut rng = rand::thread_rng();
    for _ in 0..1024 {
        let candidate = format!("{:012x}", rng.gen::<u64>() & 0xffff_ffff_ffff);
        let suffix = format!("_{}.wav", candidate);
        let collision = std::fs::read_dir(base)?.any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name.ends_with(&suffix))
        });
        if !collision {
            return Ok(candidate);
        }
    }
    anyhow::bail!("could not allocate a unique random audio filename hash")
}

pub fn validate_run_tag(tag: &str) -> Result<()> {
    if tag.is_empty()
        || tag.len() > 48
        || !tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        anyhow::bail!("run tag must be 1..48 ASCII letters, digits, '-' or '_'");
    }
    Ok(())
}
