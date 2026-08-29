use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct FileIdentity {
    pub path: String,
    pub present: bool,
    pub bytes: Option<u64>,
    pub modified_unix_ms: Option<u128>,
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceIdentity {
    pub build_commit: String,
    pub build_dirty: bool,
    pub live_commit: Option<String>,
    pub live_dirty: Option<bool>,
    pub live_branch: Option<String>,
    pub source_manifest: Vec<FileIdentity>,
    pub warnings: Vec<String>,
}

pub(crate) fn identify_file(path: impl AsRef<Path>) -> Result<FileIdentity> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(FileIdentity {
            path: path.display().to_string(),
            present: false,
            bytes: None,
            modified_unix_ms: None,
            sha256: None,
        });
    }
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("reading metadata for {}", path.display()))?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    Ok(FileIdentity {
        path: path.display().to_string(),
        present: true,
        bytes: Some(metadata.len()),
        modified_unix_ms,
        sha256: Some(sha256_file(path)?),
    })
}

pub(crate) fn identify_files(paths: &[PathBuf]) -> Result<Vec<FileIdentity>> {
    paths.iter().map(identify_file).collect()
}

pub(crate) fn unchanged(before: &[FileIdentity], after: &[FileIdentity]) -> bool {
    before.len() == after.len()
        && before.iter().zip(after).all(|(left, right)| {
            left.path == right.path
                && left.present == right.present
                && left.bytes == right.bytes
                && left.modified_unix_ms == right.modified_unix_ms
                && left.sha256 == right.sha256
        })
}

pub(crate) fn source_identity() -> SourceIdentity {
    let mut warnings = Vec::new();
    let live_commit = git_output(&["rev-parse", "HEAD"]);
    let live_branch = git_output(&["branch", "--show-current"]);
    let live_dirty = git_output(&["status", "--porcelain", "--untracked-files=no"])
        .map(|status| !status.is_empty());
    let build_matches_live = live_commit
        .as_deref()
        .is_some_and(|commit| commit.starts_with(super::BUILD_COMMIT));
    if !build_matches_live {
        warnings.push(format!(
            "running source commit {:?} differs from binary build commit {}",
            live_commit,
            super::BUILD_COMMIT
        ));
    }
    let source_manifest = tracked_source_files().unwrap_or_else(|error| {
        warnings.push(format!("source manifest unavailable: {error}"));
        Vec::new()
    });
    SourceIdentity {
        build_commit: super::BUILD_COMMIT.to_string(),
        build_dirty: super::BUILD_DIRTY == "true",
        live_commit,
        live_dirty,
        live_branch,
        source_manifest,
        warnings,
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|text| text.trim().to_string())
}

fn tracked_source_files() -> Result<Vec<FileIdentity>> {
    let output = std::process::Command::new("git")
        .args(["ls-files", "-z"])
        .output()
        .context("running git ls-files for source provenance")?;
    if !output.status.success() {
        anyhow::bail!("git ls-files exited with {}", output.status);
    }
    let mut paths: Vec<PathBuf> = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|bytes| !bytes.is_empty())
        .filter_map(|bytes| std::str::from_utf8(bytes).ok())
        .map(PathBuf::from)
        .collect();
    paths.sort();
    identify_files(&paths)
}

pub(crate) fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let file =
        File::open(path).with_context(|| format!("opening {} for hashing", path.display()))?;
    let mut reader = BufReader::with_capacity(128 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            buffered: 0,
            bytes: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.bytes = self.bytes.wrapping_add(input.len() as u64);
        if self.buffered != 0 {
            let take = (64 - self.buffered).min(input.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&input[..take]);
            self.buffered += take;
            input = &input[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64].try_into().expect("64-byte SHA-256 block");
            self.compress(block);
            input = &input[64..];
        }
        self.buffer[..input.len()].copy_from_slice(input);
        self.buffered = input.len();
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.bytes.wrapping_mul(8);
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffered..56].fill(0);
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut digest = [0u8; 32];
        for (index, word) in self.state.into_iter().enumerate() {
            let start = index * 4;
            digest[start..start + 4].copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut schedule = [0u32; 64];
        for (index, word) in schedule[..16].iter_mut().enumerate() {
            let start = index * 4;
            let bytes: [u8; 4] = block[start..start + 4]
                .try_into()
                .expect("four-byte SHA-256 word");
            *word = u32::from_be_bytes(bytes);
        }
        for i in 16..64 {
            let s0 = schedule[i - 15].rotate_right(7)
                ^ schedule[i - 15].rotate_right(18)
                ^ (schedule[i - 15] >> 3);
            let s1 = schedule[i - 2].rotate_right(17)
                ^ schedule[i - 2].rotate_right(19)
                ^ (schedule[i - 2] >> 10);
            schedule[i] = schedule[i - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choice)
                .wrapping_add(K[i])
                .wrapping_add(schedule[i]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_standard_known_answers() {
        assert_eq!(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn file_identity_detects_byte_changes() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "titan-provenance-{}-{}.bin",
            std::process::id(),
            super::super::unix_time_ms()
        ));
        std::fs::write(&path, b"before")?;
        let before = identify_file(&path)?;
        std::fs::write(&path, b"after")?;
        let after = identify_file(&path)?;
        assert_ne!(before.sha256, after.sha256);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
