use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    Some(value.trim().to_string())
}

fn main() {
    for git_path in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git_output(&["rev-parse", "--git-path", git_path]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    for project_path in ["Cargo.toml", "README.md", "METRICS.md", "math.md"] {
        println!("cargo:rerun-if-changed={project_path}");
    }

    if let Some(symbolic_ref) =
        git_output(&["symbolic-ref", "-q", "HEAD"]).filter(|value| !value.is_empty())
    {
        if let Some(path) = git_output(&["rev-parse", "--git-path", &symbolic_ref]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    let commit = git_output(&["rev-parse", "--short=12", "HEAD"])
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    // `git diff --quiet` misses staged changes. A release built from a staged
    // but uncommitted tree must never claim to represent clean HEAD.
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .map(|output| !output.status.success() || !output.stdout.is_empty())
        .unwrap_or(true);

    println!("cargo:rustc-env=TITAN_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=TITAN_GIT_DIRTY={dirty}");
}
