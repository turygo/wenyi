use super::Result;
use std::{collections::BTreeSet, fs, path::Path, process::Command};

pub(super) fn check(root: &Path, args: &[String]) -> Result<()> {
    let mut changed = BTreeSet::new();
    if args.is_empty() {
        for arguments in [
            vec!["diff", "--name-only", "-z"],
            vec!["diff", "--cached", "--name-only", "-z"],
            vec!["ls-files", "--others", "--exclude-standard", "-z"],
        ] {
            changed.extend(paths(root, &arguments)?);
        }
    } else if args.len() == 4 && args[0] == "--base" && args[2] == "--head" {
        if args[1].starts_with('-') || args[3].starts_with('-') {
            return Err("commit references must not be options".into());
        }
        changed.extend(paths(
            root,
            &["diff", "--name-only", "-z", &args[1], &args[3], "--"],
        )?);
    } else {
        return Err("usage: cargo xtask changelog [--base REF --head REF]".into());
    }
    if changed.iter().any(|path| code_path(path)) && !changed.contains("CHANGELOG.md") {
        return Err("code, test or tooling changes require CHANGELOG.md in the same change".into());
    }
    let text = fs::read_to_string(root.join("CHANGELOG.md"))?;
    let unreleased = text
        .split("## [Unreleased]")
        .nth(1)
        .ok_or("CHANGELOG.md lacks [Unreleased]")?
        .split("\n## ")
        .next()
        .unwrap_or("");
    if !unreleased
        .lines()
        .any(|line| line.trim_start().starts_with("- "))
    {
        return Err("[Unreleased] requires a concrete change entry".into());
    }
    println!("Changelog: change range and [Unreleased] passed");
    Ok(())
}
fn paths(root: &Path, args: &[&str]) -> Result<Vec<String>> {
    let output = Command::new("git").current_dir(root).args(args).output()?;
    if !output.status.success() {
        return Err(format!("git failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    let text = String::from_utf8(output.stdout)?;
    Ok(text
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect())
}
fn code_path(path: &str) -> bool {
    [
        "crates/",
        "xtask/",
        "native/",
        "scripts/",
        "trans_novel/",
        "tests/",
        ".github/workflows/",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
        || matches!(
            path,
            "Cargo.toml"
                | "Cargo.lock"
                | "rust-toolchain.toml"
                | "rustfmt.toml"
                | "pyproject.toml"
                | "uv.lock"
                | ".pre-commit-config.yaml"
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_and_target_gates_cover_rust_and_retained_python_changes() {
        for path in [
            "crates/wenyi-book/src/ingest.rs",
            "xtask/src/main.rs",
            "native/sources.lock.json",
            "Cargo.lock",
            ".github/workflows/rust.yml",
            "trans_novel/config.py",
            "tests/pipeline/test_application.py",
        ] {
            assert!(code_path(path));
        }
        for path in [
            "docs/rust-r1-spec.md",
            "CHANGELOG.md",
            "state/private/run.json",
        ] {
            assert!(!code_path(path));
        }
    }
}
