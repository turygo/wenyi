use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use crate::config::expand_home;

pub const DEFAULT_CONFIG: &str = include_str!("../assets/config.example.yaml");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitResult {
    Created { path: PathBuf },
    Existing { path: PathBuf },
}
#[derive(Debug)]
pub struct InitError {
    pub path: PathBuf,
    pub detail: String,
}
impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "config_write ({}): {}", self.path.display(), self.detail)
    }
}
impl std::error::Error for InitError {}

pub(crate) fn create(path: &Path, force: bool) -> Result<InitResult, InitError> {
    let target = expand_home(path).map_err(|error| InitError {
        path: path.into(),
        detail: error.to_string(),
    })?;
    let parent = target
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let io_error = |error: std::io::Error| InitError {
        path: target.clone(),
        detail: error.to_string(),
    };
    fs::create_dir_all(parent).map_err(io_error)?;
    let mut options = OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    let mut stream = match options.open(&target) {
        Ok(stream) => stream,
        Err(error) if !force && error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Ok(InitResult::Existing { path: target });
        }
        Err(error) => return Err(io_error(error)),
    };
    stream
        .write_all(DEFAULT_CONFIG.as_bytes())
        .map_err(io_error)?;
    Ok(InitResult::Created { path: target })
}
