#![forbid(unsafe_code)]

pub mod config;
mod init;
mod status;

use std::path::Path;

pub use init::{DEFAULT_CONFIG, InitError, InitResult};
pub use status::{StatusError, StatusReport, StatusRequest};

pub struct Application;

impl Application {
    pub fn init(config_path: &Path, force: bool) -> Result<InitResult, InitError> {
        init::create(config_path, force)
    }

    pub fn status(request: &StatusRequest) -> Result<StatusReport, StatusError> {
        status::inspect(request)
    }
}
