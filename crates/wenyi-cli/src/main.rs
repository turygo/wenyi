#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::{Arg, Command, value_parser};

fn command() -> Command {
    Command::new("wenyi")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Native CLI for R0 help, version, and config-path arguments.")
        .arg_required_else_help(true)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("PATH")
                .value_parser(value_parser!(PathBuf))
                .default_value("config.yaml"),
        )
}

fn main() {
    let mut cli = command();
    let matches = cli.get_matches_mut();
    let _config = matches
        .get_one::<PathBuf>("config")
        .expect("the config argument has a default value");
    if let Err(error) = cli.print_help() {
        cli.error(clap::error::ErrorKind::Io, error).exit();
    }
    std::process::exit(2);
}
