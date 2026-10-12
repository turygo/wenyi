#![forbid(unsafe_code)]

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use std::path::{Path, PathBuf};
use wenyi_app::{Application, InitResult, StatusReport, StatusRequest};

fn command() -> Command {
    Command::new("wenyi")
        .version(env!("CARGO_PKG_VERSION"))
        .about("EPUB source admission, configuration and read-only run status.")
        .arg_required_else_help(true)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .global(true)
                .value_name("PATH")
                .value_parser(value_parser!(PathBuf))
                .default_value("config.yaml"),
        )
        .subcommand(
            Command::new("init")
                .about("Create the default configuration.")
                .arg(Arg::new("force").long("force").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("status")
                .about("Inspect EPUB run progress without changing files.")
                .arg(
                    Arg::new("input")
                        .value_name("INPUT")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                ),
        )
}

fn main() {
    let mut cli = command();
    let matches = cli.get_matches_mut();
    let config = matches
        .get_one::<PathBuf>("config")
        .expect("config has a default");
    let code = match matches.subcommand() {
        Some(("init", arguments)) => init(config, arguments.get_flag("force")),
        Some(("status", arguments)) => status(config, arguments),
        _ => {
            if let Err(error) = cli.print_help() {
                cli.error(clap::error::ErrorKind::Io, error).exit();
            }
            2
        }
    };
    std::process::exit(code);
}

fn init(config: &Path, force: bool) -> i32 {
    match Application::init(config, force) {
        Ok(InitResult::Created { path }) => {
            println!("Created configuration: {}", path.display());
            0
        }
        Ok(InitResult::Existing { path }) => {
            eprintln!(
                "Configuration already exists: {} (use --force to replace it)",
                path.display()
            );
            1
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

fn status(config: &Path, arguments: &ArgMatches) -> i32 {
    let root = match std::env::current_dir() {
        Ok(root) => root,
        Err(error) => {
            eprintln!("Cannot determine working directory: {error}");
            return 1;
        }
    };
    let request = StatusRequest {
        input: arguments
            .get_one::<PathBuf>("input")
            .expect("input is required")
            .clone(),
        config_path: config.to_path_buf(),
        state_root: root.join("state"),
    };
    match Application::status(&request) {
        Ok(report) => {
            print_status(&report);
            report.exit_code()
        }
        Err(error) => {
            eprintln!("{error}");
            error.exit_code()
        }
    }
}

fn print_status(report: &StatusReport) {
    println!(
        "{} (epub): {}",
        report.source.identity.title,
        report.outcome_label()
    );
    let observed = &report.observation;
    if let (Some(source), Some(target)) = (&observed.source_lang, &observed.target_lang) {
        println!("Languages: {source} -> {target}");
    }
    for chapter in &observed.chapters {
        println!(
            "Chapter {}: {} [{:?}]",
            chapter.index, chapter.title, chapter.progress.status
        );
    }
    match observed.glossary {
        Some(counts) => println!(
            "Glossary: terms={}, open_conflicts={}",
            counts.terms, counts.open_conflicts
        ),
        None => println!("Glossary: unavailable"),
    }
    for diagnostic in &observed.diagnostics {
        println!("{}: {}", diagnostic.code, diagnostic.detail);
    }
    for diagnostic in &report.source.diagnostics {
        println!(
            "Source {} [{}]: {}",
            diagnostic.code, diagnostic.resource, diagnostic.detail
        );
    }
}
