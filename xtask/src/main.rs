#![forbid(unsafe_code)]
mod governance;
mod native;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let result = match args.first().map(String::as_str) {
        Some("architecture") if args.len() == 1 => governance::architecture(root),
        Some("changelog") => governance::changelog(root, &args[1..]),
        _ => native::run(args),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
