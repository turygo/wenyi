#![forbid(unsafe_code)]
#[path = "../../../crates/wenyi-book/native/cache.rs"]
mod cache;
mod provision;

pub fn run(args: Vec<String>) -> cache::Result<()> {
    if args.len() != 4 || args[0] != "native" || args[1] != "provision" || args[2] != "--cache" {
        return Err("usage: cargo xtask native provision --cache DIR".into());
    }
    let lock = cache::lock(include_bytes!("../../../native/sources.lock.json"))?;
    provision::provision(std::path::Path::new(&args[3]), &lock)
}
