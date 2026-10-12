#![forbid(unsafe_code)]

#[path = "native/cache.rs"]
mod cache;
#[path = "native/build.rs"]
mod native_build;

fn main() {
    for name in [
        "WENYI_NATIVE_CACHE",
        "CMAKE",
        "BASH",
        "MAKE",
        "CC",
        "CXX",
        "SDKROOT",
        "WindowsSDKVersion",
        "VCToolsVersion",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=../../native/sources.lock.json");
    println!("cargo:rerun-if-changed=native");
    if let Err(error) = native_build::build() {
        panic!("pinned native build: {error}");
    }
}
