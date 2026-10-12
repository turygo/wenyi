use crate::cache::{self, Result};
use serde_json::{Value, json};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const ICU_CPPFLAGS: &str = "-DU_STATIC_IMPLEMENTATION -DUCONFIG_NO_COLLATION=1 -DUCONFIG_NO_FORMATTING=1 -DUCONFIG_NO_TRANSLITERATION=1 -DUCONFIG_NO_BREAK_ITERATION=1 -DUCONFIG_NO_REGEX=1 -DICU_NO_USER_DATA_OVERRIDE=1";

pub fn build() -> Result<()> {
    let cache_dir = env::var_os("WENYI_NATIVE_CACHE")
        .ok_or("WENYI_NATIVE_CACHE is required; run cargo xtask native provision --cache DIR")?;
    let lock = cache::lock(include_bytes!("../../../native/sources.lock.json"))?;
    let cache_dir = Path::new(&cache_dir);
    for item in &lock.components {
        println!(
            "cargo:rerun-if-changed={}",
            cache_dir
                .join("archives")
                .join(&item.archive.filename)
                .display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            item.source(cache_dir).display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            cache_dir
                .join("manifests")
                .join(format!("{}.json", item.key()))
                .display()
        );
    }
    let sources = lock
        .components
        .iter()
        .map(|item| cache::verify(cache_dir, item))
        .collect::<Result<Vec<_>>>()?;
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR missing")?);
    let target = env::var("TARGET")?;
    let host = env::var("HOST")?;
    if target != host {
        return Err("native R1 builds require runner-native HOST=TARGET tooling".into());
    }
    let windows = target.contains("windows-msvc");
    if target.contains("windows") && !windows {
        return Err("Windows native boundary requires MSVC".into());
    }
    let prefix = out.join("native-prefix");
    fs::create_dir_all(&prefix)?;
    let cmake = env::var_os("CMAKE").unwrap_or_else(|| "cmake".into());
    let version = Command::new(&cmake).arg("--version").output()?;
    if !version.status.success()
        || String::from_utf8_lossy(&version.stdout).lines().next() != Some("cmake version 4.1.3")
    {
        return Err("native build requires CMake 4.1.3 (set CMAKE to isolated executable)".into());
    }
    let mut records = Vec::new();
    records.push(json!({"cmake":cmake,"version":String::from_utf8_lossy(&version.stdout)}));
    let icu = &sources[1];
    icu_build(icu, &out, &prefix, windows, &mut records)?;
    let icu_uc = static_library(&prefix, if windows { "sicuuc" } else { "icuuc" }, windows)?;
    let icu_data = static_library(&prefix, if windows { "sicudt" } else { "icudata" }, windows)?;
    parser_builds(
        &cmake,
        &sources,
        &out,
        &prefix,
        &icu_uc,
        &icu_data,
        &mut records,
    )?;
    let xml_name = if windows { "libxml2" } else { "xml2" };
    let xml = static_library(&prefix, xml_name, windows)?;
    let expat = static_library(&prefix, if windows { "libexpat" } else { "expat" }, windows)?;
    compile_glue(&prefix, windows, &mut records)?;
    let libraries = [
        (&xml, xml_name),
        (&expat, if windows { "libexpat" } else { "expat" }),
        (&icu_uc, if windows { "sicuuc" } else { "icuuc" }),
        (&icu_data, if windows { "sicudt" } else { "icudata" }),
    ];
    link_libraries(&libraries, &target, windows)?;
    record_output(&out, &lock, &sources, &libraries, &target, records)?;
    Ok(())
}

fn parser_builds(
    cmake: &std::ffi::OsStr,
    sources: &[PathBuf],
    out: &Path,
    prefix: &Path,
    icu_uc: &Path,
    icu_data: &Path,
    records: &mut Vec<Value>,
) -> Result<()> {
    let expat_flags = [
        "-DEXPAT_SHARED_LIBS=OFF",
        "-DEXPAT_BUILD_TOOLS=OFF",
        "-DEXPAT_BUILD_EXAMPLES=OFF",
        "-DEXPAT_BUILD_TESTS=OFF",
        "-DEXPAT_BUILD_DOCS=OFF",
        "-DEXPAT_BUILD_FUZZERS=OFF",
        "-DEXPAT_BUILD_PKGCONFIG=OFF",
        "-DEXPAT_DTD=ON",
        "-DEXPAT_GE=ON",
        "-DEXPAT_NS=ON",
        "-DEXPAT_CHAR_TYPE=char",
        "-DEXPAT_RELEASE_POSTFIX=",
    ];
    cmake_build(
        cmake,
        &sources[0],
        &out.join("expat-build"),
        prefix,
        &expat_flags,
        records,
    )?;
    let mut xml_flags = vec![
        "-DLIBXML2_WITH_ICU=ON".to_owned(),
        format!("-DICU_INCLUDE_DIR={}", prefix.join("include").display()),
        format!("-DICU_UC_LIBRARY_RELEASE={}", icu_uc.display()),
        format!("-DICU_DATA_LIBRARY_RELEASE={}", icu_data.display()),
        "-DCMAKE_C_FLAGS=-DU_STATIC_IMPLEMENTATION -DLIBXML_STATIC".to_owned(),
    ];
    for feature in [
        "CATALOG",
        "DEBUG",
        "DOCS",
        "HTTP",
        "ICONV",
        "ISO8859X",
        "LEGACY",
        "MODULES",
        "PROGRAMS",
        "PYTHON",
        "READLINE",
        "TESTS",
        "XINCLUDE",
        "ZLIB",
        "C14N",
        "PATTERN",
        "READER",
        "REGEXPS",
        "RELAXNG",
        "SAX1",
        "SCHEMAS",
        "SCHEMATRON",
        "VALID",
        "WRITER",
        "XPATH",
        "XPTR",
    ] {
        xml_flags.push(format!("-DLIBXML2_WITH_{feature}=OFF"));
    }
    for feature in ["HTML", "OUTPUT", "PUSH", "THREADS"] {
        xml_flags.push(format!("-DLIBXML2_WITH_{feature}=ON"));
    }
    let xml_refs = xml_flags.iter().map(String::as_str).collect::<Vec<_>>();
    cmake_build(
        cmake,
        &sources[2],
        &out.join("xml-build"),
        prefix,
        &xml_refs,
        records,
    )
}

fn compile_glue(prefix: &Path, windows: bool, records: &mut Vec<Value>) -> Result<()> {
    let mut glue = cc::Build::new();
    glue.file("native/xml_bridge.c")
        .file("native/xml_limits.c")
        .file("native/xml_serialize.c")
        .file("native/expat_bridge.c")
        .file("native/encoding_bridge.c")
        .include(prefix.join("include"))
        .include(prefix.join("include/libxml2"))
        .define("U_STATIC_IMPLEMENTATION", None)
        .define("LIBXML_STATIC", None)
        .define("XML_STATIC", None)
        .define("XML_DTD", None)
        .std("c11");
    let compiler = glue.get_compiler();
    let compiler_version = compiler
        .to_command()
        .arg(if windows { "/Bv" } else { "--version" })
        .output()?;
    records.push(json!({"glue_compiler":format!("{compiler:?}"), "version_stdout":String::from_utf8_lossy(&compiler_version.stdout),
        "version_stderr":String::from_utf8_lossy(&compiler_version.stderr)}));
    glue.compile("wenyi_xml_bridge");
    Ok(())
}

fn link_libraries(libraries: &[(&PathBuf, &str)], target: &str, windows: bool) -> Result<()> {
    for (path, name) in libraries {
        println!(
            "cargo:rustc-link-search=native={}",
            path.parent().ok_or("invalid library path")?.display()
        );
        println!("cargo:rustc-link-lib=static={name}");
    }
    if windows {
        for library in ["advapi32", "user32", "bcrypt"] {
            println!("cargo:rustc-link-lib={library}");
        }
    } else {
        println!(
            "cargo:rustc-link-lib={}",
            if target.contains("apple") {
                "c++"
            } else {
                "stdc++"
            }
        );
        for library in ["m", "pthread"] {
            println!("cargo:rustc-link-lib={library}");
        }
        if target.contains("linux") {
            println!("cargo:rustc-link-lib=dl");
        }
    }
    Ok(())
}

fn record_output(
    out: &Path,
    lock: &cache::SourceLock,
    sources: &[PathBuf],
    linked: &[(&PathBuf, &str)],
    target: &str,
    records: Vec<Value>,
) -> Result<()> {
    let mut libraries = Vec::new();
    for (library, _) in linked {
        let (size, digest) = cache::hash_file(library, 512 * 1024 * 1024)?;
        libraries.push(json!({"path":library,"size_bytes":size,"sha256":digest}));
    }
    let bridge = out.join(if target.contains("windows-msvc") {
        "wenyi_xml_bridge.lib"
    } else {
        "libwenyi_xml_bridge.a"
    });
    let (size, digest) = cache::hash_file(&bridge, 512 * 1024 * 1024)?;
    libraries.push(json!({"path":bridge,"size_bytes":size,"sha256":digest}));
    let notice_dir = out.join("native-notices");
    fs::create_dir_all(&notice_dir)?;
    for (item, source) in lock.components.iter().zip(sources) {
        for (index, notice) in item.notices.iter().enumerate() {
            let relative = cache::safe_relative(notice)?;
            let input = source.join(relative);
            let output = notice_dir.join(format!("{}-{index}.txt", item.name));
            if input.extension() == Some(std::ffi::OsStr::new("c")) {
                fs::write(output, source_notice(&input)?)?;
            } else {
                fs::copy(input, output)?;
            }
        }
    }
    let mut sdk = [
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "WindowsSDKVersion",
        "VCToolsVersion",
    ]
    .into_iter()
    .map(|name| {
        (
            name.to_owned(),
            Value::String(env::var(name).unwrap_or_default()),
        )
    })
    .collect::<serde_json::Map<String, Value>>();
    if target.contains("apple") {
        for option in ["--show-sdk-path", "--show-sdk-version"] {
            let result = Command::new("xcrun").arg(option).output()?;
            if !result.status.success() {
                return Err(format!("cannot record native SDK: xcrun {option}").into());
            }
            sdk.insert(
                option.into(),
                Value::String(String::from_utf8(result.stdout)?.trim().to_owned()),
            );
        }
    }
    fs::write(
        out.join("native-build.json"),
        serde_json::to_vec_pretty(
            &json!({"schema_version":1,"target":target,"commands":records,
        "sdk":sdk,"libraries":libraries,"notices":notice_dir,"source_lock_sha256":cache::hash_file(Path::new("../../native/sources.lock.json"), 1024*1024)?.1}),
        )?,
    )?;
    Ok(())
}

fn source_notice(path: &Path) -> Result<Vec<u8>> {
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(fs::File::open(path)?);
    let mut notice = Vec::new();
    loop {
        if reader.read_until(b'\n', &mut notice)? == 0 {
            return Err(format!("incomplete source license notice: {}", path.display()).into());
        }
        if !notice.starts_with(b"/*") {
            return Err(format!(
                "source license notice is not a leading comment: {}",
                path.display()
            )
            .into());
        }
        if let Some(end) = notice.windows(2).position(|bytes| bytes == b"*/") {
            notice.truncate(end + 2);
            notice.push(b'\n');
            return Ok(notice);
        }
    }
}

fn run(command: &mut Command, records: &mut Vec<Value>) -> Result<()> {
    records.push(json!({"command":format!("{command:?}")}));
    let status = command.status()?;
    if !status.success() {
        return Err(format!("native command failed ({status}): {command:?}").into());
    }
    Ok(())
}

fn static_library(prefix: &Path, name: &str, windows: bool) -> Result<PathBuf> {
    let filename = if windows {
        format!("{name}.lib")
    } else {
        format!("lib{name}.a")
    };
    let path = prefix.join("lib").join(filename);
    if !path.is_file() {
        return Err(format!("required pinned static library missing: {}", path.display()).into());
    }
    Ok(path)
}

fn cmake_build(
    cmake: &std::ffi::OsStr,
    source: &Path,
    build: &Path,
    prefix: &Path,
    flags: &[&str],
    records: &mut Vec<Value>,
) -> Result<()> {
    let mut configure = Command::new(cmake);
    configure
        .arg("-S")
        .arg(source)
        .arg("-B")
        .arg(build)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
            "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
        ])
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()))
        .args(flags);
    run(&mut configure, records)?;
    run(
        Command::new(cmake)
            .arg("--build")
            .arg(build)
            .args(["--config", "Release", "--parallel"])
            .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "1".into())),
        records,
    )?;
    run(
        Command::new(cmake)
            .arg("--install")
            .arg(build)
            .args(["--config", "Release"]),
        records,
    )?;
    let info = build.join("CMakeCache.txt");
    records.push(json!({
        "cmake_cache":fs::read_to_string(info)?,
        "cmake_c_compiler":fs::read_to_string(build.join("CMakeFiles/4.1.3/CMakeCCompiler.cmake"))?,
    }));
    Ok(())
}

fn icu_build(
    source: &Path,
    out: &Path,
    prefix: &Path,
    windows: bool,
    records: &mut Vec<Value>,
) -> Result<()> {
    let build = out.join("icu-build");
    fs::create_dir_all(&build)?;
    if !source.join("source/data/in/icudt78l.dat").is_file() {
        return Err("pinned ICU source must contain its bundled icudt78l.dat; Python data generation is not admitted".into());
    }
    let platform = if windows {
        "Cygwin/MSVC"
    } else if env::var("TARGET")?.contains("apple") {
        "MacOSX"
    } else {
        "Linux"
    };
    let bash = env::var_os("BASH").unwrap_or_else(|| "bash".into());
    let make = env::var_os("MAKE").unwrap_or_else(|| "make".into());
    let (script, prefix_arg) = if windows {
        let convert = |path: &Path| -> Result<String> {
            let result = Command::new("cygpath").arg("-u").arg(path).output()?;
            if !result.status.success() {
                return Err("cygpath failed; configure Cygwin/MSVC tool environment".into());
            }
            Ok(String::from_utf8(result.stdout)?.trim().to_owned())
        };
        (
            convert(&source.join("source/runConfigureICU"))?,
            convert(prefix)?,
        )
    } else {
        (
            source
                .join("source/runConfigureICU")
                .to_string_lossy()
                .into_owned(),
            prefix.to_string_lossy().into_owned(),
        )
    };
    let mut configure = Command::new(bash);
    configure
        .current_dir(&build)
        .arg(script)
        .arg(platform)
        .arg(format!("--prefix={prefix_arg}"))
        .args([
            "--enable-static",
            "--disable-shared",
            "--with-data-packaging=static",
            "--disable-tests",
            "--disable-samples",
            "--disable-extras",
            "--disable-icuio",
            "--disable-dyload",
            "--disable-layout",
        ])
        .env("CPPFLAGS", ICU_CPPFLAGS)
        .env("ac_cv_prog_PYTHON", "");
    if windows {
        configure
            .env("CFLAGS", "-O2 -std:c11")
            .env("CXXFLAGS", "-O2 -std:c++17");
    } else {
        let mut cxx_flags = String::from("-O2 -fPIC -std=c++17");
        if platform == "MacOSX" {
            let sdk = Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()?;
            if !sdk.status.success() {
                return Err("cannot locate the selected macOS SDK".into());
            }
            let sdk = String::from_utf8(sdk.stdout)?.trim().to_owned();
            let headers = Path::new(&sdk).join("usr/include/c++/v1");
            if headers.join("memory").is_file() {
                // 使用当前 SDK 的标准 C++ 头文件，不发现或链接系统 XML 库。
                cxx_flags.push_str(&format!(" -isystem \"{}\"", headers.display()));
            }
        }
        configure
            .env("CFLAGS", "-O2 -fPIC")
            .env("CXXFLAGS", cxx_flags);
    }
    run(&mut configure, records)?;
    run(
        Command::new(&make)
            .current_dir(&build)
            .arg(
                "LOCAL_SUBDIRS=stubdata common i18n tools/toolutil tools/icupkg tools/pkgdata data",
            )
            .arg(format!(
                "-j{}",
                env::var("NUM_JOBS").unwrap_or_else(|_| "1".into())
            )),
        records,
    )?;
    run(
        Command::new(&make)
            .current_dir(&build)
            .arg("LOCAL_SUBDIRS=common data")
            .arg("install"),
        records,
    )?;
    records.push(json!({"icu_config_log":fs::read_to_string(build.join("config.log"))?}));
    Ok(())
}
