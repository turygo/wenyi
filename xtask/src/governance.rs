#![forbid(unsafe_code)]
mod changelog;
mod rust;
use cargo_metadata::MetadataCommand;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};
type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

pub(crate) fn architecture(root: &Path) -> Result<()> {
    let metadata = MetadataCommand::new()
        .current_dir(root)
        .no_deps()
        .other_options(vec!["--locked".into()])
        .exec()?;
    let members: BTreeMap<_, _> = metadata
        .workspace_packages()
        .into_iter()
        .map(|package| (package.name.to_string(), package))
        .collect();
    let mut graph = BTreeMap::new();
    for (name, package) in &members {
        let allowed = allowed_dependencies(name)?;
        let mut edges = BTreeSet::new();
        for dependency in &package.dependencies {
            if members.contains_key(&dependency.name) {
                if !allowed.contains(&dependency.name.as_str()) {
                    return Err(format!(
                        "undeclared workspace dependency: {name} -> {} ({:?}, target={:?})",
                        dependency.name, dependency.kind, dependency.target
                    )
                    .into());
                }
                edges.insert(dependency.name.clone());
            }
        }
        graph.insert(name.clone(), edges);
        for target in &package.targets {
            if target
                .kind
                .iter()
                .any(|kind| kind.to_string() == "custom-build")
                && name != "wenyi-book"
            {
                return Err(format!("owned build script is forbidden: {name}").into());
            }
            let source = fs::read_to_string(&target.src_path)?;
            let syntax = syn::parse_file(&source)?;
            if !syntax.attrs.iter().any(rust::forbids_unsafe) {
                return Err(format!(
                    "crate/test/build root must deny unsafe_code: {}",
                    target.src_path
                )
                .into());
            }
        }
    }
    check_cycles(&graph)?;
    let mut files = Vec::new();
    for directory in [root.join("crates"), root.join("xtask")] {
        collect(&directory, &mut files)?;
    }
    let mut types = BTreeMap::new();
    for path in files {
        rust::check(root, &path, &mut types)?;
    }
    for (name, size) in types {
        if size > 400 {
            return Err(format!("aggregate type budget exceeded: {name}: {size} > 400").into());
        }
    }
    println!("Rust architecture: dependencies, source boundaries, unsafe and budgets passed");
    Ok(())
}
fn allowed_dependencies(owner: &str) -> Result<&'static [&'static str]> {
    Ok(match owner {
        "wenyi-core" | "xtask" => &[],
        "wenyi-book" | "wenyi-store" => &["wenyi-core"],
        "wenyi-app" => &["wenyi-core", "wenyi-book", "wenyi-store"],
        "wenyi-cli" => &["wenyi-core", "wenyi-app"],
        _ => return Err(format!("undeclared workspace owner: {owner}").into()),
    })
}
fn check_cycles(graph: &BTreeMap<String, BTreeSet<String>>) -> Result<()> {
    fn visit<'a>(
        node: &'a str,
        graph: &'a BTreeMap<String, BTreeSet<String>>,
        active: &mut BTreeSet<&'a str>,
        done: &mut BTreeSet<&'a str>,
    ) -> Result<()> {
        if done.contains(node) {
            return Ok(());
        }
        if !active.insert(node) {
            return Err(format!("workspace dependency cycle at {node}").into());
        }
        for target in &graph[node] {
            visit(target, graph, active, done)?;
        }
        active.remove(node);
        done.insert(node);
        Ok(())
    }
    let mut active = BTreeSet::new();
    let mut done = BTreeSet::new();
    for node in graph.keys() {
        visit(node, graph, &mut active, &mut done)?;
    }
    Ok(())
}
fn collect(directory: &Path, result: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let kind = fs::symlink_metadata(&path)?.file_type();
        if kind.is_symlink() {
            return Err(format!("owned source symlink is not admitted: {}", path.display()).into());
        }
        if kind.is_dir() {
            collect(&path, result)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            result.push(path);
        }
    }
    Ok(())
}

pub(crate) fn changelog(root: &Path, args: &[String]) -> Result<()> {
    changelog::check(root, args)
}
