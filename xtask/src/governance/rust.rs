use super::Result;
use proc_macro2::{Span, TokenStream, TokenTree};
use std::{collections::BTreeMap, fs, path::Path};
use syn::{
    Item, UseTree,
    spanned::Spanned,
    visit::{self, Visit},
};

pub(super) fn forbids_unsafe(attribute: &syn::Attribute) -> bool {
    match &attribute.meta {
        syn::Meta::List(list) if list.path.is_ident("forbid") || list.path.is_ident("deny") => list
            .tokens
            .clone()
            .into_iter()
            .any(|token| matches!(token, TokenTree::Ident(name) if name == "unsafe_code")),
        _ => false,
    }
}

pub(super) fn check(root: &Path, path: &Path, types: &mut BTreeMap<String, usize>) -> Result<()> {
    let relative = path.strip_prefix(root)?;
    let source = fs::read_to_string(path)?;
    budget(relative, "file", source.lines().count(), 500, 800)?;
    let file = syn::parse_file(&source)?;
    let owner = relative
        .components()
        .nth(if relative.starts_with("crates") { 1 } else { 0 })
        .ok_or("missing owner")?
        .as_os_str()
        .to_string_lossy()
        .into_owned();
    let ffi = relative == Path::new("crates/wenyi-book/src/source/xml_ffi.rs");
    let mut checker = Checker {
        path: relative,
        owner,
        ffi,
        failures: Vec::new(),
        types,
    };
    checker.visit_file(&file);
    if !checker.failures.is_empty() {
        return Err(checker.failures.join("\n").into());
    }
    Ok(())
}
fn lines(span: Span) -> usize {
    span.end().line.saturating_sub(span.start().line) + 1
}
fn budget(path: &Path, kind: &str, size: usize, warning: usize, hard: usize) -> Result<()> {
    if size > hard {
        return Err(format!(
            "{}: {kind} budget exceeded: {size} > {hard}",
            path.display()
        )
        .into());
    }
    if size > warning {
        eprintln!("warning: {}: {kind} {size} > {warning}", path.display());
    }
    Ok(())
}
struct Checker<'a> {
    path: &'a Path,
    owner: String,
    ffi: bool,
    failures: Vec<String>,
    types: &'a mut BTreeMap<String, usize>,
}
impl Checker<'_> {
    fn violation(&mut self, text: impl Into<String>) {
        self.failures
            .push(format!("{}: {}", self.path.display(), text.into()));
    }
    fn unsafe_code(&mut self) {
        if !self.ffi {
            self.violation("unsafe Rust is confined to book/source/xml_ffi.rs");
        }
    }
    fn budget(&mut self, kind: &str, span: Span, warning: usize, hard: usize) {
        if let Err(error) = budget(self.path, kind, lines(span), warning, hard) {
            self.failures.push(error.to_string());
        }
    }
    fn aggregate(&mut self, name: &str, span: Span) {
        *self
            .types
            .entry(format!("{}::{name}", self.owner))
            .or_default() += lines(span);
    }
}
impl<'ast> Visit<'ast> for Checker<'_> {
    fn visit_item(&mut self, item: &'ast Item) {
        match item {
            Item::Struct(value) => self.aggregate(&value.ident.to_string(), value.span()),
            Item::Enum(value) => self.aggregate(&value.ident.to_string(), value.span()),
            Item::Trait(value) => {
                self.aggregate(&value.ident.to_string(), value.span());
                if value.unsafety.is_some() {
                    self.unsafe_code();
                }
            }
            Item::ForeignMod(value) if value.unsafety.is_some() => self.unsafe_code(),
            _ => {}
        }
        visit::visit_item(self, item);
    }
    fn visit_item_fn(&mut self, value: &'ast syn::ItemFn) {
        self.budget("function", value.span(), 80, 120);
        if value.sig.unsafety.is_some() {
            self.unsafe_code();
        }
        visit::visit_item_fn(self, value);
    }
    fn visit_impl_item_fn(&mut self, value: &'ast syn::ImplItemFn) {
        self.budget("method", value.span(), 80, 120);
        if value.sig.unsafety.is_some() {
            self.unsafe_code();
        }
        visit::visit_impl_item_fn(self, value);
    }
    fn visit_item_impl(&mut self, value: &'ast syn::ItemImpl) {
        self.budget("impl", value.span(), 250, 400);
        if let syn::Type::Path(path) = value.self_ty.as_ref()
            && let Some(name) = path.path.segments.last()
        {
            self.aggregate(&name.ident.to_string(), value.span());
        }
        if value.unsafety.is_some() {
            self.unsafe_code();
        }
        visit::visit_item_impl(self, value);
    }
    fn visit_expr_unsafe(&mut self, value: &'ast syn::ExprUnsafe) {
        self.unsafe_code();
        visit::visit_expr_unsafe(self, value);
    }
    fn visit_macro(&mut self, value: &'ast syn::Macro) {
        if value.path.is_ident("include") {
            self.violation("generated or included Rust application code is forbidden");
        }
        if has_unsafe(value.tokens.clone()) {
            self.unsafe_code();
        }
        visit::visit_macro(self, value);
    }
    fn visit_attribute(&mut self, value: &'ast syn::Attribute) {
        if let syn::Meta::List(list) = &value.meta
            && list.path.is_ident("allow")
            && has_name(list.tokens.clone(), "unsafe_code")
            && !self.ffi
        {
            self.violation("unsafe_code allowance outside the FFI boundary");
        }
        visit::visit_attribute(self, value);
    }
    fn visit_item_use(&mut self, value: &'ast syn::ItemUse) {
        let mut paths = Vec::new();
        use_paths(&value.tree, &mut Vec::new(), &mut paths);
        if !matches!(value.vis, syn::Visibility::Inherited)
            && paths
                .iter()
                .any(|path| path.last().is_some_and(|name| name == "*"))
        {
            self.violation("wildcard exports are forbidden");
        }
        if self.path.starts_with("crates/wenyi-book/src/source") {
            for path in paths {
                if path.first().is_some_and(|name| name == "crate")
                    && path.get(1).is_some_and(|name| {
                        matches!(name.as_str(), "css" | "ingest" | "style_projection")
                    })
                {
                    self.violation("neutral source cannot depend on CSS or ingestion");
                }
            }
        }
        visit::visit_item_use(self, value);
    }
    fn visit_path(&mut self, path: &'ast syn::Path) {
        let mut segments = path.segments.iter();
        if self.path.starts_with("crates/wenyi-book/src/source")
            && segments
                .next()
                .is_some_and(|segment| segment.ident == "crate")
            && segments.next().is_some_and(|segment| {
                matches!(
                    segment.ident.to_string().as_str(),
                    "css" | "ingest" | "style_projection"
                )
            })
        {
            self.violation("neutral source cannot use qualified CSS or ingestion paths");
        }
        visit::visit_path(self, path);
    }
}
fn has_name(tokens: TokenStream, name: &str) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(value) => value == name,
        TokenTree::Group(group) => has_name(group.stream(), name),
        _ => false,
    })
}
fn has_unsafe(tokens: TokenStream) -> bool {
    has_name(tokens, "unsafe")
}
fn use_paths(tree: &UseTree, prefix: &mut Vec<String>, result: &mut Vec<Vec<String>>) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            use_paths(&path.tree, prefix, result);
            prefix.pop();
        }
        UseTree::Group(group) => {
            for item in &group.items {
                use_paths(item, prefix, result);
            }
        }
        UseTree::Name(name) => {
            let mut path = prefix.clone();
            path.push(name.ident.to_string());
            result.push(path);
        }
        UseTree::Rename(name) => {
            let mut path = prefix.clone();
            path.push(name.ident.to_string());
            result.push(path);
        }
        UseTree::Glob(_) => {
            let mut path = prefix.clone();
            path.push("*".into());
            result.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(path: &str, source: &str) -> Result<()> {
        let directory = tempfile::tempdir()?;
        let file = directory.path().join(path);
        fs::create_dir_all(file.parent().unwrap())?;
        fs::write(&file, source)?;
        check(directory.path(), &file, &mut BTreeMap::new())
    }

    #[test]
    fn target_cfg_and_macro_tokens_cannot_escape_the_unsafe_boundary() {
        assert!(
            input(
                "crates/wenyi-core/src/lib.rs",
                "#[cfg(windows)] unsafe fn platform() {}"
            )
            .is_err()
        );
        assert!(
            input(
                "crates/wenyi-core/src/lib.rs",
                "macro_rules! hidden { () => { unsafe { operation() } }; }"
            )
            .is_err()
        );
        input(
            "crates/wenyi-book/src/source/xml_ffi.rs",
            "unsafe extern \"C\" { fn native(); }",
        )
        .unwrap();
        input(
            "crates/wenyi-core/src/lib.rs",
            "fn message() { println!(\"unsafe text is data\"); }",
        )
        .unwrap();
    }

    #[test]
    fn aliases_and_qualified_calls_cannot_reverse_neutral_source_dependencies() {
        assert!(
            input(
                "crates/wenyi-book/src/source/check.rs",
                "use crate::css::SourceCss as Neutral;"
            )
            .is_err()
        );
        assert!(
            input(
                "crates/wenyi-book/src/source/check.rs",
                "fn reverse() { crate::ingest::admit_epub(); }"
            )
            .is_err()
        );
        input(
            "crates/wenyi-book/src/css/check.rs",
            "use crate::source::SourceTree;",
        )
        .unwrap();
        assert!(input("crates/wenyi-core/src/lib.rs", "pub use hidden::*;").is_err());
        assert!(
            input(
                "crates/wenyi-core/src/lib.rs",
                "include!(\"generated.rs\");"
            )
            .is_err()
        );
    }
}
