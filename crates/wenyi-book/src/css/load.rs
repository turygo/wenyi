use super::{CssAsset, Rule, SourceCss, error, select::Selection};
use crate::archive::Archive;
use crate::error::BookError;
use crate::references::resolve_href;
use crate::source::{ConversionPolicy, NodeId, SourceTree, convert_encoding};
use lightningcss::declaration::DeclarationBlock;
use lightningcss::rules::{CssRule, CssRuleList};
use lightningcss::stylesheet::{ParserOptions, PrinterOptions, StyleSheet};
use lightningcss::targets::{Features, Targets};
use lightningcss::traits::ToCss;
use static_self::IntoOwned;
use std::collections::{HashMap, HashSet};

const MAX_DEPTH: usize = 64;
const MAX_MEMBER: usize = 512 * 1024 * 1024;
const MAX_TOTAL: usize = 4 * 1024 * 1024 * 1024;
struct Loader<'a> {
    archive: &'a mut Archive,
    output: SourceCss,
    active: HashSet<String>,
    asset_ids: HashSet<String>,
    admitted_bytes: usize,
    admitted_rules: usize,
}
pub(super) fn load(
    archive: &mut Archive,
    tree: &SourceTree,
    resource: &str,
) -> Result<SourceCss, BookError> {
    let mut loader = Loader {
        archive,
        output: SourceCss {
            assets: Vec::new(),
            rules: Vec::new(),
            conditions: Vec::new(),
            inline: Vec::new(),
        },
        active: HashSet::new(),
        asset_ids: HashSet::new(),
        admitted_bytes: 0,
        admitted_rules: 0,
    };
    let base = tree.nodes.iter().enumerate().find_map(|(index, _)| {
        let node = NodeId::new(index).ok()?;
        (tree.local_name(node) == Some("base"))
            .then(|| tree.attribute(node, "href"))
            .flatten()
    });
    let base = base.map_or_else(|| Ok(resource.to_owned()), |href| local(resource, href));
    let base = base?;
    for (index, _) in tree.nodes.iter().enumerate() {
        let node = NodeId::new(index)?;
        loader.source_node(tree, node, &base, resource)?;
    }
    Ok(loader.output)
}
impl Loader<'_> {
    fn source_node(
        &mut self,
        tree: &SourceTree,
        node: NodeId,
        base: &str,
        resource: &str,
    ) -> Result<(), BookError> {
        if tree.local_name(node).is_none() {
            return Ok(());
        }
        if let Some(style) = tree.attribute(node, "style") {
            let block = DeclarationBlock::parse_string(style, ParserOptions::default())
                .map_err(|cause| error(resource, "css-inline-syntax", format!("{cause:?}")))?
                .into_owned();
            self.output.inline.push((node, block));
        }
        let tag = tree.local_name(node).unwrap_or("");
        if tag == "link"
            && tree.attribute(node, "rel").is_some_and(|rel| {
                rel.split_ascii_whitespace()
                    .any(|word| word.eq_ignore_ascii_case("stylesheet"))
            })
        {
            let href = tree
                .attribute(node, "href")
                .ok_or_else(|| error(resource, "css-reference", "stylesheet link has no href"))?;
            let conditions = self.media_attribute(tree.attribute(node, "media"), resource)?;
            if let Some(conditions) = conditions {
                self.external(&local(base, href)?, &conditions, 0)?;
            }
        } else if tag == "style" {
            if tree
                .attribute(node, "type")
                .is_some_and(|value| !value.eq_ignore_ascii_case("text/css"))
            {
                return Ok(());
            }
            let text = tree.itertext(node)?;
            let conditions = self.media_attribute(tree.attribute(node, "media"), resource)?;
            if let Some(conditions) = conditions {
                self.sheet(&text, base, &conditions, 0)?;
            }
        }
        Ok(())
    }
    fn media_attribute(
        &mut self,
        media: Option<&str>,
        resource: &str,
    ) -> Result<Option<Vec<usize>>, BookError> {
        let Some(media) = media.filter(|value| !value.trim().is_empty()) else {
            return Ok(Some(Vec::new()));
        };
        let text = format!("@media {media} {{}}");
        let sheet = StyleSheet::parse(&text, ParserOptions::default())
            .map_err(|cause| error(resource, "css-media-syntax", cause.to_string()))?;
        let Some(CssRule::Media(rule)) = sheet.rules.0.first() else {
            return Err(error(
                resource,
                "css-media-syntax",
                "invalid media attribute",
            ));
        };
        if rule.query.never_matches() {
            return Ok(None);
        }
        if rule.query.always_matches() {
            return Ok(Some(Vec::new()));
        }
        Ok(Some(vec![
            self.condition(
                rule.query
                    .to_css_string(PrinterOptions::default())
                    .map_err(|cause| error(resource, "css-printer", cause.to_string()))?,
                resource,
            )?,
        ]))
    }
    fn condition(&mut self, text: String, resource: &str) -> Result<usize, BookError> {
        if let Some(index) = self
            .output
            .conditions
            .iter()
            .position(|value| value == &text)
        {
            return Ok(index);
        }
        if self.output.conditions.len() >= 6 {
            return Err(error(
                resource,
                "css-alternatives",
                "conditional alternatives exceed 64",
            ));
        }
        self.output.conditions.push(text);
        Ok(self.output.conditions.len() - 1)
    }
    fn external(
        &mut self,
        href: &str,
        conditions: &[usize],
        depth: usize,
    ) -> Result<(), BookError> {
        if depth >= MAX_DEPTH {
            return Err(error(href, "css-depth", "import depth exceeds 64"));
        }
        if !self.active.insert(href.to_owned()) {
            return Err(error(
                href,
                "css-import-cycle",
                "active stylesheet import cycle",
            ));
        }
        let bytes = self.archive.read_resource(href)?;
        let text = decode(&bytes, href)?;
        if self.asset_ids.insert(href.to_owned()) {
            self.output.assets.push(CssAsset {
                href: href.to_owned(),
                bytes,
            });
        }
        let result = self.sheet(&text, href, conditions, depth + 1);
        self.active.remove(href);
        result
    }
    fn sheet(
        &mut self,
        text: &str,
        resource: &str,
        conditions: &[usize],
        depth: usize,
    ) -> Result<(), BookError> {
        self.admitted_bytes = self
            .admitted_bytes
            .checked_add(text.len())
            .filter(|value| *value <= MAX_TOTAL)
            .ok_or_else(|| {
                error(
                    resource,
                    "css-budget",
                    "aggregate stylesheet admission exceeds archive policy",
                )
            })?;
        if text.len() > MAX_MEMBER {
            return Err(error(
                resource,
                "css-budget",
                "stylesheet exceeds archive member policy",
            ));
        }
        let sheet = StyleSheet::parse(
            text,
            ParserOptions {
                filename: resource.to_owned(),
                ..ParserOptions::default()
            },
        )
        .map_err(|cause| error(resource, "css-syntax", cause.to_string()))?;
        let owned = if contains_nesting(&sheet.rules, 0, resource)? {
            let printed = sheet
                .to_css(PrinterOptions {
                    targets: Targets {
                        include: Features::Nesting,
                        ..Targets::default()
                    },
                    ..PrinterOptions::default()
                })
                .map_err(|cause| error(resource, "css-printer", cause.to_string()))?
                .code;
            if printed.len() > MAX_MEMBER {
                return Err(error(
                    resource,
                    "css-budget",
                    "standard projection exceeds archive member policy",
                ));
            }
            StyleSheet::parse(&printed, ParserOptions::default())
                .map_err(|cause| error(resource, "css-projection", cause.to_string()))?
                .into_owned()
        } else {
            sheet.into_owned()
        };
        let mut namespaces = HashMap::new();
        self.rules(owned.rules, resource, conditions, depth, &mut namespaces)
    }
}
impl Loader<'_> {
    fn rules(
        &mut self,
        rules: CssRuleList<'static>,
        resource: &str,
        conditions: &[usize],
        depth: usize,
        namespaces: &mut HashMap<String, String>,
    ) -> Result<(), BookError> {
        if depth >= MAX_DEPTH {
            return Err(error(
                resource,
                "css-depth",
                "stylesheet topology exceeds 64",
            ));
        }
        for rule in rules.0 {
            self.admitted_rules += 1;
            if self.admitted_rules > self.admitted_bytes.saturating_add(1) {
                return Err(error(
                    resource,
                    "css-budget",
                    "rule metadata exceeds admitted byte budget",
                ));
            }
            match rule {
                CssRule::Import(import) => self.import(import, resource, conditions, depth)?,
                CssRule::Namespace(namespace) => {
                    namespaces.insert(
                        namespace
                            .prefix
                            .map_or_else(String::new, |prefix| prefix.as_ref().to_owned()),
                        namespace.url.as_ref().to_owned(),
                    );
                }
                CssRule::Style(style) => self.style(style, resource, conditions, namespaces)?,
                CssRule::Media(media) => {
                    if media.query.never_matches() {
                        continue;
                    }
                    let mut nested = conditions.to_vec();
                    if !media.query.always_matches() {
                        nested.push(
                            self.condition(
                                media
                                    .query
                                    .to_css_string(PrinterOptions::default())
                                    .map_err(|cause| {
                                        error(resource, "css-printer", cause.to_string())
                                    })?,
                                resource,
                            )?,
                        );
                    }
                    self.rules(media.rules, resource, &nested, depth + 1, namespaces)?;
                }
                CssRule::Supports(supports) => {
                    let mut nested = conditions.to_vec();
                    nested.push(self.condition(
                        format!(
                                "supports:{}",
                                supports
                                    .condition
                                    .to_css_string(PrinterOptions::default())
                                    .map_err(|cause| error(
                                        resource,
                                        "css-printer",
                                        cause.to_string()
                                    ))?
                            ),
                        resource,
                    )?);
                    self.rules(supports.rules, resource, &nested, depth + 1, namespaces)?;
                }
                CssRule::Keyframes(_)
                | CssRule::FontFace(_)
                | CssRule::FontPaletteValues(_)
                | CssRule::FontFeatureValues(_)
                | CssRule::CounterStyle(_)
                | CssRule::Page(_)
                | CssRule::Ignored => {}
                CssRule::LayerStatement(_) => {}
                CssRule::LayerBlock(layer) => {
                    self.unsupported_rules(layer.rules, resource, conditions, depth, namespaces)?
                }
                CssRule::Container(container) => self.unsupported_rules(
                    container.rules,
                    resource,
                    conditions,
                    depth,
                    namespaces,
                )?,
                CssRule::Scope(scope) => {
                    self.unsupported_rules(scope.rules, resource, conditions, depth, namespaces)?
                }
                CssRule::StartingStyle(starting) => {
                    self.unsupported_rules(starting.rules, resource, conditions, depth, namespaces)?
                }
                _ => {
                    return Err(error(
                        resource,
                        "css-rule",
                        "unsupported source stylesheet rule",
                    ));
                }
            }
        }
        Ok(())
    }
    fn unsupported_rules(
        &mut self,
        rules: CssRuleList<'static>,
        resource: &str,
        conditions: &[usize],
        depth: usize,
        namespaces: &mut HashMap<String, String>,
    ) -> Result<(), BookError> {
        let first = self.output.rules.len();
        self.rules(rules, resource, conditions, depth + 1, namespaces)?;
        for rule in &mut self.output.rules[first..] {
            rule.unsupported = true;
        }
        Ok(())
    }
    fn import(
        &mut self,
        import: lightningcss::rules::import::ImportRule<'static>,
        resource: &str,
        conditions: &[usize],
        depth: usize,
    ) -> Result<(), BookError> {
        let first = self.output.rules.len();
        if import.media.never_matches() {
            return Ok(());
        }
        let mut nested = conditions.to_vec();
        if !import.media.always_matches() {
            nested.push(
                self.condition(
                    import
                        .media
                        .to_css_string(PrinterOptions::default())
                        .map_err(|cause| error(resource, "css-printer", cause.to_string()))?,
                    resource,
                )?,
            );
        }
        if let Some(supports) = import.supports {
            nested.push(self.condition(
                format!(
                        "supports:{}",
                        supports
                            .to_css_string(PrinterOptions::default())
                            .map_err(|cause| error(resource, "css-printer", cause.to_string()))?
                    ),
                resource,
            )?);
        }
        self.external(&local(resource, &import.url)?, &nested, depth)?;
        if import.layer.is_some() {
            for rule in &mut self.output.rules[first..] {
                rule.unsupported = true;
            }
        }
        Ok(())
    }
    fn style(
        &mut self,
        style: lightningcss::rules::style::StyleRule<'static>,
        resource: &str,
        conditions: &[usize],
        namespaces: &HashMap<String, String>,
    ) -> Result<(), BookError> {
        let evidence = style
            .to_css_string(PrinterOptions::default())
            .map_err(|cause| error(resource, "css-printer", cause.to_string()))?;
        let selectors = style
            .selectors
            .to_css_string(PrinterOptions::default())
            .map_err(|cause| error(resource, "css-printer", cause.to_string()))?;
        if !style.rules.0.is_empty() {
            return Err(error(
                resource,
                "css-nesting",
                "standard library did not lower nested rules",
            ));
        }
        self.output.rules.push(Rule {
            selection: Selection::parse(&selectors, namespaces),
            declarations: style.declarations,
            conditions: conditions.to_vec(),
            evidence,
            unsupported: false,
        });
        Ok(())
    }
}
fn contains_nesting(
    rules: &CssRuleList<'_>,
    depth: usize,
    resource: &str,
) -> Result<bool, BookError> {
    if depth >= MAX_DEPTH {
        return Err(error(
            resource,
            "css-depth",
            "stylesheet topology exceeds 64",
        ));
    }
    let mut found = false;
    for rule in &rules.0 {
        found |= match rule {
            CssRule::Style(style) => !style.rules.0.is_empty(),
            CssRule::Nesting(_) => true,
            CssRule::Media(media) => contains_nesting(&media.rules, depth + 1, resource)?,
            CssRule::Supports(supports) => contains_nesting(&supports.rules, depth + 1, resource)?,
            CssRule::LayerBlock(layer) => contains_nesting(&layer.rules, depth + 1, resource)?,
            CssRule::Container(container) => {
                contains_nesting(&container.rules, depth + 1, resource)?
            }
            CssRule::Scope(scope) => contains_nesting(&scope.rules, depth + 1, resource)?,
            CssRule::StartingStyle(starting) => {
                contains_nesting(&starting.rules, depth + 1, resource)?
            }
            _ => false,
        };
    }
    Ok(found)
}
fn local(base: &str, href: &str) -> Result<String, BookError> {
    let reference = resolve_href(base, href)?;
    if reference.external || reference.has_query || !reference.fragment.is_empty() {
        return Err(error(
            base,
            "css-reference",
            "stylesheet reference is not an archive-local resource",
        ));
    }
    Ok(reference.resource_href)
}
fn decode(bytes: &[u8], resource: &str) -> Result<String, BookError> {
    let (encoding, bytes) = if let Some(bytes) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        ("UTF-8", bytes)
    } else if let Some(bytes) = bytes.strip_prefix(&[0xff, 0xfe]) {
        ("UTF-16LE", bytes)
    } else if let Some(bytes) = bytes.strip_prefix(&[0xfe, 0xff]) {
        ("UTF-16BE", bytes)
    } else if bytes.starts_with(b"@charset \"") {
        let end = bytes[10..]
            .iter()
            .position(|byte| *byte == b'"')
            .map(|offset| offset + 10)
            .filter(|end| *end <= 110 && bytes.get(*end + 1) == Some(&b';'))
            .ok_or_else(|| error(resource, "css-encoding", "invalid declared CSS encoding"))?;
        let encoding = std::str::from_utf8(&bytes[10..end])
            .map_err(|cause| error(resource, "css-encoding", cause.to_string()))?;
        (encoding, bytes)
    } else {
        ("UTF-8", bytes)
    };
    let converted = convert_encoding(bytes, encoding, "UTF-8", ConversionPolicy::Strict)
        .map_err(|cause| cause.with_resource(resource))?;
    String::from_utf8(converted).map_err(|cause| error(resource, "css-encoding", cause.to_string()))
}
