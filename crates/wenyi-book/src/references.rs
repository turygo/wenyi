#![forbid(unsafe_code)]
use crate::error::{BookError, BookErrorKind};
use fluent_uri::IriRef;
use percent_encoding::percent_decode_str;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedReference {
    pub resource_href: String,
    pub fragment: String,
    pub external: bool,
    pub has_query: bool,
}
impl ResolvedReference {
    pub fn target_key(&self) -> String {
        if self.resource_href.is_empty() {
            String::new()
        } else if self.fragment.is_empty() {
            self.resource_href.clone()
        } else {
            format!("{}#{}", self.resource_href, self.fragment)
        }
    }
}
pub fn resolve_href(base_path: &str, href: &str) -> Result<ResolvedReference, BookError> {
    let reference = IriRef::parse(href).map_err(|error| {
        BookError::new(BookErrorKind::Reference, "invalid_uri", error.to_string())
    })?;
    let external = reference.scheme().is_some() || reference.authority().is_some();
    let fragment = reference
        .fragment()
        .map_or_else(String::new, |value| decode(value.as_str()));
    let resource_href = if external {
        String::new()
    } else {
        let decoded = decode(reference.path().as_str());
        let joined = if let Some(path) = decoded.strip_prefix('/') {
            path.to_owned()
        } else if decoded.is_empty() {
            base_path.to_owned()
        } else {
            base_path
                .rsplit_once('/')
                .map_or_else(|| decoded.clone(), |(dir, _)| format!("{dir}/{decoded}"))
        };
        local_path(&joined)?
    };
    Ok(ResolvedReference {
        resource_href,
        fragment,
        external,
        has_query: reference
            .query()
            .is_some_and(|query| !query.as_str().is_empty()),
    })
}
fn decode(value: &str) -> String {
    percent_decode_str(value).decode_utf8_lossy().into_owned()
}
fn local_path(value: &str) -> Result<String, BookError> {
    let mut parts = Vec::new();
    for part in value.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(BookError::new(
                        BookErrorKind::Reference,
                        "unsafe_reference",
                        "reference escapes archive root",
                    ));
                }
            }
            _ => parts.push(part),
        }
    }
    let path = parts.join("/");
    if !path.is_empty() && !crate::archive::guard::safe_name(&path) {
        return Err(BookError::new(
            BookErrorKind::Reference,
            "unsafe_reference",
            "unsafe archive reference",
        ));
    }
    Ok(path)
}
