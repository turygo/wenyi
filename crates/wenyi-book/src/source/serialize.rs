#![forbid(unsafe_code)]
use super::tree::{safety, source};
use super::{
    BomKind, ConversionPolicy, NodeId, ParseMode, ParsedDocument, SourceTree, convert_encoding,
    xml_ffi,
};
use crate::error::BookError;

pub fn serialize_source(document: &ParsedDocument) -> Result<Vec<u8>, BookError> {
    let envelope = &document.envelope;
    let mut native_envelope = envelope.clone();
    if envelope.mode == ParseMode::RecoveredHtml {
        native_envelope.encoding = "UTF-8".into();
    }
    if native_envelope.encoding.eq_ignore_ascii_case("UTF-16") {
        native_envelope.encoding = match envelope.bom {
            Some(BomKind::Utf16Le) => "UTF-16LE",
            Some(BomKind::Utf16Be) => "UTF-16BE",
            _ => "UTF-16",
        }
        .into();
    }
    let mut bytes = xml_ffi::serialize(&document.tree, Some(&native_envelope), None)?;
    let generated_bom = strip_bom(&mut bytes);
    let mut prefix = String::new();
    if envelope.declaration {
        let encoding = if envelope.mode == ParseMode::RecoveredHtml {
            "UTF-8"
        } else {
            &envelope.encoding
        };
        if encoding.contains(['\'', '"', '<', '>', '\0']) {
            return Err(source("invalid owned encoding declaration"));
        }
        prefix.push_str(&format!("<?xml version='1.0' encoding='{encoding}'?>\n"));
    }
    if let Some(doctype) = &envelope.doctype {
        if !doctype.starts_with("<!DOCTYPE") || !doctype.ends_with('>') || doctype.contains('\0') {
            return Err(source("invalid owned DOCTYPE"));
        }
        // 保留声明证据，但不把 DTD 挂到临时文档，避免加载资源或 XHTML 元标签注入。
        prefix.push_str(doctype);
        prefix.push('\n');
    }
    let mut output = if prefix.is_empty() {
        bytes
    } else {
        let mut output = convert_encoding(
            prefix.as_bytes(),
            "UTF-8",
            &native_envelope.encoding,
            ConversionPolicy::Strict,
        )?;
        strip_bom(&mut output);
        output
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| safety("serialized source length overflow"))?;
        output.extend_from_slice(&bytes);
        output
    };
    let bom = if envelope.mode == ParseMode::Xml {
        envelope.bom.or(generated_bom)
    } else {
        None
    };
    let prefix: &[u8] = match bom {
        Some(BomKind::Utf8) => &[0xef, 0xbb, 0xbf],
        Some(BomKind::Utf16Le) => &[0xff, 0xfe],
        Some(BomKind::Utf16Be) => &[0xfe, 0xff],
        None => &[],
    };
    if !prefix.is_empty() {
        let capacity = output
            .len()
            .checked_add(prefix.len())
            .ok_or_else(|| safety("serialized BOM length overflow"))?;
        let mut prefixed = Vec::with_capacity(capacity);
        prefixed.extend_from_slice(prefix);
        prefixed.extend_from_slice(&output);
        output = prefixed;
    }
    Ok(output)
}
fn strip_bom(bytes: &mut Vec<u8>) -> Option<BomKind> {
    let (kind, length) = if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        (BomKind::Utf8, 3)
    } else if bytes.starts_with(&[0xff, 0xfe]) {
        (BomKind::Utf16Le, 2)
    } else if bytes.starts_with(&[0xfe, 0xff]) {
        (BomKind::Utf16Be, 2)
    } else {
        return None;
    };
    bytes.drain(..length);
    Some(kind)
}
pub fn serialize_block(tree: &SourceTree, node: NodeId) -> Result<Vec<u8>, BookError> {
    xml_ffi::serialize(tree, None, Some(node))
}
