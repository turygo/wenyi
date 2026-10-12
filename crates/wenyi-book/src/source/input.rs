#![forbid(unsafe_code)]
use super::{BomKind, DocumentEnvelope, MAX_BYTES, ParseMode};
use crate::error::{BookError, BookErrorKind};

pub(crate) fn source_envelope(bytes: &[u8]) -> Result<DocumentEnvelope, BookError> {
    if bytes.len() > MAX_BYTES {
        return Err(super::tree::safety("source input limit exceeded"));
    }
    let bytes = &bytes[..bytes.len().min(1024)];
    let (bom, encoding, evidence) =
        if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[b'<', 0, b'?', 0]) {
            (
                bytes.starts_with(&[0xff, 0xfe]).then_some(BomKind::Utf16Le),
                "UTF-16LE",
                ascii_utf16(bytes, true),
            )
        } else if bytes.starts_with(&[0xfe, 0xff]) || bytes.starts_with(&[0, b'<', 0, b'?']) {
            (
                bytes.starts_with(&[0xfe, 0xff]).then_some(BomKind::Utf16Be),
                "UTF-16BE",
                ascii_utf16(bytes, false),
            )
        } else {
            let bom = bytes
                .starts_with(&[0xef, 0xbb, 0xbf])
                .then_some(BomKind::Utf8);
            let bytes = if bom.is_some() { &bytes[3..] } else { bytes };
            (
                bom,
                "UTF-8",
                bytes
                    .iter()
                    .map(|&byte| if byte.is_ascii() { byte as char } else { ' ' })
                    .collect(),
            )
        };
    let evidence = evidence.trim_start_matches('\u{feff}');
    let declared = declaration_encoding(evidence).ok().flatten();
    Ok(DocumentEnvelope {
        mode: ParseMode::Xml,
        encoding: declared.unwrap_or_else(|| encoding.into()),
        bom,
        declaration: evidence.starts_with("<?xml")
            && evidence
                .as_bytes()
                .get(5)
                .is_some_and(u8::is_ascii_whitespace),
        doctype: doctype(evidence),
    })
}
fn ascii_utf16(bytes: &[u8], little: bool) -> String {
    bytes
        .chunks_exact(2)
        .map(|pair| {
            let unit = if little {
                u16::from_le_bytes([pair[0], pair[1]])
            } else {
                u16::from_be_bytes([pair[0], pair[1]])
            };
            if unit == 0xfeff {
                '\u{feff}'
            } else if unit < 128 {
                unit as u8 as char
            } else {
                ' '
            }
        })
        .collect()
}
fn encoding_span(declaration: &str) -> Result<Option<(usize, usize)>, BookError> {
    let Some(position) = declaration.find("encoding") else {
        return Ok(None);
    };
    let remainder = &declaration[position + 8..];
    let remainder = remainder.trim_start();
    let Some(remainder) = remainder.strip_prefix('=') else {
        return Err(encoding("invalid XML encoding declaration"));
    };
    let remainder = remainder.trim_start();
    let quote = remainder
        .chars()
        .next()
        .ok_or_else(|| encoding("missing XML encoding name"))?;
    if quote != '\'' && quote != '"' {
        return Err(encoding("unquoted XML encoding name"));
    }
    let start = declaration.len() - remainder.len() + 1;
    let length = remainder[1..]
        .find(quote)
        .ok_or_else(|| encoding("unterminated XML encoding name"))?;
    if length == 0 || length > 100 {
        return Err(encoding("invalid XML encoding name length"));
    }
    Ok(Some((start, start + length)))
}
fn declaration_encoding(text: &str) -> Result<Option<String>, BookError> {
    if !text.starts_with("<?xml") {
        return Ok(None);
    }
    let Some(end) = text.find("?>") else {
        return Ok(None);
    };
    Ok(encoding_span(&text[..end])?.map(|(start, finish)| text[start..finish].to_owned()))
}
// 此处只提取包 XML 的信封证据；语法准入由真正的原生解析器负责。
fn doctype(text: &str) -> Option<String> {
    let mut position = 0;
    while position < text.len() {
        let rest = &text[position..];
        if rest.starts_with("<!--") {
            position += rest.find("-->")? + 3;
        } else if rest.starts_with("<?") {
            position += rest.find("?>")? + 2;
        } else if rest.starts_with("<!DOCTYPE") {
            let mut quote = None;
            let mut brackets = 0_u32;
            for (offset, character) in rest.char_indices().skip(9) {
                if let Some(delimiter) = quote {
                    if character == delimiter {
                        quote = None;
                    }
                } else {
                    match character {
                        '\'' | '"' => quote = Some(character),
                        '[' => brackets += 1,
                        ']' => brackets = brackets.saturating_sub(1),
                        '>' if brackets == 0 => return Some(rest[..offset + 1].to_owned()),
                        _ => {}
                    }
                }
            }
            return None;
        } else if rest.starts_with('<') {
            break;
        } else {
            let character = rest.chars().next()?;
            if !character.is_ascii_whitespace() {
                break;
            }
            position += character.len_utf8();
        }
    }
    None
}
fn encoding(detail: &str) -> BookError {
    BookError::new(BookErrorKind::Encoding, "source-encoding", detail)
}
