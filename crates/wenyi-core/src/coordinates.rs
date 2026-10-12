use crate::{CoreError, CoreErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScalarOffset(usize);

impl ScalarOffset {
    pub fn new(value: usize, text: &str) -> Result<Self, CoreError> {
        if value > text.chars().count() {
            return Err(invalid("scalar offset exceeds text"));
        }
        Ok(Self(value))
    }

    pub fn from_byte(text: &str, byte: usize) -> Result<Self, CoreError> {
        if !text.is_char_boundary(byte) {
            return Err(invalid("byte offset is not a UTF8 scalar boundary"));
        }
        Ok(Self(text[..byte].chars().count()))
    }

    pub fn value(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarRange {
    pub start: ScalarOffset,
    pub end: ScalarOffset,
}

impl ScalarRange {
    pub fn new(start: usize, end: usize, text: &str) -> Result<Self, CoreError> {
        if start > end {
            return Err(invalid("scalar range is reversed"));
        }
        if end > text.chars().count() {
            return Err(invalid("scalar range exceeds text"));
        }
        Ok(Self {
            start: ScalarOffset(start),
            end: ScalarOffset(end),
        })
    }

    pub fn byte_range(&self, text: &str) -> Result<std::ops::Range<usize>, CoreError> {
        if self.start > self.end {
            return Err(invalid("scalar range is reversed"));
        }
        let mut start = None;
        let mut end = None;
        for (scalar, byte) in text
            .char_indices()
            .map(|(byte, _)| byte)
            .chain(std::iter::once(text.len()))
            .enumerate()
        {
            if scalar == self.start.0 {
                start = Some(byte);
            }
            if scalar == self.end.0 {
                end = Some(byte);
                break;
            }
        }
        match (start, end) {
            (Some(start), Some(end)) => Ok(start..end),
            _ => Err(invalid("scalar range exceeds text")),
        }
    }
}

fn invalid(detail: &str) -> CoreError {
    CoreError::new(
        CoreErrorKind::Coordinate,
        "invalid_scalar_coordinate",
        detail,
    )
}
