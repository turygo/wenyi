use lightningcss::declaration::DeclarationBlock;
use lightningcss::properties::custom::{Token, TokenList, TokenOrValue};
use lightningcss::properties::font::{
    AbsoluteFontWeight, FontSize, FontStyle, FontWeight, VerticalAlign, VerticalAlignKeyword,
};
use lightningcss::properties::{CSSWideKeyword, Property, PropertyId};
use lightningcss::values::length::{LengthPercentage, LengthValue};

pub(super) const COUNT: usize = 6;
#[derive(Clone, Copy, Debug)]
pub(super) enum Value {
    Number(f32),
    Relative(f32),
    Bolder,
    Lighter,
    Initial,
    Inherit,
    Unset,
    Revert,
    Unknown,
}
#[derive(Clone, Debug)]
pub(super) struct Projection {
    pub(super) values: [Option<(Value, bool)>; COUNT],
    pub(super) animation: bool,
}
impl Projection {
    pub(super) fn relevant(&self) -> bool {
        self.animation || self.values.iter().any(Option::is_some)
    }
}
pub(super) fn project(block: &DeclarationBlock<'_>) -> Projection {
    let mut result = Projection {
        values: [None; COUNT],
        animation: false,
    };
    for (property, important) in block.iter() {
        let id = property.property_id();
        let name = id.name();
        if name.starts_with("animation") {
            result.animation = true;
        }
        if let Property::All(keyword) = property {
            for value in &mut result.values {
                set(value, wide(keyword), important);
            }
            continue;
        }
        for (index, id) in [
            PropertyId::FontWeight,
            PropertyId::FontStyle,
            PropertyId::VerticalAlign,
            PropertyId::FontSize,
        ]
        .iter()
        .enumerate()
        {
            if property.property_id() == *id {
                set(&mut result.values[index], typed(property, index), important);
            } else if let Some(longhand) = property.longhand(id) {
                set(
                    &mut result.values[index],
                    typed(&longhand, index),
                    important,
                );
            } else if let Property::Unparsed(unparsed) = property
                && unparsed.property_id == PropertyId::Font
                && matches!(index, 0 | 1 | 3)
            {
                set(
                    &mut result.values[index],
                    reset(&unparsed.value).unwrap_or(Value::Unknown),
                    important,
                );
            }
        }
        if let Property::Custom(custom) = property {
            let index = match custom.name.as_ref() {
                "float" => 4,
                "initial-letter" => 5,
                _ => continue,
            };
            set(
                &mut result.values[index],
                geometry(&custom.value, index),
                important,
            );
        }
    }
    result
}
fn set(slot: &mut Option<(Value, bool)>, value: Value, important: bool) {
    if slot.is_none_or(|(_, previous)| !previous || important) {
        *slot = Some((value, important));
    }
}
fn wide(keyword: &CSSWideKeyword) -> Value {
    match keyword {
        CSSWideKeyword::Initial => Value::Initial,
        CSSWideKeyword::Inherit => Value::Inherit,
        CSSWideKeyword::Unset => Value::Unset,
        CSSWideKeyword::Revert => Value::Revert,
        CSSWideKeyword::RevertLayer => Value::Unknown,
    }
}
fn reset(tokens: &TokenList<'_>) -> Option<Value> {
    let mut values = tokens.0.iter().filter(|value| !value.is_whitespace());
    let TokenOrValue::Token(Token::Ident(name)) = values.next()? else {
        return None;
    };
    if values.next().is_some() {
        return None;
    }
    match name.as_ref() {
        "initial" => Some(Value::Initial),
        "inherit" => Some(Value::Inherit),
        "unset" => Some(Value::Unset),
        "revert" => Some(Value::Revert),
        _ => None,
    }
}
fn typed(property: &Property<'_>, index: usize) -> Value {
    match property {
        Property::FontWeight(weight) => match weight {
            FontWeight::Absolute(AbsoluteFontWeight::Weight(value)) => Value::Number(*value),
            FontWeight::Absolute(AbsoluteFontWeight::Normal) => Value::Number(400.0),
            FontWeight::Absolute(AbsoluteFontWeight::Bold) => Value::Number(700.0),
            FontWeight::Bolder => Value::Bolder,
            FontWeight::Lighter => Value::Lighter,
        },
        Property::FontStyle(style) => Value::Number(if matches!(style, FontStyle::Normal) {
            0.0
        } else {
            1.0
        }),
        Property::VerticalAlign(VerticalAlign::Keyword(keyword)) => Value::Number(match keyword {
            VerticalAlignKeyword::Super => 1.0,
            VerticalAlignKeyword::Sub => -1.0,
            _ => 0.0,
        }),
        Property::VerticalAlign(VerticalAlign::Length(_)) => Value::Number(0.0),
        Property::FontSize(size) => size_value(size),
        Property::Unparsed(unparsed) => reset(&unparsed.value).unwrap_or(Value::Unknown),
        _ => {
            let _ = index;
            Value::Unknown
        }
    }
}
fn size_value(size: &FontSize) -> Value {
    match size {
        FontSize::Length(LengthPercentage::Dimension(LengthValue::Em(value))) => {
            Value::Relative(*value)
        }
        FontSize::Length(LengthPercentage::Dimension(LengthValue::Rem(value))) => {
            Value::Number(*value * 16.0)
        }
        FontSize::Length(LengthPercentage::Dimension(LengthValue::Px(value))) => {
            Value::Number(*value)
        }
        FontSize::Length(LengthPercentage::Dimension(LengthValue::Pt(value))) => {
            Value::Number(*value * 4.0 / 3.0)
        }
        FontSize::Length(LengthPercentage::Percentage(value)) => Value::Relative(value.0),
        _ => Value::Unknown,
    }
}
fn geometry(tokens: &TokenList<'_>, index: usize) -> Value {
    if let Some(reset) = reset(tokens) {
        return reset;
    }
    let mut tokens = tokens.0.iter().filter(|value| !value.is_whitespace());
    let Some(first) = tokens.next() else {
        return Value::Unknown;
    };
    if index == 4 {
        if tokens.next().is_some() {
            return Value::Unknown;
        }
        return match first {
            TokenOrValue::Token(Token::Ident(name)) => match name.as_ref() {
                "none" => Value::Number(0.0),
                "left" | "right" | "inline-start" | "inline-end" => Value::Number(1.0),
                _ => Value::Unknown,
            },
            _ => Value::Unknown,
        };
    }
    match first {
        TokenOrValue::Token(Token::Ident(name))
            if name.as_ref() == "normal" && tokens.next().is_none() =>
        {
            Value::Number(0.0)
        }
        TokenOrValue::Token(Token::Number { value, .. }) if *value >= 1.0 => {
            if let Some(second) = tokens.next()
                && !matches!(second, TokenOrValue::Token(Token::Number { value, .. }) if *value >= 1.0 && value.fract() == 0.0)
            {
                return Value::Unknown;
            }
            if tokens.next().is_some() {
                Value::Unknown
            } else {
                Value::Number(*value)
            }
        }
        _ => Value::Unknown,
    }
}
