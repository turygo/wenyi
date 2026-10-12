#![forbid(unsafe_code)]
use super::{Archive, Limits, stream_member};
use std::io::Cursor;
#[path = "../../tests/support/archive.rs"]
mod support;

#[test]
fn declared_and_actual_bounds_include_exact_limit_not_plus_one() {
    let limits = Limits {
        members: 2,
        member: 5,
        total: 8,
    };
    for (size, succeeds) in [(4, true), (5, true), (6, false)] {
        let bytes = vec![b'x'; size];
        let mut total = 0;
        let result = stream_member(
            &mut Cursor::new(&bytes),
            size as u64,
            limits,
            &mut total,
            None,
        );
        assert_eq!(result.is_ok(), succeeds);
        let temp = support::Temp::new(&support::zip(
            &[("x", &bytes)],
            zip::CompressionMethod::Stored,
        ));
        assert_eq!(
            Archive::open_with_limits(&temp.path, limits).is_ok(),
            succeeds
        );
    }
    for (second, succeeds) in [(3, true), (4, false)] {
        let mut total = 0;
        stream_member(&mut Cursor::new(b"12345"), 5, limits, &mut total, None).unwrap();
        let result = stream_member(
            &mut Cursor::new(vec![b'x'; second]),
            second as u64,
            limits,
            &mut total,
            None,
        );
        assert_eq!(result.is_ok(), succeeds);
        let bytes = vec![b'x'; second];
        let temp = support::Temp::new(&support::zip(
            &[("a", b"12345"), ("b", &bytes)],
            zip::CompressionMethod::Stored,
        ));
        assert_eq!(
            Archive::open_with_limits(&temp.path, limits).is_ok(),
            succeeds
        );
    }
    let temp = support::Temp::new(&support::zip(
        &[("a", b""), ("b", b""), ("c", b"")],
        zip::CompressionMethod::Stored,
    ));
    assert_eq!(
        Archive::open_with_limits(&temp.path, limits)
            .err()
            .unwrap()
            .code,
        "archive_member_limit"
    );
}
#[test]
fn declared_size_must_equal_actual_eof() {
    let limits = Limits {
        members: 1,
        member: 5,
        total: 5,
    };
    assert_eq!(
        stream_member(&mut Cursor::new(b"1234"), 5, limits, &mut 0, None)
            .unwrap_err()
            .code,
        "member_size_mismatch"
    );
    assert_eq!(
        stream_member(&mut Cursor::new(b"123456"), 5, limits, &mut 0, None)
            .unwrap_err()
            .code,
        "member_too_large"
    );
}
#[test]
fn actual_directory_count_is_guarded_before_library_construction() {
    let names: Vec<_> = (0..10001).map(|index| format!("member-{index}")).collect();
    let entries: Vec<_> = names
        .iter()
        .map(|name| (name.as_str(), b"".as_slice()))
        .collect();
    let mut bytes = support::zip(&entries, zip::CompressionMethod::Stored);
    let eocd = support::positions(&bytes, b"PK\x05\x06")[0];
    support::set16(&mut bytes, eocd + 8, 1);
    support::set16(&mut bytes, eocd + 10, 1);
    let temp = support::Temp::new(&bytes);
    assert_eq!(
        Archive::open(&temp.path).err().unwrap().code,
        "archive_member_limit"
    );
}
