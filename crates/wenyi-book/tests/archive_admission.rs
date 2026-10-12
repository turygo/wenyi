#![forbid(unsafe_code)]
#[path = "support/archive.rs"]
mod support;
use support::{Temp, positions, raw_member, set16, set32};
use wenyi_book::{
    archive::{Archive, MemberKind},
    error::BookErrorKind,
};
use zip::CompressionMethod;

fn failure(bytes: &[u8]) -> String {
    let temp = Temp::new(bytes);
    let error = Archive::open(&temp.path).err().expect("archive must fail");
    assert_eq!(error.kind, BookErrorKind::Archive);
    error.code.to_owned()
}
#[test]
fn supported_compressions_stream_actual_resources() {
    let payload = b"qualified ZIP compression payload\n";
    for method in [
        CompressionMethod::Stored,
        CompressionMethod::Deflated,
        CompressionMethod::Bzip2,
    ] {
        let temp = Temp::new(&support::zip(&[("resource", payload)], method));
        let mut archive = Archive::open(&temp.path).unwrap();
        assert_eq!(archive.read_resource("resource").unwrap(), payload);
        assert_eq!(
            archive.members()["resource"].expanded_size,
            payload.len() as u64
        );
        archive.finish().unwrap();
    }
}
#[test]
fn directories_and_case_sensitive_member_identity() {
    let temp = Temp::new(&support::zip(
        &[
            ("dir/", b""),
            ("dir/File", b"x"),
            ("dir/file", b"y"),
            ("é", b"nfc"),
            ("e\u{301}", b"nfd"),
        ],
        CompressionMethod::Stored,
    ));
    let archive = Archive::open(&temp.path).unwrap();
    assert_eq!(archive.members()["dir/"].kind, MemberKind::Directory);
    assert_eq!(archive.members().len(), 5);
    assert_ne!(
        archive.members()["é"].sha256,
        archive.members()["e\u{301}"].sha256
    );
    archive.finish().unwrap();
}
#[test]
fn names_duplicates_types_encryption_and_method_fail_closed() {
    for name in [
        "", "/x", "a\\b", "a\0b", "a//b", "./x", "a/../b", ".", "..", "a/.",
    ] {
        assert_eq!(
            failure(&raw_member(name.as_bytes(), 0, 0, b"x", b"x")),
            "unsafe_entry"
        );
    }
    let mut duplicate = support::zip(&[("a", b"x"), ("b", b"y")], CompressionMethod::Stored);
    let central = positions(&duplicate, b"PK\x01\x02");
    duplicate[central[1] + 46] = b'a';
    assert_eq!(failure(&duplicate), "duplicate_entry");
    assert_eq!(
        failure(&support::zip(
            &[("a", b"x"), ("a/", b"")],
            CompressionMethod::Stored
        )),
        "duplicate_entry"
    );
    let mut special = raw_member(b"x", 0, 0, b"x", b"x");
    let central = positions(&special, b"PK\x01\x02")[0];
    for mode in [0o120777_u32, 0o020666, 0o060666, 0o010666] {
        set32(&mut special, central + 38, mode << 16);
        assert_eq!(failure(&special), "special_entry");
    }
    assert_eq!(
        failure(&raw_member(b"x", 1, 0, b"x", b"x")),
        "encrypted_entry"
    );
    assert_eq!(
        failure(&raw_member(b"x", 0, 99, b"x", b"x")),
        "unsupported_compression"
    );
}
#[test]
fn unused_corruption_crc_size_and_truncation_are_fatal() {
    let original = support::zip(
        &[("used", b"hello"), ("unused", b"corrupt-me")],
        CompressionMethod::Stored,
    );
    let mut bytes = original.clone();
    let local = positions(&bytes, b"PK\x03\x04")[1];
    let name_len = u16::from_le_bytes(bytes[local + 26..local + 28].try_into().unwrap()) as usize;
    let extra_len = u16::from_le_bytes(bytes[local + 28..local + 30].try_into().unwrap()) as usize;
    bytes[local + 30 + name_len + extra_len] ^= 0x80;
    assert_eq!(failure(&bytes), "member_read");
    for declared in [4, 6] {
        let mut bytes = raw_member(b"x", 0, 0, b"hello", b"hello");
        let central = positions(&bytes, b"PK\x01\x02")[0];
        set32(&mut bytes, central + 24, declared);
        assert!(["member_read", "member_size_mismatch"].contains(&failure(&bytes).as_str()));
    }
    for cut in [1, 10, 21] {
        assert_eq!(failure(&original[..original.len() - cut]), "invalid_zip");
    }
    assert_eq!(failure(b"not a ZIP"), "invalid_zip");
    let mut truncated_data = raw_member(b"x", 0, 0, b"x", b"x");
    let central = positions(&truncated_data, b"PK\x01\x02")[0];
    set32(&mut truncated_data, central + 20, 1000);
    assert!(
        ["invalid_zip", "member_read", "member_size_mismatch"]
            .contains(&failure(&truncated_data).as_str())
    );
}
#[test]
fn end_record_guards_counts_and_preserves_prefix_offsets() {
    let original = raw_member(b"x", 0, 0, b"x", b"x");
    let mut oversized = original.clone();
    let end = positions(&oversized, b"PK\x05\x06")[0];
    set16(&mut oversized, end + 8, 10001);
    set16(&mut oversized, end + 10, 10001);
    assert_eq!(failure(&oversized), "archive_member_limit");
    let mut count_mismatch = original.clone();
    set16(&mut count_mismatch, end + 8, 0);
    set16(&mut count_mismatch, end + 10, 0);
    assert_eq!(failure(&count_mismatch), "invalid_zip");
    assert_eq!(
        failure(&support::zip64(original.clone(), u64::MAX, &[])),
        "archive_member_limit"
    );
    for bytes in [original.clone(), support::zip64(original, 1, &[])] {
        for prefix in [b"".as_slice(), b"legal SFX prefix".as_slice()] {
            let mut prefixed = prefix.to_vec();
            prefixed.extend_from_slice(&bytes);
            let temp = Temp::new(&prefixed);
            let mut archive = Archive::open(&temp.path).unwrap();
            assert_eq!(archive.read_resource("x").unwrap(), b"x");
            archive.finish().unwrap();
        }
    }
}
#[test]
fn finish_rejects_content_replacement_and_deletion() {
    for mutation in 0..3 {
        let bytes = raw_member(b"x", 0, 0, b"x", b"x");
        let temp = Temp::new(&bytes);
        let archive = Archive::open(&temp.path).unwrap();
        match mutation {
            0 => {
                let mut changed = bytes;
                changed[31] = b'y';
                std::fs::write(&temp.path, changed).unwrap();
            }
            1 => {
                let replacement = temp.root.join("replacement");
                std::fs::write(&replacement, bytes).unwrap();
                std::fs::remove_file(&temp.path).unwrap();
                std::fs::rename(replacement, &temp.path).unwrap();
            }
            _ => std::fs::remove_file(&temp.path).unwrap(),
        }
        let error = archive.finish().unwrap_err();
        assert_eq!(error.kind, BookErrorKind::Source);
        assert_eq!(error.code, "source_changed");
    }
}

#[test]
fn construction_metadata_budget_covers_standard_and_zip64_directories() {
    let original = raw_member(b"x", 0, 0, b"x", b"x");
    let mut bytes = original.clone();
    let end = positions(&bytes, b"PK\x05\x06")[0];
    set32(&mut bytes, end + 12, 65 * 1024 * 1024);
    assert_eq!(failure(&bytes), "archive_metadata_limit");
    let mut bytes = support::zip64(original.clone(), 1, &[]);
    let record = positions(&bytes, b"PK\x06\x06")[0];
    bytes[record + 40..record + 48].copy_from_slice(&(65_u64 * 1024 * 1024).to_le_bytes());
    assert_eq!(failure(&bytes), "archive_metadata_limit");
    let bytes = support::zip64(original, 1, &vec![0; 70 * 1024]);
    let temp = Temp::new(&bytes);
    let mut archive = Archive::open(&temp.path).unwrap();
    assert_eq!(archive.read_resource("x").unwrap(), b"x");
    archive.finish().unwrap();
}

#[test]
fn earlier_footer_candidates_are_bounded_before_construction() {
    let mut bytes = raw_member(b"x", 0, 0, b"x", b"x");
    let end = positions(&bytes, b"PK\x05\x06")[0];
    set16(&mut bytes, end + 8, 10001);
    set16(&mut bytes, end + 10, 10001);
    let later = raw_member(b"y", 0, 0, b"y", b"y");
    bytes.extend_from_slice(&later);
    assert_eq!(failure(&bytes), "archive_member_limit");
}
