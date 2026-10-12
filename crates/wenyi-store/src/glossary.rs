use crate::{StoreError, StoreErrorKind, observation::GlossaryCounts};
use rusqlite::{Connection, OpenFlags, types::ValueRef};
use std::{collections::BTreeMap, path::Path};

const GLOSSARY: &[(&str, &str, bool)] = &[
    ("source", "TEXT", false),
    ("target", "TEXT", true),
    ("reading", "TEXT", false),
    ("type", "TEXT", false),
    ("gender", "TEXT", false),
    ("aliases", "TEXT", false),
    ("first_chapter", "INTEGER", false),
    ("note", "TEXT", false),
    ("confidence", "TEXT", false),
    ("locked", "INTEGER", false),
    ("status", "TEXT", false),
    ("updated_at", "REAL", false),
];
const CONFLICTS: &[(&str, &str, bool)] = &[
    ("id", "INTEGER", false),
    ("source", "TEXT", true),
    ("existing_target", "TEXT", false),
    ("proposed_target", "TEXT", false),
    ("chapter", "INTEGER", false),
    ("note", "TEXT", false),
    ("resolved", "INTEGER", false),
    ("created_at", "REAL", false),
];

fn invalid(path: &Path, detail: impl Into<String>) -> StoreError {
    StoreError::new(StoreErrorKind::Sqlite, "invalid_glossary", path, detail)
}

pub(crate) fn inspect(path: &Path) -> Result<GlossaryCounts, StoreError> {
    let absolute = std::fs::canonicalize(path).map_err(|error| StoreError::io(path, error))?;
    let mut uri = url::Url::from_file_path(&absolute)
        .map_err(|()| invalid(path, "cannot encode glossary file URI"))?;
    uri.set_query(Some("mode=ro&immutable=1"));
    let connection = Connection::open_with_flags(
        uri.as_str(),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|error| invalid(path, error.to_string()))?;
    inspect_connection(&connection, path).map_err(|error| invalid(path, error.to_string()))
}

fn inspect_connection(connection: &Connection, path: &Path) -> Result<GlossaryCounts, StoreError> {
    let mut integrity = connection
        .prepare("PRAGMA integrity_check")
        .map_err(|error| invalid(path, error.to_string()))?;
    let results = integrity
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| invalid(path, error.to_string()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid(path, error.to_string()))?;
    if results != ["ok"] {
        return Err(invalid(path, "SQLite integrity check failed"));
    }
    schema(
        connection,
        "glossary",
        "PRAGMA table_info(glossary)",
        GLOSSARY,
        path,
    )?;
    schema(
        connection,
        "term_conflicts",
        "PRAGMA table_info(term_conflicts)",
        CONFLICTS,
        path,
    )?;
    validate_rows(
        connection,
        "SELECT source,target,reading,type,gender,aliases,first_chapter,note,confidence,locked,status,updated_at FROM glossary",
        GLOSSARY,
        path,
    )?;
    validate_rows(
        connection,
        "SELECT id,source,existing_target,proposed_target,chapter,note,resolved,created_at FROM term_conflicts",
        CONFLICTS,
        path,
    )?;
    let terms: i64 = connection
        .query_row("SELECT COUNT(*) FROM glossary", [], |row| row.get(0))
        .map_err(|error| invalid(path, error.to_string()))?;
    let open_conflicts: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM term_conflicts WHERE resolved=0",
            [],
            |row| row.get(0),
        )
        .map_err(|error| invalid(path, error.to_string()))?;
    Ok(GlossaryCounts {
        terms: u64::try_from(terms).map_err(|_| invalid(path, "negative glossary count"))?,
        open_conflicts: u64::try_from(open_conflicts)
            .map_err(|_| invalid(path, "negative conflict count"))?,
    })
}

fn schema(
    connection: &Connection,
    name: &str,
    query: &str,
    expected: &[(&str, &str, bool)],
    path: &Path,
) -> Result<(), StoreError> {
    let kind: String = connection
        .query_row(
            "SELECT type FROM sqlite_schema WHERE name=?1",
            [name],
            |row| row.get(0),
        )
        .map_err(|error| invalid(path, error.to_string()))?;
    if kind != "table" {
        return Err(invalid(path, "glossary relation must be a table"));
    }
    let mut statement = connection
        .prepare(query)
        .map_err(|error| invalid(path, error.to_string()))?;
    let columns = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?,
                (
                    row.get::<_, String>(2)?,
                    row.get::<_, bool>(3)?,
                    row.get::<_, i64>(5)?,
                ),
            ))
        })
        .map_err(|error| invalid(path, error.to_string()))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| invalid(path, error.to_string()))?;
    for (column, affinity, not_null) in expected {
        let actual = columns
            .get(*column)
            .ok_or_else(|| invalid(path, format!("missing {name}.{column}")))?;
        if !actual.0.eq_ignore_ascii_case(affinity) || actual.1 != *not_null {
            return Err(invalid(path, format!("invalid schema for {name}.{column}")));
        }
    }
    let primary = if name == "glossary" { "source" } else { "id" };
    if columns.get(primary).is_none_or(|column| column.2 != 1) {
        return Err(invalid(path, "missing glossary primary key"));
    }
    Ok(())
}

fn validate_rows(
    connection: &Connection,
    query: &str,
    expected: &[(&str, &str, bool)],
    path: &Path,
) -> Result<(), StoreError> {
    let mut statement = connection
        .prepare(query)
        .map_err(|error| invalid(path, error.to_string()))?;
    let mut rows = statement
        .query([])
        .map_err(|error| invalid(path, error.to_string()))?;
    while let Some(row) = rows
        .next()
        .map_err(|error| invalid(path, error.to_string()))?
    {
        for (index, (name, affinity, required)) in expected.iter().enumerate() {
            let value = row
                .get_ref(index)
                .map_err(|error| invalid(path, error.to_string()))?;
            let valid = match value {
                ValueRef::Null => !required,
                ValueRef::Text(bytes) => *affinity == "TEXT" && std::str::from_utf8(bytes).is_ok(),
                ValueRef::Integer(number) => {
                    (*affinity == "INTEGER" || *affinity == "REAL")
                        && (!matches!(*name, "locked" | "resolved") || matches!(number, 0 | 1))
                }
                ValueRef::Real(number) => *affinity == "REAL" && number.is_finite(),
                ValueRef::Blob(_) => false,
            };
            if !valid {
                return Err(invalid(path, format!("invalid raw value in {name}")));
            }
        }
    }
    Ok(())
}

pub(crate) fn hot_journal(header: &[u8], size: u64) -> bool {
    size > 512 && header.iter().any(|byte| *byte != 0)
}
