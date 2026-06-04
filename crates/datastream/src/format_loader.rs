//! Dataset format loader — supports 6 popular ML training formats.
//!
//! Auto-detects format from file extension and loads into `Vec<DataSample>`.
//!
//! ## Supported formats
//!
//! | Extension | Format | Feature | Notes |
//! |-----------|--------|---------|-------|
//! | `.csv` | CSV | `csv` | comma-separated, looks for `text` column |
//! | `.tsv` | TSV | `csv` | tab-separated, looks for `text` column |
//! | `.json` | JSON | `json` | top-level array of objects with `text` field |
//! | `.jsonl` / `.ndjson` | JSON Lines | `json` | one JSON object per line with `text` field |
//! | `.parquet` | Apache Parquet | `parquet` | columnar, looks for `text` column |
//! | `.arrow` / `.ipc` | Apache Arrow IPC | `arrow` | zero-copy columnar format (existing) |

use std::io::{BufRead, BufReader};
use std::path::Path;

use tracing::{info, warn};
use uuid::Uuid;

use crate::types::{DataSample, SourceInfo};

/// Streaming dataset iterator — yields `DataSample` one at a time
/// without loading the entire dataset into memory.
///
/// The iterator holds an open file handle and reads on demand.
/// Available for JSONL and CSV formats (naturally line-oriented).
pub struct StreamingDatasetIterator {
    reader: Box<dyn BufRead + Send>,
    source: SourceInfo,
    format: StreamingFormat,
    line_number: usize,
    finished: bool,
}

enum StreamingFormat {
    Jsonl,
    Csv {
        /// Column index for text data
        text_col: Option<usize>,
    },
}

impl StreamingDatasetIterator {
    /// Create a streaming iterator from a file path.
    /// Auto-detects format from extension.
    pub fn open(path: &Path, source: SourceInfo) -> Result<Self, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let mut reader: Box<dyn BufRead + Send> = Box::new(BufReader::new(file));

        match ext.as_str() {
            "jsonl" | "ndjson" => Ok(Self {
                reader,
                source,
                format: StreamingFormat::Jsonl,
                line_number: 0,
                finished: false,
            }),
            "csv" => {
                // Parse header from existing reader (no re-open needed)
                let text_col = {
                    let mut header_buf = String::new();
                    reader
                        .read_line(&mut header_buf)
                        .map_err(|e| format!("CSV header error: {}", e))?;
                    let headers: Vec<String> = header_buf
                        .trim()
                        .split(',')
                        .map(|h| h.trim_matches('"').to_string())
                        .collect();
                    find_text_column(&headers)
                };

                Ok(Self {
                    reader,
                    source,
                    format: StreamingFormat::Csv { text_col },
                    line_number: 1,
                    finished: false,
                })
            }
            _ => Err(format!(
                "Streaming not supported for '.{}' format. Use load_dataset() for batch loading.",
                ext
            )),
        }
    }
}

impl Iterator for StreamingDatasetIterator {
    type Item = Result<DataSample, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match &self.format {
            StreamingFormat::Jsonl => {
                let mut line = String::new();
                loop {
                    line.clear();
                    match self.reader.read_line(&mut line) {
                        Ok(0) => {
                            self.finished = true;
                            return None;
                        }
                        Ok(_) => {
                            self.line_number += 1;
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                                Ok(v) => v,
                                Err(e) => {
                                    return Some(Err(format!(
                                        "JSONL parse error at line {}: {}",
                                        self.line_number, e
                                    )));
                                }
                            };
                            if let Some(text) = extract_text(&value) {
                                if !text.is_empty() {
                                    return Some(Ok(make_sample(text, &self.source)));
                                }
                            }
                            // No text field found, skip silently
                        }
                        Err(e) => {
                            self.finished = true;
                            return Some(Err(format!("Read error at line {}: {}", self.line_number, e)));
                        }
                    }
                }
            }
            StreamingFormat::Csv { text_col } => {
                let mut line = String::new();
                loop {
                    line.clear();
                    match self.reader.read_line(&mut line) {
                        Ok(0) => {
                            self.finished = true;
                            return None;
                        }
                        Ok(_) => {
                            self.line_number += 1;
                            // Skip header (first line)
                            if self.line_number == 1 {
                                continue;
                            }
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            let fields: Vec<&str> = trimmed.split(',').collect();
                            let text = if let Some(col) = text_col {
                                fields
                                    .get(*col)
                                    .map(|s| s.trim_matches('"').to_string())
                                    .unwrap_or_default()
                            } else {
                                fields.join(" ")
                            };
                            if !text.trim().is_empty() {
                                return Some(Ok(make_sample(text.trim().to_string(), &self.source)));
                            }
                        }
                        Err(e) => {
                            self.finished = true;
                            return Some(Err(format!("Read error at line {}: {}", self.line_number, e)));
                        }
                    }
                }
            }
        }
    }
}

/// Open a dataset file for streaming iteration (memory-efficient).
/// Supports JSONL and CSV. For other formats, use `load_dataset()`.
pub fn stream_dataset(path: &Path, source: SourceInfo) -> Result<StreamingDatasetIterator, String> {
    StreamingDatasetIterator::open(path, source)
}

/// Detect file format via magic bytes, returning the extension that should be used.
fn detect_format_by_magic(path: &Path) -> Option<&'static str> {
    let mut buf = [0u8; 8];
    let file = std::fs::File::open(path).ok()?;
    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    reader.read_exact(&mut buf).ok()?;

    match &buf[..4] {
        b"PAR1" => Some("parquet"),
        [0x28, 0xB5, 0x2F, 0xFD] => {
            warn!("{} appears to be Zstandard-compressed data. Decompress before loading or use a .arrow.zst extension.", path.display());
            None
        }
        [0x04, 0x22, 0x4D, 0x18] => {
            warn!("{} appears to be LZ4-compressed data. Decompress before loading or use a .arrow.lz4 extension.", path.display());
            None
        }
        _ => match &buf[..6] {
            b"ARROW1" => Some("arrow"),
            _ => None,
        },
    }
}

/// Detect format from file extension (with magic byte fallback) and load samples.
pub fn load_dataset(path: &Path, source: SourceInfo) -> Result<Vec<DataSample>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let load_by_ext = |ext: &str| -> Option<Result<Vec<DataSample>, String>> {
        match ext {
            "csv" => {
                info!("Loading CSV: {}", path.display());
                Some(load_csv(path, &source))
            }
            "tsv" => {
                info!("Loading TSV: {}", path.display());
                Some(load_tsv(path, &source))
            }
            "json" => {
                info!("Loading JSON: {}", path.display());
                Some(load_json(path, &source))
            }
            "jsonl" | "ndjson" => {
                info!("Loading JSONL: {}", path.display());
                Some(load_jsonl(path, &source))
            }
            "parquet" => {
                info!("Loading Parquet: {}", path.display());
                Some(load_parquet(path, &source))
            }
            "arrow" | "ipc" | "feather" => {
                info!("Loading Arrow: {}", path.display());
                Some(load_arrow(path, &source))
            }
            _ => None,
        }
    };

    // Detect by magic bytes first (more reliable than extension)
    if let Some(actual_ext) = detect_format_by_magic(path) {
        if actual_ext != ext {
            warn!(
                "File {} has extension '.{}' but detected as '.{}' format. Loading as {}.",
                path.display(),
                ext,
                actual_ext,
                actual_ext.to_uppercase(),
            );
        }
        if let Some(result) = load_by_ext(actual_ext) {
            return result;
        }
    }

    // Fallback: try extension-based detection
    if let Some(result) = load_by_ext(&ext) {
        return result;
    }

    Err(format!(
        "Unsupported dataset format '.{}' for file {}",
        ext,
        path.display()
    ))
}

fn make_sample(text: String, source: &SourceInfo) -> DataSample {
    let word_count = text.split_whitespace().count();
    DataSample {
        id: Uuid::new_v4(),
        text,
        token_ids: None,
        metadata: Default::default(),
        source: source.clone(),
        stats: crate::types::SampleStats {
            char_count: 0,
            word_count,
            token_count: 0,
            line_count: 0,
            entropy: 0.0,
            perplexity: 0.0,
            quality_score: 0.0,
        },
        domains: vec![],
        score: None,
        curriculum_level: None,
    }
}

// ── CSV ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "csv")]
fn load_csv(path: &Path, source: &SourceInfo) -> Result<Vec<DataSample>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| format!("CSV open error: {}", e))?;

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("CSV headers error: {}", e))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    let text_col = find_text_column(&headers);
    let mut samples = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| format!("CSV row error: {}", e))?;
        if let Some(col) = text_col {
            if let Some(text) = record.get(col) {
                let text = text.trim();
                if !text.is_empty() {
                    samples.push(make_sample(text.to_string(), &source));
                }
            }
        } else {
            let text: String = record.iter().collect::<Vec<&str>>().join(" ");
            if !text.trim().is_empty() {
                samples.push(make_sample(text.trim().to_string(), &source));
            }
        }
    }
    info!("Loaded {} samples from CSV", samples.len());
    Ok(samples)
}

// ── TSV ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "csv")]
fn load_tsv(path: &Path, source: &SourceInfo) -> Result<Vec<DataSample>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| format!("TSV open error: {}", e))?;

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("TSV headers error: {}", e))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    let text_col = find_text_column(&headers);
    let mut samples = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| format!("TSV row error: {}", e))?;
        if let Some(col) = text_col {
            if let Some(text) = record.get(col) {
                let text = text.trim();
                if !text.is_empty() {
                    samples.push(make_sample(text.to_string(), &source));
                }
            }
        } else {
            let text: String = record.iter().collect::<Vec<&str>>().join(" ");
            if !text.trim().is_empty() {
                samples.push(make_sample(text.trim().to_string(), &source));
            }
        }
    }
    info!("Loaded {} samples from TSV", samples.len());
    Ok(samples)
}

#[cfg(not(feature = "csv"))]
fn load_csv(_: &Path, _: &SourceInfo) -> Result<Vec<DataSample>, String> {
    Err("CSV support not enabled (feature 'csv')".into())
}

#[cfg(not(feature = "csv"))]
fn load_tsv(_: &Path, _: &SourceInfo) -> Result<Vec<DataSample>, String> {
    Err("TSV support not enabled (feature 'csv')".into())
}

fn find_text_column(headers: &[String]) -> Option<usize> {
    // Common column names for text data
    let text_names = [
        "text",
        "content",
        "body",
        "sentence",
        "document",
        "input",
        "prompt",
        "instruction",
        "code",
        "title",
        "teks",
        "konten",
        "kalimat",
    ];
    headers.iter().position(|h| {
        let h_lower = h.to_lowercase().trim().to_string();
        text_names.contains(&h_lower.as_str())
    })
}

// ── JSON array ───────────────────────────────────────────────────────────────

#[cfg(feature = "json")]
fn load_json(path: &Path, source: &SourceInfo) -> Result<Vec<DataSample>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("JSON read error: {}", e))?;

    let value: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut samples = Vec::new();

    match &value {
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(text) = extract_text(item) {
                    if !text.is_empty() {
                        samples.push(make_sample(text, &source));
                    }
                }
            }
        }
        serde_json::Value::Object(_) => {
            if let Some(text) = extract_text(&value) {
                if !text.is_empty() {
                    samples.push(make_sample(text, &source));
                }
            }
        }
        _ => return Err("JSON must be an array or object with 'text' field".into()),
    }

    info!("Loaded {} samples from JSON", samples.len());
    Ok(samples)
}

// ── JSONL / NDJSON ──────────────────────────────────────────────────────────

#[cfg(feature = "json")]
fn load_jsonl(path: &Path, source: &SourceInfo) -> Result<Vec<DataSample>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("JSONL read error: {}", e))?;

    let mut samples = Vec::new();

    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("JSONL parse error at line {}: {}", i + 1, e))?;
        if let Some(text) = extract_text(&value) {
            if !text.is_empty() {
                samples.push(make_sample(text, &source));
            }
        } else {
            warn!("JSONL line {}: no text field found, skipping", i + 1);
        }
    }

    info!("Loaded {} samples from JSONL", samples.len());
    Ok(samples)
}

#[cfg(not(feature = "json"))]
fn load_json(_: &Path, _: &SourceInfo) -> Result<Vec<DataSample>, String> {
    Err("JSON support not enabled (feature 'json')".into())
}

#[cfg(not(feature = "json"))]
fn load_jsonl(_: &Path, _: &SourceInfo) -> Result<Vec<DataSample>, String> {
    Err("JSONL support not enabled (feature 'json')".into())
}

/// Extract text field from a JSON value, checking common field names.
fn extract_text(value: &serde_json::Value) -> Option<String> {
    let obj = value.as_object()?;
    let text_names = [
        "text",
        "content",
        "body",
        "sentence",
        "document",
        "input",
        "prompt",
        "instruction",
        "code",
        "title",
        "teks",
        "konten",
        "kalimat",
        "messages",
        "chat",
        "conversation",
    ];
    for name in &text_names {
        if let Some(v) = obj.get(*name) {
            match v {
                serde_json::Value::String(s) => return Some(s.clone()),
                serde_json::Value::Array(arr) => {
                    // messages/chat array: concatenate content fields
                    let parts: Vec<String> = arr
                        .iter()
                        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                        .map(|s| s.to_string())
                        .collect();
                    if !parts.is_empty() {
                        return Some(parts.join("\n"));
                    }
                }
                _ => {}
            }
        }
    }
    None
}

// ── Parquet ─────────────────────────────────────────────────────────────────

#[cfg(feature = "parquet")]
fn load_parquet(path: &Path, source: &SourceInfo) -> Result<Vec<DataSample>, String> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file = std::fs::File::open(path).map_err(|e| format!("Parquet open error: {}", e))?;

    let reader =
        SerializedFileReader::new(file).map_err(|e| format!("Parquet reader error: {}", e))?;

    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();
    let num_rows = file_metadata.num_rows() as usize;

    let mut samples = Vec::with_capacity(num_rows);

    let iter = reader
        .get_row_iter(None)
        .map_err(|e| format!("Parquet row iterator error: {}", e))?;

    for result in iter {
        let row = match result {
            Ok(r) => r,
            Err(e) => {
                warn!("Parquet row error: {}", e);
                continue;
            }
        };
        let text = extract_text_from_parquet_row(&row);
        if let Some(text) = text {
            if !text.is_empty() {
                samples.push(make_sample(text, &source));
            }
        }
    }

    info!("Loaded {} samples from Parquet", samples.len());
    Ok(samples)
}

#[cfg(feature = "parquet")]
fn extract_text_from_parquet_row(row: &parquet::record::Row) -> Option<String> {
    let text_names = [
        "text",
        "content",
        "body",
        "sentence",
        "document",
        "input",
        "prompt",
        "instruction",
        "code",
        "title",
        "teks",
        "konten",
        "kalimat",
    ];

    for (name, field) in row.get_column_iter() {
        if text_names.contains(&name.as_str()) {
            if let Some(s) = field_to_string(field) {
                let s = s.trim().to_string();
                if !s.is_empty() {
                    return Some(s);
                }
            }
        }
    }
    None
}

#[cfg(feature = "parquet")]
fn field_to_string(field: &parquet::record::Field) -> Option<String> {
    use parquet::record::Field;
    match field {
        Field::Str(s) => Some(s.clone()),
        Field::Bytes(b) => match String::from_utf8(b.data().to_vec()) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!("Invalid UTF-8 in parquet bytes field: {}", e);
                None
            }
        },
        Field::Long(v) => Some(v.to_string()),
        Field::Int(v) => Some(v.to_string()),
        Field::Double(v) => Some(v.to_string()),
        Field::Float(v) => Some(v.to_string()),
        Field::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

#[cfg(not(feature = "parquet"))]
fn load_parquet(_: &Path, _: &SourceInfo) -> Result<Vec<DataSample>, String> {
    Err("Parquet support not enabled (feature 'parquet')".into())
}

// ── Arrow (delegate to existing arrow_reader) ────────────────────────────────

#[cfg(feature = "arrow")]
fn load_arrow(path: &Path, source: &SourceInfo) -> Result<Vec<DataSample>, String> {
    crate::arrow_reader::read_arrow_file(path, source.clone())
        .map_err(|e| format!("Arrow load error: {}", e))
}

#[cfg(not(feature = "arrow"))]
fn load_arrow(_: &Path, _: &SourceInfo) -> Result<Vec<DataSample>, String> {
    Err("Arrow support not enabled (feature 'arrow')".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_csv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        std::fs::write(&path, "text,score\nHello world,0.9\nFoo bar,0.8\n").unwrap();
        let src = SourceInfo {
            name: "test".into(),
            url: None,
            trust_score: 1.0,
            category: crate::types::SourceCategory::Other,
            fetch_timestamp: 0,
        };
        let samples = load_csv(&path, &src).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].text, "Hello world");
        assert_eq!(samples[1].text, "Foo bar");
    }

    #[test]
    fn test_load_json_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        std::fs::write(&path, r#"[{"text":"Hello"},{"text":"World"}]"#).unwrap();
        let src = SourceInfo {
            name: "test".into(),
            url: None,
            trust_score: 1.0,
            category: crate::types::SourceCategory::Other,
            fetch_timestamp: 0,
        };
        let samples = load_json(&path, &src).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].text, "Hello");
        assert_eq!(samples[1].text, "World");
    }

    #[test]
    fn test_load_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        std::fs::write(
            &path,
            r#"{"text":"Line1"}
{"text":"Line2"}
{"text":"Line3"}"#,
        )
        .unwrap();
        let src = SourceInfo {
            name: "test".into(),
            url: None,
            trust_score: 1.0,
            category: crate::types::SourceCategory::Other,
            fetch_timestamp: 0,
        };
        let samples = load_jsonl(&path, &src).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].text, "Line1");
        assert_eq!(samples[2].text, "Line3");
    }

    #[test]
    fn test_load_tsv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tsv");
        std::fs::write(&path, "text\tscore\nHello\t0.9\nWorld\t0.8\n").unwrap();
        let src = SourceInfo {
            name: "test".into(),
            url: None,
            trust_score: 1.0,
            category: crate::types::SourceCategory::Other,
            fetch_timestamp: 0,
        };
        let samples = load_tsv(&path, &src).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].text, "Hello");
        assert_eq!(samples[1].text, "World");
    }

    #[test]
    fn test_find_text_column() {
        let headers = vec!["id".into(), "text".into(), "score".into()];
        assert_eq!(find_text_column(&headers), Some(1));
        let headers = vec!["content".into(), "label".into()];
        assert_eq!(find_text_column(&headers), Some(0));
        let headers = vec!["a".into(), "b".into()];
        assert_eq!(find_text_column(&headers), None);
    }

    #[test]
    fn test_extract_text() {
        let v: serde_json::Value = serde_json::from_str(r#"{"text":"hello"}"#).unwrap();
        assert_eq!(extract_text(&v), Some("hello".into()));

        let v: serde_json::Value =
            serde_json::from_str(r#"{"instruction":"do X","output":"Y"}"#).unwrap();
        assert_eq!(extract_text(&v), Some("do X".into()));

        let v: serde_json::Value = serde_json::from_str(r#"{"nope":123}"#).unwrap();
        assert_eq!(extract_text(&v), None);
    }

    #[test]
    fn test_extract_text_messages() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]}"#
        ).unwrap();
        let result = extract_text(&v);
        assert_eq!(result, Some("Hi\nHello!".into()));
    }

    #[test]
    fn test_format_detection() {
        let dir = tempfile::tempdir().unwrap();

        let csv = dir.path().join("data.csv");
        std::fs::write(&csv, "text\nHello\n").unwrap();
        let src = SourceInfo {
            name: "test".into(),
            url: None,
            trust_score: 1.0,
            category: crate::types::SourceCategory::Other,
            fetch_timestamp: 0,
        };
        assert!(load_dataset(&csv, src.clone()).is_ok());

        let json = dir.path().join("data.json");
        std::fs::write(&json, r#"[{"text":"Hello"}]"#).unwrap();
        assert!(load_dataset(&json, src.clone()).is_ok());

        let jsonl = dir.path().join("data.jsonl");
        std::fs::write(&jsonl, r#"{"text":"Hello"}"#).unwrap();
        assert!(load_dataset(&jsonl, src.clone()).is_ok());

        let tsv = dir.path().join("data.tsv");
        std::fs::write(&tsv, "text\tval\nHello\t1\n").unwrap();
        assert!(load_dataset(&tsv, src.clone()).is_ok());

        let bad = dir.path().join("data.xyz");
        assert!(load_dataset(&bad, src).is_err());
    }
}
