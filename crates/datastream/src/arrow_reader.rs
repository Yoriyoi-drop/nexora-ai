use anyhow::{bail, Context, Result};
use std::io::{Cursor, Read};
use std::path::Path;
use uuid::Uuid;

use crate::types::{DataSample, SampleStats, SourceInfo};
use arrow::array::{Array, LargeStringArray, StringArray};
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;

/// Check magic bytes to help diagnose mislabeled files.
/// Returns a description of the detected format, or None if unrecognized.
fn detect_actual_format(path: &Path) -> Option<&'static str> {
    let mut buf = [0u8; 8];
    let file = std::fs::File::open(path).ok()?;
    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    reader.read_exact(&mut buf).ok()?;

    match &buf[..4] {
        b"PAR1" => Some("Apache Parquet"),
        [0x28, 0xB5, 0x2F, 0xFD] => Some("Zstandard compressed data"),
        [0x04, 0x22, 0x4D, 0x18] => Some("LZ4 compressed data"),
        _ => match &buf[..6] {
            b"ARROW1" => None, // correct format
            _ => Some("unknown (not Arrow IPC File format)"),
        },
    }
}

fn get_text_value(col: &dyn Array, i: usize) -> Option<String> {
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        if !arr.is_null(i) {
            return Some(arr.value(i).to_string());
        }
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        if !arr.is_null(i) {
            return Some(arr.value(i).to_string());
        }
    }
    None
}

fn is_zstd_compressed(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 4];
    if let Ok(file) = std::fs::File::open(path) {
        let mut reader = std::io::BufReader::new(file);
        if reader.read_exact(&mut buf).is_ok() {
            return &buf == &[0x28, 0xB5, 0x2F, 0xFD];
        }
    }
    false
}

fn decompress_zstd(path: &Path) -> Result<std::fs::File> {
    #[cfg(feature = "compression-zstd")]
    {
        let compressed = std::fs::read(path)
            .with_context(|| format!("Failed to read compressed file: {}", path.display()))?;

        let decoder = zstd::decode_all(&compressed[..])
            .with_context(|| format!("Failed to decompress zstd file: {}", path.display()))?;

        // Write to temp file
        let temp_path = path.with_extension("arrow.tmp");
        std::fs::write(&temp_path, decoder)
            .with_context(|| format!("Failed to write decompressed data: {}", temp_path.display()))?;

        std::fs::File::open(&temp_path)
            .with_context(|| format!("Failed to open decompressed file: {}", temp_path.display()))
    }

    #[cfg(not(feature = "compression-zstd"))]
    {
        anyhow::bail!("zstd compression support not enabled. Rebuild with --features compression-zstd or decompress the file manually: zstd -d {} -o {}", path.display(), path.with_extension("arrow").display())
    }
}

/// Streaming iterator over arrow record batches.
/// Yields one `Vec<DataSample>` per record batch — callers control memory
/// by not holding all batches simultaneously.
pub struct ArrowBatchStream {
    reader: FileReader<std::fs::File>,
    source: SourceInfo,
    text_idx: usize,
    output_idx: Option<usize>,
}

impl ArrowBatchStream {
    pub fn try_new(path: &Path, source: SourceInfo) -> Result<Self> {
        let file = if is_zstd_compressed(path) {
            decompress_zstd(path)?
        } else {
            std::fs::File::open(path)
                .with_context(|| format!("Failed to open arrow file: {}", path.display()))?
        };

        let reader = match FileReader::try_new(file, None) {
            Ok(r) => r,
            Err(err) => {
                let hint = detect_actual_format(path)
                    .map(|fmt| format!(". Detected format: {fmt}. If this is a {fmt} file, rename to .parquet or use the correct --data format."))
                    .unwrap_or_default();
                bail!(
                    "Failed to read arrow IPC file: {}\n  Reason: {}\n  Hint: File has .arrow extension but is not valid Arrow IPC format{}",
                    path.display(), err, hint,
                );
            }
        };

        let schema = reader.schema();
        let text_idx = schema
            .index_of("text")
            .or_else(|_| schema.index_of("Text"))
            .or_else(|_| schema.index_of("input"))
            .or_else(|_| schema.index_of("Input"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Arrow file must have a 'text', 'Text', or 'input' column. Found columns: {:?}",
                    schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
                )
            })?;

        let output_idx = schema
            .index_of("output")
            .or_else(|_| schema.index_of("Output"))
            .ok();

        Ok(Self { reader, source, text_idx, output_idx })
    }
}

impl Iterator for ArrowBatchStream {
    type Item = Result<Vec<DataSample>>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.reader.next() {
            Some(Ok(b)) => b,
            Some(Err(e)) => return Some(Err(e.into())),
            None => return None,
        };

        let col = batch.column(self.text_idx);
        let mut samples = Vec::with_capacity(col.len());
        for i in 0..col.len() {
            let text = get_text_value(col, i).unwrap_or_default();
            let text = if let Some(out_idx) = &self.output_idx {
                let output_col = batch.column(*out_idx);
                let output = get_text_value(output_col, i).unwrap_or_default();
                if output.is_empty() {
                    text
                } else if text.is_empty() {
                    output
                } else {
                    format!("{}\n{}", text, output)
                }
            } else {
                text
            };
            samples.push(DataSample {
                id: Uuid::new_v4(),
                text,
                token_ids: None,
                metadata: std::collections::HashMap::new(),
                source: self.source.clone(),
                stats: SampleStats::default(),
                domains: vec![],
                score: None,
                curriculum_level: None,
            });
        }
        Some(Ok(samples))
    }
}

/// Streaming batch reader from in-memory arrow bytes.
pub struct ArrowBytesStream {
    reader: FileReader<Cursor<Vec<u8>>>,
    source: SourceInfo,
    text_idx: usize,
}

impl ArrowBytesStream {
    pub fn try_new(data: Vec<u8>, source: SourceInfo) -> Result<Self> {
        let cursor = Cursor::new(data);
        let reader = FileReader::try_new(cursor, None)
            .with_context(|| "Failed to read arrow IPC from bytes".to_string())?;
        let schema = reader.schema();
        let text_idx = schema
            .index_of("text")
            .or_else(|_| schema.index_of("Text"))
            .or_else(|_| schema.index_of("input"))
            .or_else(|_| schema.index_of("content"))
            .map_err(|_| anyhow::anyhow!("No text column found in arrow schema"))?;
        Ok(Self { reader, source, text_idx })
    }
}

impl Iterator for ArrowBytesStream {
    type Item = Result<Vec<DataSample>>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.reader.next() {
            Some(Ok(b)) => b,
            Some(Err(e)) => return Some(Err(e.into())),
            None => return None,
        };
        let col = batch.column(self.text_idx);
        let mut samples = Vec::with_capacity(col.len());
        for i in 0..col.len() {
            if let Some(text) = get_text_value(col, i) {
                samples.push(DataSample {
                    id: Uuid::new_v4(),
                    text,
                    token_ids: None,
                    metadata: std::collections::HashMap::new(),
                    source: self.source.clone(),
                    stats: SampleStats::default(),
                    domains: vec![],
                    score: None,
                    curriculum_level: None,
                });
            }
        }
        Some(Ok(samples))
    }
}

pub fn read_arrow_file(path: &Path, source: SourceInfo) -> Result<Vec<DataSample>> {
    let mut all_samples = Vec::new();
    for batch_result in ArrowBatchStream::try_new(path, source)? {
        let mut batch = batch_result?;
        all_samples.append(&mut batch);
    }
    Ok(all_samples)
}

/// Read arrow IPC data from in-memory bytes.
pub fn read_arrow_bytes(data: &[u8], source: SourceInfo) -> Result<Vec<DataSample>> {
    let mut all_samples = Vec::new();
    for batch_result in ArrowBytesStream::try_new(data.to_vec(), source)? {
        let mut batch = batch_result?;
        all_samples.append(&mut batch);
    }
    Ok(all_samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_writer;
    use crate::types::{SampleStats, SourceCategory, SourceInfo};
    use std::path::Path;

    fn source_info() -> SourceInfo {
        SourceInfo {
            name: "test".into(),
            url: None,
            trust_score: 0.5,
            category: SourceCategory::Other,
            fetch_timestamp: 0,
        }
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_arrow_file(Path::new("/nonexistent/file.arrow"), source_info());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_text_value_string_array() {
        use arrow::array::StringArray;
        let arr = StringArray::from(vec![Some("hello"), None, Some("world")]);
        assert_eq!(get_text_value(&arr, 0), Some("hello".to_string()));
        assert_eq!(get_text_value(&arr, 1), None);
        assert_eq!(get_text_value(&arr, 2), Some("world".to_string()));
    }

    #[test]
    fn test_write_then_read_roundtrip() {
        use crate::types::DataSample;
        use uuid::Uuid;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.arrow");

        let sample = DataSample {
            id: Uuid::new_v4(),
            text: "roundtrip test content".into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: source_info(),
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        };

        arrow_writer::write_arrow_file(&[sample], &path).unwrap();
        let samples = read_arrow_file(&path, source_info()).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].text, "roundtrip test content");
    }
}
