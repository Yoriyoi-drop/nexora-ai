use anyhow::{Context, Result};
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use std::path::Path;
use std::sync::Arc;

use crate::types::DataSample;

pub fn write_arrow_file(samples: &[DataSample], path: &Path) -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("source_name", DataType::Utf8, false),
        Field::new("source_url", DataType::Utf8, true),
        Field::new("source_category", DataType::Utf8, false),
        Field::new("trust_score", DataType::Float64, false),
        Field::new("fetch_timestamp", DataType::Int64, false),
    ]);

    let n = samples.len();
    let mut ids = Vec::with_capacity(n);
    let mut texts = Vec::with_capacity(n);
    let mut source_names = Vec::with_capacity(n);
    let mut source_urls: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut source_cats = Vec::with_capacity(n);
    let mut trust_scores = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);

    for s in samples {
        ids.push(s.id.to_string());
        texts.push(s.text.clone());
        source_names.push(s.source.name.clone());
        source_urls.push(s.source.url.as_deref());
        source_cats.push(format!("{:?}", s.source.category));
        trust_scores.push(s.source.trust_score);
        timestamps.push(s.source.fetch_timestamp);
    }

    let id_array = StringArray::from(ids);
    let text_array = StringArray::from(texts);
    let src_name_array = StringArray::from(source_names);
    let src_url_array = StringArray::from(source_urls);
    let src_cat_array = StringArray::from(source_cats);
    let trust_array = arrow::array::Float64Array::from(trust_scores);
    let ts_array = arrow::array::Int64Array::from(timestamps);

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_array),
            Arc::new(text_array),
            Arc::new(src_name_array),
            Arc::new(src_url_array),
            Arc::new(src_cat_array),
            Arc::new(trust_array),
            Arc::new(ts_array),
        ],
    )
    .context("Failed to create RecordBatch")?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create arrow file: {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref())
        .context("Failed to create Arrow FileWriter")?;
    writer
        .write(&batch)
        .context("Failed to write RecordBatch to Arrow file")?;
    writer.finish().context("Failed to finalize Arrow file")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};
    use std::path::Path;
    use uuid::Uuid;

    fn sample() -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: "hello world".into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.8,
                category: SourceCategory::Other,
                fetch_timestamp: 12345,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_write_arrow_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.arrow");
        let samples = vec![sample()];
        let result = write_arrow_file(&samples, &path);
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_write_arrow_empty_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.arrow");
        let result = write_arrow_file(&[], &path);
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_write_multiple_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.arrow");
        let samples = vec![sample(), sample(), sample()];
        let result = write_arrow_file(&samples, &path);
        assert!(result.is_ok());

        // Verify we can read it back
        let read_result =
            arrow::ipc::reader::FileReader::try_new(std::fs::File::open(&path).unwrap(), None);
        assert!(read_result.is_ok());
    }

    #[test]
    fn test_write_arrow_invalid_path() {
        let path = Path::new("/nonexistent/dir/output.arrow");
        let samples = vec![sample()];
        let result = write_arrow_file(&samples, path);
        assert!(result.is_err());
    }
}
