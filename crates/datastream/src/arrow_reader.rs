use anyhow::{Context, Result};
use std::path::Path;
use uuid::Uuid;

use crate::types::{DataSample, SampleStats, SourceInfo};
use arrow::array::{Array, LargeStringArray, StringArray};

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

pub fn read_arrow_file(path: &Path, source: SourceInfo) -> Result<Vec<DataSample>> {
    use arrow::ipc::reader::FileReader;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open arrow file: {}", path.display()))?;

    let reader = FileReader::try_new(file, None)
        .with_context(|| format!("Failed to read arrow IPC file: {}", path.display()))?;

    let schema = reader.schema();
    let text_idx = schema
        .index_of("text")
        .or_else(|_| schema.index_of("Text"))
        .map_err(|_| anyhow::anyhow!("Arrow file must have a 'text' or 'Text' column"))?;

    let mut samples = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let col = batch.column(text_idx);

        for i in 0..col.len() {
            let text = get_text_value(col, i).unwrap_or_default();
            samples.push(DataSample {
                id: Uuid::new_v4(),
                text,
                token_ids: None,
                metadata: std::collections::HashMap::new(),
                source: source.clone(),
                stats: SampleStats::default(),
                domains: vec![],
                score: None,
                curriculum_level: None,
            });
        }
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SampleStats, SourceInfo, SourceCategory};
    use crate::arrow_writer;
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
