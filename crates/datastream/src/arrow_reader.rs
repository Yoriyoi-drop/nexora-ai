use std::path::Path;
use anyhow::{Context, Result};
use uuid::Uuid;

use arrow::array::{Array, StringArray, LargeStringArray};
use crate::types::{DataSample, SourceInfo, SampleStats};

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
    let text_idx = schema.index_of("text")
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
