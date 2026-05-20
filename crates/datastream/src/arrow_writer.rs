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
