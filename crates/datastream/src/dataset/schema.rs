use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{error, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSchema {
    pub fields: Vec<FieldSchema>,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub dtype: String,
    pub nullable: bool,
}

impl DatasetSchema {
    pub fn text() -> Self {
        Self {
            fields: vec![FieldSchema {
                name: "text".into(),
                dtype: "utf8".into(),
                nullable: false,
            }],
            format: "arrow".into(),
        }
    }

    pub fn tokenized() -> Self {
        Self {
            fields: vec![
                FieldSchema {
                    name: "text".into(),
                    dtype: "utf8".into(),
                    nullable: false,
                },
                FieldSchema {
                    name: "tokens".into(),
                    dtype: "list<int32>".into(),
                    nullable: true,
                },
                FieldSchema {
                    name: "label".into(),
                    dtype: "int64".into(),
                    nullable: true,
                },
            ],
            format: "arrow".into(),
        }
    }

    #[cfg(feature = "arrow")]
    pub fn validate_arrow(&self, arrow_schema: &arrow::datatypes::Schema) -> SchemaValidation {
        let mut issues = Vec::with_capacity(self.fields.len());

        for field in &self.fields {
            match arrow_schema.index_of(&field.name) {
                Ok(idx) => {
                    let actual = &arrow_schema.field(idx);
                    let actual_dtype = format!("{:?}", actual.data_type()).to_lowercase();
                    let expected_dtype = field.dtype.to_lowercase();
                    if actual_dtype != expected_dtype {
                        issues.push(SchemaIssue::TypeMismatch {
                            field: field.name.clone(),
                            expected: field.dtype.clone(),
                            actual: actual_dtype,
                        });
                    }
                }
                Err(_) => {
                    if !field.nullable {
                        issues.push(SchemaIssue::MissingColumn {
                            field: field.name.clone(),
                        });
                    }
                }
            }
        }

        SchemaValidation {
            valid: issues.is_empty(),
            issues,
        }
    }
}

#[derive(Debug)]
pub struct SchemaValidation {
    pub valid: bool,
    pub issues: Vec<SchemaIssue>,
}

#[derive(Debug)]
pub enum SchemaIssue {
    MissingColumn {
        field: String,
    },
    TypeMismatch {
        field: String,
        expected: String,
        actual: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptedShardAction {
    Skip,
    Warn,
    Fail,
}

impl Default for CorruptedShardAction {
    fn default() -> Self {
        CorruptedShardAction::Warn
    }
}

pub struct CorruptedShardRecovery {
    pub action: CorruptedShardAction,
    pub max_failures: usize,
    failures: usize,
    skipped: Vec<String>,
}

impl CorruptedShardRecovery {
    pub fn new(action: CorruptedShardAction) -> Self {
        Self {
            action,
            max_failures: 5,
            failures: 0,
            skipped: Vec::new(),
        }
    }

    pub fn handle_failure(&mut self, path: &Path, error: &str) -> Result<(), ()> {
        self.failures += 1;

        match self.action {
            CorruptedShardAction::Skip => {
                warn!("Corrupted shard (skip): {} - {}", path.display(), error);
                self.skipped.push(path.to_string_lossy().to_string());
                Ok(())
            }
            CorruptedShardAction::Warn => {
                warn!("Corrupted shard (warn): {} - {}", path.display(), error);
                self.skipped.push(path.to_string_lossy().to_string());
                if self.failures >= self.max_failures {
                    error!("Too many corrupted shards ({}), failing", self.failures);
                    Err(())
                } else {
                    Ok(())
                }
            }
            CorruptedShardAction::Fail => {
                error!("Corrupted shard (fail): {} - {}", path.display(), error);
                Err(())
            }
        }
    }

    pub fn skipped_shards(&self) -> &[String] {
        &self.skipped
    }

    pub fn total_failures(&self) -> usize {
        self.failures
    }
}

impl Default for CorruptedShardRecovery {
    fn default() -> Self {
        Self::new(CorruptedShardAction::Warn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_schema_text() {
        let schema = DatasetSchema::text();
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].name, "text");
        assert_eq!(schema.fields[0].dtype, "utf8");
        assert!(!schema.fields[0].nullable);
    }

    #[test]
    fn test_dataset_schema_tokenized() {
        let schema = DatasetSchema::tokenized();
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.fields[1].name, "tokens");
        assert_eq!(schema.fields[1].dtype, "list<int32>");
    }

    #[test]
    fn test_corrupted_shard_action_default() {
        assert_eq!(CorruptedShardAction::default(), CorruptedShardAction::Warn);
    }

    #[test]
    fn test_corrupted_shard_recovery_new() {
        let r = CorruptedShardRecovery::new(CorruptedShardAction::Fail);
        assert_eq!(r.action, CorruptedShardAction::Fail);
        assert_eq!(r.total_failures(), 0);
    }

    #[test]
    fn test_corrupted_shard_recovery_skip() {
        let mut r = CorruptedShardRecovery::new(CorruptedShardAction::Skip);
        let result = r.handle_failure(Path::new("test.arrow"), "corrupted");
        assert!(result.is_ok());
        assert_eq!(r.total_failures(), 1);
        assert_eq!(r.skipped_shards().len(), 1);
    }

    #[test]
    fn test_corrupted_shard_recovery_fail() {
        let mut r = CorruptedShardRecovery::new(CorruptedShardAction::Fail);
        let result = r.handle_failure(Path::new("test.arrow"), "fatal");
        assert!(result.is_err());
        assert_eq!(r.total_failures(), 1);
    }

    #[test]
    fn test_corrupted_shard_recovery_warn_max_failures() {
        let mut r = CorruptedShardRecovery::new(CorruptedShardAction::Warn);
        r.max_failures = 3;
        for i in 0..3 {
            let result = r.handle_failure(Path::new(&format!("{}.arrow", i)), "error");
            assert_eq!(result.is_ok(), i < 2); // third failure should Err
        }
        assert_eq!(r.total_failures(), 3);
    }

    #[test]
    fn test_schema_validation() {
        let validation = SchemaValidation {
            valid: true,
            issues: vec![],
        };
        assert!(validation.valid);
        assert!(validation.issues.is_empty());
    }

    #[test]
    fn test_schema_issue_missing_column() {
        let issue = SchemaIssue::MissingColumn {
            field: "text".into(),
        };
        assert!(matches!(issue, SchemaIssue::MissingColumn { .. }));
    }
}
