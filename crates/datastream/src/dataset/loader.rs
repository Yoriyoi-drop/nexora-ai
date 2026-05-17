use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Semaphore};
use tokio::task;
use tracing::{info, warn, error, debug};

use crate::types::{DataSample, SourceInfo, SourceCategory, SampleStats};
use crate::arrow_reader;
use super::scanner::{ShardPath, ShardScanner};
use super::compression::Compression;
use super::progress::{ProgressTracker, ResumeState, ResumeError, StreamingStats};
use super::manifest::{DatasetManifest, ManifestError};
use super::schema::{DatasetSchema, SchemaValidation, CorruptedShardRecovery, CorruptedShardAction};
use super::cache::{DatasetCache, TokenizerCache};
use super::iterator::BatchIterator;
use super::shuffle::shuffle_shards;

#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub batch_size: usize,
    pub prefetch_batches: usize,
    pub num_workers: usize,
    pub shuffle_buffer: usize,
    pub seq_length: usize,
    pub resume: bool,
    pub cache_dir: Option<PathBuf>,
    pub schema: Option<DatasetSchema>,
    pub corrupted_shard_action: CorruptedShardAction,
    pub progress_report_interval_secs: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            prefetch_batches: 4,
            num_workers: 2,
            shuffle_buffer: 10000,
            seq_length: 128,
            resume: false,
            cache_dir: None,
            schema: Some(DatasetSchema::text()),
            corrupted_shard_action: CorruptedShardAction::Warn,
            progress_report_interval_secs: 5,
        }
    }
}

pub struct StreamingLoader {
    pub config: StreamingConfig,
    pub progress: ProgressTracker,
    pub stats: StreamingStats,
    batch_rx: mpsc::Receiver<Vec<DataSample>>,
    shard_paths: Vec<ShardPath>,
    current_shard_idx: usize,
    current_sample_offset: u64,
    recovery: CorruptedShardRecovery,
    cache: Option<Arc<DatasetCache>>,
    batch_iterator: BatchIterator,
    total_epochs: usize,
    current_epoch: usize,
    epoch_samples_loaded: u64,
    manifest: Option<DatasetManifest>,
    resume_state_path: Option<PathBuf>,
}

impl StreamingLoader {
    pub async fn new(path: &Path, config: StreamingConfig) -> Result<Self, LoaderError> {
        let total_epochs = 1;
        let mut recovery = CorruptedShardRecovery::new(config.corrupted_shard_action);

        // --- 1. Load manifest if exists ---
        let (manifest, base_path) = match DatasetManifest::from_path(&path.join("manifest.json")) {
            Ok(m) => {
                info!("Manifest loaded: {} v{} ({} samples, {} shards)",
                    m.name, m.version, m.total_samples, m.total_shards);
                (Some(m), path.to_path_buf())
            }
            Err(ManifestError::NotFound(_)) | Err(ManifestError::Io(_)) => {
                // No manifest — scan directory directly
                (None, path.to_path_buf())
            }
            Err(e) => {
                warn!("Manifest parse error (will scan directly): {}", e);
                (None, path.to_path_buf())
            }
        };

        // --- 2. Scan shards ---
        let scanner = ShardScanner::new();
        let mut shard_paths = scanner.scan(&base_path);
        if shard_paths.is_empty() {
            return Err(LoaderError::NoShards(base_path.to_string_lossy().to_string()));
        }
        info!("Scanned {} shards from {}", shard_paths.len(), base_path.display());

        // --- 3. Estimate total samples ---
        let total_samples: u64 = if let Some(ref m) = manifest {
            m.total_for_split("train").max(1)
        } else {
            shard_paths.iter().map(|s| estimate_samples(&s.path).unwrap_or(0)).sum::<u64>().max(1)
        };

        // --- 4. Initialize cache ---
        let cache = config.cache_dir.as_ref().map(|d| {
            Arc::new(DatasetCache::new(d.join("dataset_cache")))
        });
        if let Some(ref c) = cache {
            info!("Dataset cache enabled: {}", c.cache_dir().display());
        }

        // --- 5. Initialize components ---
        let mut progress = ProgressTracker::new(total_samples, total_epochs);
        let stats = StreamingStats::new();
        let (batch_tx, batch_rx) = mpsc::channel(config.prefetch_batches);
        let batch_iterator = BatchIterator::new(config.batch_size, config.shuffle_buffer);

        // --- 6. Handle resume ---
        let resume_state_path = config.resume.then(|| base_path.join("resume_state.json"));
        let (start_epoch, start_shard, start_offset) = if let Some(ref rsp) = resume_state_path {
            match ResumeState::load(rsp) {
                Ok(state) => {
                    info!("Resuming from epoch {}, shard {}, offset {}",
                        state.epoch, state.shard_index, state.sample_offset);
                    progress.start_epoch(state.epoch);
                    (state.epoch, state.shard_index, state.sample_offset)
                }
                Err(ResumeError::Io(_)) => {
                    info!("No resume state found, starting fresh");
                    progress.start_epoch(1);
                    (1, 0, 0)
                }
                Err(e) => {
                    warn!("Resume state error (starting fresh): {}", e);
                    progress.start_epoch(1);
                    (1, 0, 0)
                }
            }
        } else {
            progress.start_epoch(1);
            (1, 0, 0)
        };

        // --- 7. Validate schema on first shard if configured ---
        if let Some(ref schema) = config.schema {
            if let Some(first_compatible) = shard_paths.iter().find(|s| is_arrow_file(&s.path)) {
                debug!("Validating schema against: {}", first_compatible.path.display());
                let validation = validate_shard_schema(schema, &first_compatible.path);
                if let Ok(validation) = validation {
                    if !validation.valid {
                        for issue in &validation.issues {
                            warn!("Schema issue on {}: {:?}", first_compatible.path.display(), issue);
                        }
                    }
                } else if let Err(e) = validation {
                    recovery.handle_failure(&first_compatible.path, &e.to_string()).ok();
                }
            }
        }

        // --- 8. Spawn workers ---
        let loader = Self {
            config: config.clone(),
            progress,
            stats,
            batch_rx,
            shard_paths,
            current_shard_idx: start_shard,
            current_sample_offset: start_offset,
            recovery,
            cache,
            batch_iterator,
            total_epochs,
            current_epoch: start_epoch,
            epoch_samples_loaded: 0,
            manifest,
            resume_state_path,
        };

        loader.spawn_workers(batch_tx).await;

        Ok(loader)
    }

    async fn spawn_workers(&self, batch_tx: mpsc::Sender<Vec<DataSample>>) {
        let num_workers = self.config.num_workers;
        let batch_size = self.config.batch_size;
        let semaphore = Arc::new(Semaphore::new(num_workers));
        let start_shard = self.current_shard_idx;
        let start_offset = self.current_sample_offset;

        let mut shards = self.shard_paths.clone();
        // Shuffle shards for better mixing each epoch
        if self.current_epoch > 1 || start_shard == 0 {
            shuffle_shards(&mut shards);
        }

        let skip_shards = start_shard.min(shards.len());

        for (idx, shard) in shards.into_iter().enumerate().skip(skip_shards) {
            let permit = semaphore.clone().acquire_owned().await;
            if permit.is_err() {
                warn!("Failed to acquire worker permit for shard {}", idx);
                continue;
            }

            let tx = batch_tx.clone();
            let sem = semaphore.clone();
            let bs = batch_size;
            let recovery_config = self.config.corrupted_shard_action;
            let cache_path = self.config.cache_dir.clone();

            task::spawn(async move {
                let _permit = permit.unwrap();
                info!("Worker {}: loading shard {}", idx, shard.path.display());

                // Wrap with corrupted shard recovery
                let mut recovery = CorruptedShardRecovery::new(recovery_config);
                match load_shard_integrated(&shard, bs, cache_path.as_deref()).await {
                    Ok(samples) => {
                        info!("Worker {}: {} samples from {}", idx, samples.len(), shard.path.display());
                        for chunk in samples.chunks(bs) {
                            if tx.send(chunk.to_vec()).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        if recovery.handle_failure(&shard.path, &e.to_string()).is_err() {
                            error!("Worker {}: shard {} failed fatally: {}", idx, shard.path.display(), e);
                        } else {
                            warn!("Worker {}: shard {} skipped: {}", idx, shard.path.display(), e);
                        }
                    }
                }
                drop(sem);
            });
        }

        drop(batch_tx);
    }

    pub async fn next_batch(&mut self) -> Option<Vec<DataSample>> {
        let batch = self.batch_rx.recv().await;
        if let Some(ref samples) = batch {
            let count = samples.len() as u64;
            self.progress.add_samples(count, 1);
            self.current_sample_offset += count;

            // Periodic progress report
            if self.progress.batches_processed % 10 == 0 {
                self.progress.report();
                if let Some(ref rsp) = self.resume_state_path {
                    let state = ResumeState {
                        epoch: self.current_epoch,
                        shard_index: self.current_shard_idx,
                        sample_offset: self.current_sample_offset,
                        optimizer_state: None,
                        best_val_loss: None,
                    };
                    state.save(rsp).ok();
                }
            }

            // Update streaming stats
            let elapsed = self.progress.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.stats.read_speed = self.progress.speed();
                self.stats.memory_mb = (std::mem::size_of::<DataSample>() * self.config.shuffle_buffer) as u64 / (1024 * 1024);
            }
        }
        batch
    }

    pub fn resume_state(&self) -> ResumeState {
        ResumeState {
            epoch: self.current_epoch,
            shard_index: self.current_shard_idx,
            sample_offset: self.current_sample_offset,
            optimizer_state: None,
            best_val_loss: None,
        }
    }

    pub fn save_resume_state(&self) -> Result<(), ResumeError> {
        if let Some(ref rsp) = self.resume_state_path {
            let state = self.resume_state();
            state.save(rsp)
        } else {
            Ok(())
        }
    }

    pub fn speed(&self) -> f64 {
        self.progress.speed()
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.progress.elapsed()
    }

    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    pub fn bottleneck_detected(&self) -> Option<String> {
        self.stats.detect_bottleneck()
    }

    pub fn skipped_shards(&self) -> &[String] {
        self.recovery.skipped_shards()
    }

    pub fn total_failures(&self) -> usize {
        self.recovery.total_failures()
    }

    pub fn manifest(&self) -> Option<&DatasetManifest> {
        self.manifest.as_ref()
    }
}

// --- Integrated shard loading with mmap + cache support ---

async fn load_shard_integrated(
    shard: &ShardPath,
    batch_size: usize,
    cache_dir: Option<&Path>,
) -> Result<Vec<DataSample>, LoaderError> {
    let source = SourceInfo {
        name: shard.path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into()),
        url: None,
        trust_score: 0.8,
        category: SourceCategory::Other,
        fetch_timestamp: chrono::Utc::now().timestamp(),
    };

    // Fast path: mmap for uncompressed arrow files
    if shard.compression == Compression::None {
        #[cfg(feature = "mmap")]
        {
            match super::cache::read_arrow_mmap(&shard.path, source.clone()) {
                Ok(samples) => {
                    debug!("mmap'd {} samples from {}", samples.len(), shard.path.display());
                    return Ok(samples);
                }
                Err(e) => {
                    debug!("mmap failed for {} (falling back): {}", shard.path.display(), e);
                }
            }
        }
        #[cfg(not(feature = "mmap"))]
        {
            // Fall through to standard read
        }
    }

    // Standard path: read + decompress + parse
    let raw = tokio::task::spawn_blocking({
        let path = shard.path.clone();
        move || -> Result<Vec<u8>, LoaderError> {
            std::fs::read(&path).map_err(|e| LoaderError::Io(e.to_string()))
        }
    }).await.map_err(|e| LoaderError::Join(e.to_string()))??;

    let decompressed = shard.compression.decompress(&raw)
        .map_err(|e| LoaderError::Compression(e.to_string()))?;

    // Write to arrow file for parsing
    let tmpdir = tempfile::TempDir::new()
        .map_err(|e| LoaderError::Io(e.to_string()))?;
    let arrow_path = tmpdir.path().join("shard.arrow");
    std::fs::write(&arrow_path, &decompressed)
        .map_err(|e| LoaderError::Io(e.to_string()))?;

    let mut samples = arrow_reader::read_arrow_file(&arrow_path, source)
        .map_err(|e| LoaderError::Arrow(e.to_string()))?;

    // Tokenizer cache: if cache_dir is set, cache token IDs
    if let Some(cache_dir) = cache_dir {
        let token_cache_dir = cache_dir.join("tokens");
        std::fs::create_dir_all(&token_cache_dir).ok();
        let mut token_cache = TokenizerCache::new(token_cache_dir);
        for sample in &mut samples {
            if sample.text.len() < 10000 {
                if let Some(cached) = token_cache.get(&sample.text) {
                    sample.token_ids = Some(cached.clone());
                }
            }
        }
    }

    Ok(samples)
}

fn validate_shard_schema(schema: &DatasetSchema, path: &Path) -> Result<SchemaValidation, LoaderError> {
    #[cfg(feature = "arrow")]
    {
        let file = std::fs::File::open(path)
            .map_err(|e| LoaderError::Io(e.to_string()))?;
        use arrow::ipc::reader::FileReader;
        let reader = FileReader::try_new(file, None)
            .map_err(|e| LoaderError::Arrow(e.to_string()))?;
        Ok(schema.validate_arrow(reader.schema().as_ref()))
    }
    #[cfg(not(feature = "arrow"))]
    {
        // Without arrow feature, skip validation
        Ok(SchemaValidation { valid: true, issues: vec![] })
    }
}

fn is_arrow_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "arrow")
        .unwrap_or(false)
}

fn estimate_samples(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    if size < 100 {
        return Some(0);
    }
    let avg_row_size = 200u64;
    Some(size / avg_row_size)
}

// --- Errors ---

#[derive(Debug)]
pub enum LoaderError {
    NoShards(String),
    Io(String),
    Compression(String),
    Arrow(String),
    Join(String),
    Manifest(String),
    Schema(String),
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::NoShards(p) => write!(f, "No shards found in {}", p),
            LoaderError::Io(msg) => write!(f, "IO error: {}", msg),
            LoaderError::Compression(msg) => write!(f, "Compression error: {}", msg),
            LoaderError::Arrow(msg) => write!(f, "Arrow error: {}", msg),
            LoaderError::Join(msg) => write!(f, "Worker join error: {}", msg),
            LoaderError::Manifest(msg) => write!(f, "Manifest error: {}", msg),
            LoaderError::Schema(msg) => write!(f, "Schema error: {}", msg),
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<ManifestError> for LoaderError {
    fn from(e: ManifestError) -> Self {
        LoaderError::Manifest(e.to_string())
    }
}
