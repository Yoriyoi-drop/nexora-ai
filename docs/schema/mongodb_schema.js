// ==================== Nexora-AI MongoDB Schema ====================
// Layer 3: Document Store for Corpus Metadata and Training Logs
// Run with: mongosh --host localhost --port 27017 -u nexora_admin -p nexora_secure_password --authenticationDatabase admin nexora < mongodb_schema.js

// Switch to nexora database
db = db.getSiblingDB('nexora');

// ==================== Corpus Metadata Collections ====================

// Collection: corpus_documents
// Stores metadata for each document in the training corpus
db.createCollection('corpus_documents');

// Indexes for corpus_documents
db.corpus_documents.createIndex({ "dedup_hash": 1 }, { unique: true });
db.corpus_documents.createIndex({ "source": 1 });
db.corpus_documents.createIndex({ "domain": 1 });
db.corpus_documents.createIndex({ "quality_score": -1 });
db.corpus_documents.createIndex({ "language": 1 });
db.corpus_documents.createIndex({ "license": 1 });
db.corpus_documents.createIndex({ "created_at": -1 });
db.corpus_documents.createIndex({ "source": 1, "domain": 1 });
db.corpus_documents.createIndex({ "quality_score": -1, "domain": 1 });

// Sample document structure for corpus_documents
/*
{
    "_id": ObjectId("..."),
    "dedup_hash": "sha256:abc123...",  // Unique hash for deduplication
    "source": "github",  // github, arxiv, stackoverflow, wikipedia, etc.
    "domain": "code",  // code, text, mixed
    "language": "python",  // python, javascript, english, etc.
    "license": "mit",  // mit, apache2, gpl, cc-by, etc.
    "quality_score": 0.85,  // 0.0 to 1.0
    "perplexity": 2.45,  // Model perplexity on this document
    "token_count": 1234,
    "file_path": "/data/corpus/github/python/abc123.py",
    "file_size_bytes": 5678,
    "created_at": ISODate("2024-01-15T10:30:00Z"),
    "last_modified": ISODate("2024-01-15T10:30:00Z"),
    "metadata": {
        "github": {
            "repo_name": "user/repo",
            "commit_hash": "abc123...",
            "stars": 100,
            "forks": 20
        },
        "arxiv": {
            "paper_id": "2301.00001",
            "title": "...",
            "authors": ["..."],
            "categories": ["cs.AI"]
        },
        "stackoverflow": {
            "question_id": 12345,
            "score": 10,
            "accepted_answer": true
        }
    },
    "processing_status": "completed",  // pending, processing, completed, failed
    "error_message": null
}
*/

// Collection: corpus_statistics
// Aggregated statistics for the corpus
db.createCollection('corpus_statistics');

db.corpus_statistics.createIndex({ "date": -1 });
db.corpus_statistics.createIndex({ "source": 1, "date": -1 });

// Sample document structure for corpus_statistics
/*
{
    "_id": ObjectId("..."),
    "date": ISODate("2024-01-15T00:00:00Z"),
    "source": "github",
    "domain": "code",
    "total_documents": 1000000,
    "total_tokens": 5000000000,
    "avg_quality_score": 0.82,
    "language_distribution": {
        "python": 0.45,
        "javascript": 0.30,
        "java": 0.15,
        "other": 0.10
    },
    "license_distribution": {
        "mit": 0.60,
        "apache2": 0.25,
        "gpl": 0.10,
        "other": 0.05
    }
}
*/

// Collection: deduplication_groups
// Groups of documents that are duplicates or near-duplicates
db.createCollection('deduplication_groups');

db.deduplication_groups.createIndex({ "group_id": 1 }, { unique: true });
db.deduplication_groups.createIndex({ "representative_hash": 1 });

// Sample document structure for deduplication_groups
/*
{
    "_id": ObjectId("..."),
    "group_id": "group_abc123",
    "representative_hash": "sha256:abc123...",
    "similarity_threshold": 0.95,
    "document_hashes": [
        "sha256:abc123...",
        "sha256:def456...",
        "sha256:ghi789..."
    ],
    "selected_document": "sha256:abc123...",  // The one kept for training
    "rejected_documents": [
        "sha256:def456...",
        "sha256:ghi789..."
    ],
    "created_at": ISODate("2024-01-15T10:30:00Z")
}
*/

// ==================== Training Run Collections ====================

// Collection: training_runs
// Metadata for each training run
db.createCollection('training_runs');

db.training_runs.createIndex({ "run_id": 1 }, { unique: true });
db.training_runs.createIndex({ "model_name": 1 });
db.training_runs.createIndex({ "status": 1 });
db.training_runs.createIndex({ "start_time": -1 });
db.training_runs.createIndex({ "model_name", "status" });

// Sample document structure for training_runs
/*
{
    "_id": ObjectId("..."),
    "run_id": "run_20240115_001",
    "model_name": "nexora-7b",
    "model_config": {
        "vocab_size": 64000,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "num_layers": 32,
        "max_seq_len": 8192,
        "use_weight_tying": true
    },
    "training_config": {
        "batch_size": 64,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_steps": 100000,
        "gradient_accumulation_steps": 4,
        "optimizer": "adamw"
    },
    "data_config": {
        "corpus_version": "v1.2",
        "domains": ["code", "text"],
        "languages": ["python", "javascript", "english"],
        "total_tokens": 100000000000
    },
    "status": "running",  // pending, running, paused, completed, failed
    "start_time": ISODate("2024-01-15T10:00:00Z"),
    "end_time": null,
    "current_step": 5000,
    "current_loss": 2.45,
    "checkpoint_dir": "/checkpoints/nexora-7b/run_20240115_001",
    "gpu_config": {
        "num_gpus": 8,
        "gpu_type": "A100-80GB",
        "distributed_backend": "nccl"
    },
    "resume_from_checkpoint": null,
    "notes": "Initial training run for Nexora-7B"
}
*/

// Collection: training_checkpoints
// Metadata for each checkpoint saved during training
db.createCollection('training_checkpoints');

db.training_checkpoints.createIndex({ "checkpoint_id": 1 }, { unique: true });
db.training_checkpoints.createIndex({ "run_id": 1 });
db.training_checkpoints.createIndex({ "step_number": -1 });
db.training_checkpoints.createIndex({ "run_id", "step_number" });

// Sample document structure for training_checkpoints
/*
{
    "_id": ObjectId("..."),
    "checkpoint_id": "nexora-7b_step_5000",
    "run_id": "run_20240115_001",
    "step_number": 5000,
    "training_loss": 2.45,
    "validation_loss": 2.52,
    "learning_rate": 0.0001,
    "file_path": "/checkpoints/nexora-7b/run_20240115_001/step_5000.bin",
    "file_size_bytes": 14000000000,
    "created_at": ISODate("2024-01-15T12:00:00Z"),
    "is_deployed": false,
    "eval_results": {
        "perplexity": 2.52,
        "accuracy": 0.65,
        "f1_score": 0.62
    },
    "metadata": {
        "gpu_memory_used": 64000000000,
        "throughput_tokens_per_sec": 50000,
        "wall_time_hours": 10.5
    }
}
*/

// Collection: training_metrics
// Detailed metrics logged during training
db.createCollection('training_metrics');

db.training_metrics.createIndex({ "run_id": 1, "step_number": 1 });
db.training_metrics.createIndex({ "run_id", "timestamp": -1 });
db.training_metrics.createIndex({ "timestamp": -1 });

// Sample document structure for training_metrics
/*
{
    "_id": ObjectId("..."),
    "run_id": "run_20240115_001",
    "step_number": 5000,
    "timestamp": ISODate("2024-01-15T12:00:00Z"),
    "loss": 2.45,
    "learning_rate": 0.0001,
    "gradient_norm": 1.23,
    "throughput_tokens_per_sec": 50000,
    "gpu_utilization": 0.95,
    "memory_used_gb": 64,
    "data_loading_time_ms": 50,
    "forward_time_ms": 200,
    "backward_time_ms": 300,
    "optimizer_time_ms": 50
}
*/

// Collection: evaluation_results
// Results from evaluating checkpoints on benchmarks
db.createCollection('evaluation_results');

db.evaluation_results.createIndex({ "checkpoint_id": 1 });
db.evaluation_results.createIndex({ "benchmark_name": 1 });
db.evaluation_results.createIndex({ "checkpoint_id", "benchmark_name" }, { unique: true });
db.evaluation_results.createIndex({ "timestamp": -1 });

// Sample document structure for evaluation_results
/*
{
    "_id": ObjectId("..."),
    "checkpoint_id": "nexora-7b_step_5000",
    "benchmark_name": "human_eval",
    "dataset": "human_eval_python",
    "timestamp": ISODate("2024-01-15T13:00:00Z"),
    "metrics": {
        "pass@1": 0.45,
        "pass@10": 0.65,
        "pass@100": 0.75
    },
    "additional_metrics": {
        "avg_tokens_per_sample": 150,
        "avg_latency_ms": 500
    },
    "config": {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 500
    },
    "notes": "Evaluated on HumanEval Python benchmark"
}
*/

// ==================== Conversation Logs Collections ====================

// Collection: conversation_logs
// Anonymized conversation logs for analysis
db.createCollection('conversation_logs');

db.conversation_logs.createIndex({ "session_id": 1 });
db.conversation_logs.createIndex({ "timestamp": -1 });
db.conversation_logs.createIndex({ "model_name": 1 });
db.conversation_logs.createIndex({ "timestamp": -1, "model_name" });

// Sample document structure for conversation_logs
/*
{
    "_id": ObjectId("..."),
    "session_id": "session_abc123",
    "timestamp": ISODate("2024-01-15T14:00:00Z"),
    "model_name": "nexora-7b",
    "model_version": "step_5000",
    "messages": [
        {
            "role": "user",
            "content": "How do I sort a list in Python?",
            "tokens": 8,
            "timestamp": ISODate("2024-01-15T14:00:00Z")
        },
        {
            "role": "assistant",
            "content": "You can use the sort() method...",
            "tokens": 25,
            "latency_ms": 500,
            "timestamp": ISODate("2024-01-15T14:00:01Z")
        }
    ],
    "total_tokens": 33,
    "total_latency_ms": 500,
    "user_anonymized_id": "user_hash_abc123",
    "metadata": {
        "language_detected": "python",
        "category": "programming",
        "rating": null  // User rating if provided
    }
}
*/

// Collection: conversation_analytics
// Aggregated analytics from conversation logs
db.createCollection('conversation_analytics');

db.conversation_analytics.createIndex({ "date": -1 });
db.conversation_analytics.createIndex({ "model_name": 1 });
db.conversation_analytics.createIndex({ "date": -1, "model_name" });

// Sample document structure for conversation_analytics
/*
{
    "_id": ObjectId("..."),
    "date": ISODate("2024-01-15T00:00:00Z"),
    "model_name": "nexora-7b",
    "total_conversations": 10000,
    "total_messages": 50000,
    "total_tokens": 2500000,
    "avg_tokens_per_conversation": 250,
    "avg_latency_ms": 450,
    "p50_latency_ms": 400,
    "p95_latency_ms": 800,
    "p99_latency_ms": 1500,
    "error_rate": 0.01,
    "category_distribution": {
        "programming": 0.60,
        "general": 0.30,
        "creative": 0.10
    },
    "language_distribution": {
        "python": 0.40,
        "javascript": 0.25,
        "english": 0.35
    }
}
*/

// ==================== Data Pipeline Collections ====================

// Collection: pipeline_jobs
// Metadata for data pipeline jobs
db.createCollection('pipeline_jobs');

db.pipeline_jobs.createIndex({ "job_id": 1 }, { unique: true });
db.pipeline_jobs.createIndex({ "status": 1 });
db.pipeline_jobs.createIndex({ "job_type": 1 });
db.pipeline_jobs.createIndex({ "start_time": -1 });

// Sample document structure for pipeline_jobs
/*
{
    "_id": ObjectId("..."),
    "job_id": "job_20240115_001",
    "job_type": "corpus_collection",  // corpus_collection, deduplication, tokenization, quality_filtering
    "status": "completed",  // pending, running, completed, failed
    "start_time": ISODate("2024-01-15T08:00:00Z"),
    "end_time": ISODate("2024-01-15T10:00:00Z"),
    "config": {
        "source": "github",
        "query": "language:python stars:>100",
        "max_repos": 1000
    },
    "results": {
        "documents_collected": 50000,
        "documents_filtered": 45000,
        "total_tokens": 250000000
    },
    "error_message": null,
    "worker_id": "worker_001"
}
*/

// Collection: synthetic_data
// Metadata for synthetically generated data
db.createCollection('synthetic_data');

db.synthetic_data.createIndex({ "generation_id": 1 }, { unique: true });
db.synthetic_data.createIndex({ "model_checkpoint": 1 });
db.synthetic_data.createIndex({ "created_at": -1 });

// Sample document structure for synthetic_data
/*
{
    "_id": ObjectId("..."),
    "generation_id": "gen_20240115_001",
    "model_checkpoint": "nexora-7b_step_5000",
    "generation_method": "self_instruct",  // self_instruct, backtranslation, etc.
    "prompt_template": "Write a Python function to...",
    "num_samples": 10000,
    "created_at": ISODate("2024-01-15T15:00:00Z"),
    "quality_score": 0.78,
    "output_path": "/data/synthetic/gen_20240115_001.jsonl",
    "metadata": {
        "temperature": 0.8,
        "top_p": 0.95,
        "filter_criteria": {
            "min_length": 50,
            "max_length": 500
        }
    }
}
*/

// ==================== System Collections ====================

// Collection: system_events
// System events and alerts
db.createCollection('system_events');

db.system_events.createIndex({ "timestamp": -1 });
db.system_events.createIndex({ "event_type": 1 });
db.system_events.createIndex({ "severity": 1 });

// Sample document structure for system_events
/*
{
    "_id": ObjectId("..."),
    "timestamp": ISODate("2024-01-15T16:00:00Z"),
    "event_type": "training_crash",  // training_crash, gpu_failure, disk_full, etc.
    "severity": "critical",  // info, warning, error, critical
    "component": "training_worker_001",
    "message": "Training crashed due to OOM",
    "details": {
        "run_id": "run_20240115_001",
        "step_number": 5000,
        "error_code": "CUDA_OUT_OF_MEMORY"
    },
    "resolved": false,
    "resolved_at": null
}
*/

// Print completion message
print("MongoDB schema created successfully for Nexora-AI");
print("Collections created:");
print("  - corpus_documents");
print("  - corpus_statistics");
print("  - deduplication_groups");
print("  - training_runs");
print("  - training_checkpoints");
print("  - training_metrics");
print("  - evaluation_results");
print("  - conversation_logs");
print("  - conversation_analytics");
print("  - pipeline_jobs");
print("  - synthetic_data");
print("  - system_events");
