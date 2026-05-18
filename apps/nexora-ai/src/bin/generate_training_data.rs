use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{StringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;

const TARGET_SHARD_BYTES: u64 = 200 * 1024 * 1024;
const ROWS_PER_BATCH: usize = 10_000;
const SAMPLE_TEXTS: &[&str] = &[
    "Nexora AI is a next-generation artificial intelligence system designed for multi-modal reasoning and adaptive learning across diverse domains.",
    "The NXR-OMNIS model architecture implements a unified reasoning framework that integrates symbolic logic with neural computation for robust decision-making.",
    "Self-aware cognitive architecture (SACA) enables meta-cognition, allowing the system to reflect on its own reasoning processes and identify potential errors.",
    "Multi-layer isolation architecture provides security at levels L0 through L6, creating robust boundaries between different operational domains.",
    "The CAFFEINE multimodal framework integrates text, image, audio, and video understanding through a shared latent representation space.",
    "Knowledge synthesis combines information from multiple sources using graph-based reasoning to generate novel insights beyond any single source.",
    "Deep learning enables powerful pattern recognition and generation capabilities across scales from individual characters to complex document structures.",
    "The STar-X tensor framework accelerates neural network operations through optimized matrix computations and automatic parallelization.",
    "Automatic differentiation enables efficient gradient computation for training deep neural networks with millions of parameters.",
    "Gradient accumulation improves training stability by simulating larger batch sizes through multiple forward-backward passes before each optimizer step.",
    "Learning rate warmup followed by cosine decay optimizes convergence by gradually increasing then smoothly decreasing the step size during training.",
    "Weight decay regularization prevents overfitting in large models by penalizing excessive weight magnitudes through L2 regularization techniques.",
    "Gradient clipping ensures stable training by limiting gradient norms to prevent exploding gradients in deep neural networks.",
    "The DAG-based data pipeline streams and filters training data efficiently using a directed acyclic graph of transformation operations.",
    "Cross-domain integration allows transfer learning between domains, enabling models to apply knowledge gained in one domain to solve problems in another.",
    "Neural architecture search automates the discovery of optimal network topologies for specific tasks and datasets.",
    "Attention mechanisms enable models to focus on relevant parts of input sequences, improving performance on long-range dependency tasks.",
    "Transformer architectures have revolutionized natural language processing through their parallelizable self-attention mechanisms.",
    "Reinforcement learning from human feedback aligns model outputs with human preferences through iterative reward modeling.",
    "Few-shot learning enables models to generalize from limited examples by leveraging prior knowledge acquired during pre-training.",
    "Mixture of experts architectures scale model capacity efficiently by routing inputs to specialized subnetworks.",
    "Contrastive learning learns robust representations by bringing similar examples together and pushing dissimilar ones apart in embedding space.",
    "Knowledge distillation transfers capabilities from large teacher models to smaller student models for efficient deployment.",
    "Quantum machine learning explores the intersection of quantum computing and artificial intelligence for potentially exponential speedups.",
    "Federated learning enables collaborative model training across decentralized data sources while preserving data privacy.",
    "The Oracle alignment system ensures AI behaviors remain consistent with human values through continuous monitoring and adjustment.",
    "EchoNet provides recursive feedback loops that allow the system to refine its outputs through multiple passes of self-correction.",
    "The VOGP framework enables principled uncertainty estimation in neural network predictions through variational inference.",
    "HLDA-VT implements hierarchical latent diffusion with variational transformers for high-quality generative modeling across modalities.",
    "SPARO enables sparse autoregressive modeling that reduces computational overhead while maintaining output quality.",
    "The ATQS adaptive training system dynamically adjusts learning parameters based on real-time performance metrics.",
    "Tensor program optimization automatically identifies and applies the most efficient computation order for neural network operations.",
    "Memory-augmented neural networks combine external memory with differentiable read and write operations for long-term information storage.",
    "Graph neural networks extend deep learning to non-Euclidean domains, enabling reasoning over relational data structures.",
    "Capsule networks preserve hierarchical spatial relationships between features, improving robustness to viewpoint changes.",
    "Normalizing flows provide a framework for learning complex probability distributions through invertible transformations.",
    "Variational autoencoders learn latent representations by maximizing a variational lower bound on the data likelihood.",
    "Generative adversarial networks pit two networks against each other in a minimax game to generate realistic synthetic data.",
    "Diffusion models generate data by progressively denoising random noise through a learned reverse diffusion process.",
    "Energy-based models assign unnormalized probabilities to configurations, enabling flexible density estimation.",
    "The Echo State Network paradigm uses fixed random recurrent connections with trained readout layers for efficient temporal processing.",
    "Bayesian neural networks place distributions over weights to capture epistemic uncertainty in predictions.",
    "Neural ordinary differential equations parameterize continuous-depth transformations for modeling dynamical systems.",
    "Self-supervised learning leverages inherent data structure to create supervisory signals without manual annotation.",
    "Multi-task learning improves generalization by jointly training on related tasks that share beneficial representations.",
    "Curriculum learning organizes training data from simple to complex examples, mirroring human learning trajectories.",
    "Adversarial training improves model robustness by exposing networks to carefully crafted input perturbations during training.",
    "Meta-learning algorithms learn to learn, acquiring the ability to rapidly adapt to new tasks with minimal data.",
    "Neural tangent kernel theory provides a unified framework for understanding gradient-based learning in overparameterized networks.",
    "The information bottleneck principle guides representation learning by balancing compression against predictive accuracy.",
    "Causal inference in machine learning moves beyond correlation to model intervention and counterfactual reasoning.",
    "The Nexora platform provides a comprehensive ecosystem for developing, training, and deploying AI systems at scale.",
    "Distributed training across multiple GPUs enables scaling to models with hundreds of billions of parameters.",
    "Mixed precision training uses lower-precision arithmetic for most operations while maintaining accuracy through strategic full-precision computations.",
    "Gradient checkpointing trades computation for memory by selectively recomputing intermediate activations during backpropagation.",
    "Model parallelism splits large models across multiple devices by partitioning layers or attention heads.",
    "Pipeline parallelism divides the model into stages that execute on different devices, with micro-batches flowing through the pipeline.",
    "Tensor parallelism distributes individual operations across devices using optimized collective communication primitives.",
    "ZeRO optimization eliminates redundant memory consumption by partitioning optimizer states, gradients, and parameters across devices.",
    "Flash attention accelerates transformer training by computing attention without materializing the full attention matrix.",
    "Sparse attention patterns reduce computational complexity from quadratic to linear in sequence length.",
    "Sliding window attention maintains a fixed-size context window for efficient processing of long sequences.",
    "The Nexora data pipeline supports real-time streaming ingestion from multiple sources with automatic schema evolution.",
    "Data quality filters automatically detect and remove anomalous samples, ensuring training data integrity.",
    "Deduplication at scale identifies both exact and near-duplicate examples using locality-sensitive hashing techniques.",
    "Toxicity detection filters remove harmful content using multi-label classification with continuous monitoring.",
    "Source trust scoring tracks provenance and reliability of data across the entire pipeline from ingestion to training.",
    "Automated data augmentation generates diverse training examples through composable transformation pipelines.",
    "Active learning selects the most informative examples for annotation, reducing labeling costs while maximizing model improvement.",
    "Weak supervision combines multiple noisy labeling sources to create training data at scale without manual annotation.",
    "Data versioning tracks the lineage and transformations applied to each training dataset for reproducibility.",
    "Differential privacy provides formal guarantees against membership inference attacks in trained models.",
    "Model watermarking embeds imperceptible signatures in model outputs for intellectual property protection.",
    "Federated learning enables training across institutional boundaries without centralizing sensitive data.",
    "Secure multi-party computation allows multiple parties to jointly compute functions over their private inputs.",
    "Homomorphic encryption enables computation on encrypted data, allowing model inference without revealing inputs.",
    "The Nexora inference engine optimizes serving through dynamic batching, KV caching, and speculative decoding.",
    "Continuous batching interleaves requests at the token level for maximum throughput in online serving.",
    "Speculative decoding uses a draft model to predict multiple tokens, verified in parallel by the target model.",
    "KV cache quantization reduces memory footprint of key-value caches through compression techniques.",
    "Prefix caching reuses computation across requests with shared prompt prefixes for improved latency.",
    "Beam search explores multiple candidate sequences in parallel during generation for higher quality outputs.",
    "Top-k sampling restricts token selection to the k most likely candidates for controlled randomness in generation.",
    "Temperature scaling adjusts the sharpness of the probability distribution for fine-grained control of creativity vs determinism.",
    "Repetition penalty discourages the model from generating repeated tokens by penalizing already-seen tokens.",
    "Logit bias allows fine-grained control over token probabilities for customizing output characteristics.",
    "The monitoring subsystem provides real-time visibility into model performance, resource usage, and system health.",
    "Distributed tracing tracks requests across service boundaries for end-to-end latency analysis and debugging.",
    "Prometheus metrics expose system internals for integration with standard monitoring infrastructure.",
    "Structured logging captures events with rich context for efficient querying and analysis of system behavior.",
    "Health check endpoints provide comprehensive status information for load balancers and orchestration systems.",
    "Rate limiting protects system resources by controlling the flow of incoming requests based on configurable policies.",
    "Circuit breakers prevent cascading failures by automatically stopping requests to degraded dependencies.",
    "Graceful degradation ensures core functionality remains available even when non-critical components fail.",
    "The Nexora CLI provides intuitive interfaces for training, evaluation, inference, and system management tasks.",
    "Training progress visualization displays loss curves, learning rate schedules, and evaluation metrics in real time.",
    "Model export supports multiple formats including SafeTensors for production deployment across platforms.",
    "Automatic checkpointing saves training state at configurable intervals with support for resuming interrupted runs.",
    "Hyperparameter optimization searches over configuration spaces using Bayesian optimization and early stopping.",
    "The TUI dashboard provides a terminal-based interface for monitoring system status and managing operations.",
    "Web-based dashboards offer rich visualizations of system metrics with customizable alerting rules.",
    "API-first design enables programmatic access to all system capabilities through REST and WebSocket endpoints.",
    "The plugin system allows extending functionality through dynamically loaded modules with versioned interfaces.",
    "Configuration management supports hot-reloading for updating system parameters without service interruption.",
    "Role-based access control provides fine-grained authorization for multi-tenant deployments.",
    "Audit logging captures all security-relevant events with tamper-evident storage for compliance requirements.",
    "The Nexora security model implements defense in depth with multiple layers of protection at each system boundary.",
    "Input validation sanitizes all external data using allowlist-based filtering to prevent injection attacks.",
    "Output encoding ensures generated content is safe for its intended context, preventing XSS and similar vulnerabilities.",
    "Rate limiting at multiple levels protects against both accidental overload and malicious denial-of-service attacks.",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("training_data")
    };

    let num_shards: usize = if args.len() > 2 {
        args[2].parse().expect("num_shards must be a number")
    } else {
        4
    };

    let with_splits = args.iter().any(|a| a == "--splits" || a == "--split");
    let small = args.iter().any(|a| a == "--small" || a == "--dev");

    std::fs::create_dir_all(&output_dir).expect("failed to create output dir");

    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
    ]));

    let avg_sample_bytes = SAMPLE_TEXTS.iter().map(|t| t.len() as u64).sum::<u64>() / SAMPLE_TEXTS.len() as u64;
    let rows_per_shard = if small {
        1000
    } else {
        (TARGET_SHARD_BYTES / avg_sample_bytes) as usize
    };

    let splits: Vec<(&str, f64)> = if with_splits {
        vec![("train", 0.8), ("val", 0.1), ("test", 0.1)]
    } else {
        vec![("train", 1.0)]
    };

    let total_shards: usize = splits.iter()
        .map(|&(_, ratio)| if with_splits { (num_shards as f64 * ratio).ceil() as usize } else { num_shards })
        .sum();
    let mut shard_infos: Vec<serde_json::Value> = Vec::with_capacity(total_shards);
    let mut total_rows_all: u64 = 0;

    for &(split_name, ratio) in &splits {
        let split_shards = if with_splits {
            (num_shards as f64 * ratio).ceil() as usize
        } else {
            num_shards
        };

        let split_dir = output_dir.join(split_name);
        std::fs::create_dir_all(&split_dir).expect("failed to create split dir");

        let split_rows = (rows_per_shard as f64 * ratio) as usize;
        let actual_rows_per_shard = split_rows.max(100);

        for shard_idx in 0..split_shards {
            let shard_path = split_dir.join(format!("shard-{:04}.arrow", shard_idx));
            let file = std::fs::File::create(&shard_path).expect("failed to create shard file");
            let mut writer = FileWriter::try_new(file, &schema).expect("failed to create arrow writer");

            let mut rows_written = 0;
            while rows_written < actual_rows_per_shard {
                let batch_size = ROWS_PER_BATCH.min(actual_rows_per_shard - rows_written);
                let mut texts = Vec::with_capacity(batch_size);

                for i in 0..batch_size {
                    let idx = (shard_idx * actual_rows_per_shard + rows_written + i) % SAMPLE_TEXTS.len();
                    let base = SAMPLE_TEXTS[idx];
                    let text = format!("{} [split={}, shard={:04}, row={}]", base, split_name, shard_idx, rows_written + i);
                    texts.push(text);
                }

                let text_array = StringArray::from(texts);
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(text_array)],
                ).expect("failed to create record batch");

                writer.write(&batch).expect("failed to write batch");
                rows_written += batch_size;

                if rows_written % (ROWS_PER_BATCH * 10) == 0 || rows_written >= actual_rows_per_shard {
                    let pct = rows_written as f64 / actual_rows_per_shard as f64 * 100.0;
                    eprintln!("  {} shard {:04}: {}/{} rows ({:.0}%)", split_name, shard_idx, rows_written, actual_rows_per_shard, pct);
                }
            }

            writer.finish().expect("failed to finalize arrow file");
            let file_size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
            eprintln!("  -> {} written ({:.2} MB)", shard_path.display(), file_size as f64 / 1024.0 / 1024.0);

            total_rows_all += rows_written as u64;

            let rel_path = if with_splits {
                format!("{}/shard-{:04}.arrow", split_name, shard_idx)
            } else {
                format!("shard-{:04}.arrow", shard_idx)
            };

            shard_infos.push(serde_json::json!({
                "path": rel_path,
                "split": split_name,
                "samples": rows_written,
                "size_bytes": file_size,
                "compression": null,
                "checksum": null,
            }));
        }
    }

    // Generate manifest.json
    let manifest = serde_json::json!({
        "name": "OpenTextDataset",
        "version": "1.0",
        "format": "arrow",
        "compression": null,
        "total_samples": total_rows_all,
        "total_shards": shard_infos.len(),
        "shards": shard_infos,
        "features": ["text"],
        "schema": {
            "text": "utf8"
        },
        "created_at": chrono::Utc::now().to_rfc3339(),
        "checksum": null,
    });

    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).expect("serialize manifest"))
        .expect("failed to write manifest.json");
    eprintln!("\n  -> {} written", manifest_path.display());
    eprintln!("\nDone! Dataset directory: {} ({} samples, {} shards)", output_dir.display(), total_rows_all, shard_infos.len());
    println!("{}", output_dir.display());
}
