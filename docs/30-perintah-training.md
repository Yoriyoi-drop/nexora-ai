# 31 Perintah Cargo Training — Nexora AI

## `train` (13 varian)

```sh
# 1  Training dari file .arrow lokal
cargo run --bin nexora -- train --data data.arrow --output ./ckpt

# 2  Training dari HuggingFace live dataset
cargo run --bin nexora -- train --hf-dataset wikitext --output ./ckpt

# 3  Training model spesifik (omnis)
cargo run --bin nexora -- train --data data.txt --output ./ckpt --model-id omnis

# 4  Training semua 10 model sekuensial
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --model-id all

# 5  Training semua 10 model paralel
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --model-id all --parallel

# 6  Custom hyperparameter
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --epochs 3 --batch-size 8 --learning-rate 0.001

# 7  HuggingFace dengan split & max samples
cargo run --bin nexora -- train --hf-dataset wikitext --hf-split train --hf-max-samples 5000 --output ./ckpt

# 8  Custom sequence length (context window)
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --seq-length 256

# 9  Dengan tokenizer custom
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --tokenizer ./tok.json

# 10 Resume dari checkpoint terakhir
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --resume

# 11 Half-precision f16 (2x VRAM savings)
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --half-precision

# 12 Multi-GPU data parallel
cargo run --bin nexora -- train --data data.arrow --output ./ckpt --num-replicas 4

# ⭐ 13 Super lengkap — semua flag
cargo run --bin nexora -- train \
  --hf-dataset wikitext \
  --hf-split train \
  --hf-max-samples 10000 \
  --output ./ckpt/super \
  --tokenizer ./tokenizer.json \
  --model-id all \
  --parallel \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 0.0005 \
  --seq-length 512 \
  --half-precision \
  --num-replicas 2 \
  --resume
```

## `train-foundation` (10 varian)

```sh
# 14 Foundation training model swift
cargo run --bin nexora -- train-foundation --data data.arrow --model-id swift --steps 500 --output ./ckpt/swift

# 14 Foundation training semua model sekuensial
cargo run --bin nexora -- train-foundation --data data.arrow --model-id all --steps 1000 --output ./ckpt/all

# 15 Foundation training semua model paralel
cargo run --bin nexora -- train-foundation --data data.arrow --model-id all --steps 1000 --output ./ckpt/all --parallel

# 16 Foundation training dari HuggingFace
cargo run --bin nexora -- train-foundation --hf-dataset dair-ai/emotion --model-id omnis --steps 200

# 17 Custom batch size
cargo run --bin nexora -- train-foundation --data data.arrow --model-id swift --steps 500 --batch-size 8

# 18 Custom learning rate
cargo run --bin nexora -- train-foundation --data data.arrow --model-id swift --steps 500 --learning-rate 0.005

# 19 Custom sequence length
cargo run --bin nexora -- train-foundation --data data.arrow --model-id swift --steps 500 --seq-length 128

# 20 Dengan validation data
cargo run --bin nexora -- train-foundation --data data.arrow --model-id swift --steps 500 --val-data ./val.arrow

# ⭐ 21 Super lengkap — semua flag
cargo run --bin nexora -- train-foundation \
  --hf-dataset dair-ai/emotion \
  --hf-split train \
  --hf-max-samples 5000 \
  --model-id all \
  --parallel \
  --steps 1000 \
  --batch-size 8 \
  --learning-rate 0.005 \
  --seq-length 128 \
  --val-data ./val.arrow \
  --output ./ckpt/foundation-super \
  --half-precision \
  --num-replicas 2
```

## `collect-data` (3 varian)

```sh
# 22 Koleksi dataset dari sumber default (hackernews,wikipedia,reddit)
cargo run --bin nexora -- collect-data --output ./data/dataset.arrow

# 23 Custom sumber & jumlah sample
cargo run --bin nexora -- collect-data --sources hackernews,wikipedia --max-samples 500 --output ./data.arrow

# 24 Dengan shard size kecil
cargo run --bin nexora -- collect-data --sources hackernews --max-samples 100 --max-shard-size-mb 50 --output ./data.arrow
```

## `load-checkpoint` (3 varian)

```sh
# 25 Load checkpoint ke model omnis (GPU)
cargo run --bin nexora -- load-checkpoint --model omnis --path ./ckpt.final.safetensors

# 26 Load checkpoint ke CPU
cargo run --bin nexora -- load-checkpoint --model omnis --path ./ckpt.final.safetensors --gpu false

# 27 Load best checkpoint ke model vortex
cargo run --bin nexora -- load-checkpoint --model vortex --path ./ckpt.best.safetensors
```

## `evaluate` (2 varian)

```sh
# 28 Evaluasi model (loss + perplexity)
cargo run --bin nexora -- evaluate --model ./model.safetensors --test-data ./test.arrow --tokenizer ./tok.json

# 29 Evaluasi dengan output file
cargo run --bin nexora -- evaluate --model ./model.safetensors --test-data ./test.arrow --tokenizer ./tok.json --output ./results.json
```

## `tokenizer train` (1 varian)

```sh
# 30 Train BPE tokenizer
cargo run --bin nexora -- tokenizer train --data ./corpus.txt --output ./tokenizer.json
```

## `baseline` (1 varian)

```sh
# 31 Benchmark suite termasuk training benchmark
cargo run --bin nexora -- baseline --model omnis --warmup 5 --samples 10 --train-steps 100
```
