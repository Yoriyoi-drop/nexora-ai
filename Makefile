.PHONY: all build check clippy fmt test test-fast clean \
        run dev debug \
        train train-resume evaluate \
        health info perf mem \
        chat generate process analyze \
        docker-up docker-down logs \
        help

BIN           ?= nexora
CARGO_FLAGS   ?=

# --- Server ---
HOST          ?= 127.0.0.1
PORT          ?= 8080

# --- Training ---
DATA          ?= ./training_data
OUTPUT        ?= ./checkpoints/model
TOKENIZER     ?= ./tokenizer.json
EPOCHS        ?= 10
BATCH         ?= 32
LR            ?= 0.001

# --- Evaluate ---
MODEL         ?= ./checkpoints/model.safetensors
TEST_DATA     ?= ./test_data

# --- Misc ---
PROMPT        ?= "Hello"
INPUT         ?= "test input"
LANG          ?= rust
FILE          ?= src/main.rs

# ============================================================================
# BUILD & QUALITY
# ============================================================================

all: check build

build:
	cargo build --release $(CARGO_FLAGS)

debug:
	cargo build $(CARGO_FLAGS)

check:
	cargo check $(CARGO_FLAGS)

clippy:
	cargo clippy $(CARGO_FLAGS)

fmt:
	cargo fmt

test:
	cargo nextest run $(CARGO_FLAGS)

test-fast:
	cargo test $(CARGO_FLAGS)

clean:
	cargo clean

# ============================================================================
# SERVER
# ============================================================================

dev:
	cargo run --bin $(BIN) -- start --host $(HOST) --port $(PORT)

run:
	cargo run --bin $(BIN) $(CMD)

# ============================================================================
# TRAINING
# ============================================================================

train:
	cargo run --bin $(BIN) -- train \
		--data $(DATA) --output $(OUTPUT) \
		--epochs $(EPOCHS) --batch-size $(BATCH) --learning-rate $(LR) \
		-g $(TRAIN_FLAGS)

train-resume:
	cargo run --bin $(BIN) -- train \
		--data $(DATA) --output $(OUTPUT) \
		--epochs $(EPOCHS) --batch-size $(BATCH) --learning-rate $(LR) \
		-g -R $(TRAIN_FLAGS)

evaluate:
	cargo run --bin $(BIN) -- evaluate \
		--model $(MODEL) --test-data $(TEST_DATA) --tokenizer $(TOKENIZER) $(EVAL_FLAGS)

# ============================================================================
# MONITORING
# ============================================================================

health:
	cargo run --bin $(BIN) -- health

health-detailed:
	cargo run --bin $(BIN) -- health --detailed

info:
	cargo run --bin $(BIN) -- info

perf:
	cargo run --bin $(BIN) -- info --performance

mem:
	cargo run --bin $(BIN) -- info --memory

# ============================================================================
# INFERENCE
# ============================================================================

chat:
	cargo run --bin $(BIN) -- chat --interactive $(CHAT_FLAGS)

generate:
	cargo run --bin $(BIN) -- generate --prompt $(PROMPT) $(GEN_FLAGS)

process:
	cargo run --bin $(BIN) -- process --input $(INPUT) $(PROC_FLAGS)

analyze:
	cargo run --bin $(BIN) -- analyze --file $(FILE) --language $(LANG) $(ANALYZE_FLAGS)

# ============================================================================
# INFRASTRUCTURE
# ============================================================================

docker-up:
	docker compose up -d $(SERVICES)

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f $(SERVICES)

# ============================================================================
# HELP
# ============================================================================

help:
	@echo '╔══════════════════════════════════════════════════╗'
	@echo '║              Nexora AI — Makefile               ║'
	@echo '╚══════════════════════════════════════════════════╝'
	@echo ''
	@echo ' BUILD'
	@echo '   make                     check + build'
	@echo '   make build               cargo build --release'
	@echo '   make debug               cargo build (debug)'
	@echo '   make check               cargo check'
	@echo '   make clippy              cargo clippy'
	@echo '   make fmt                 cargo fmt'
	@echo '   make test                cargo nextest run'
	@echo '   make clean               cargo clean'
	@echo ''
	@echo ' SERVER'
	@echo '   make dev                 nexora start (HOST=127.0.0.1 PORT=8080)'
	@echo ''
	@echo ' TRAINING'
	@echo '   make train               nexora train (DATA=./training_data)'
	@echo '   make train-resume        nexora train --resume'
	@echo '   make evaluate            nexora evaluate'
	@echo ''
	@echo ' MONITORING'
	@echo '   make health              nexora health'
	@echo '   make info                nexora info'
	@echo '   make perf                nexora info --performance'
	@echo '   make mem                 nexora info --memory'
	@echo ''
	@echo ' INFERENCE'
	@echo '   make chat                interactive chat session'
	@echo '   make generate            nexora generate --prompt=...'
	@echo '   make process             nexora process --input=...'
	@echo '   make analyze             nexora analyze --file=... --language=...'
	@echo ''
	@echo ' INFRASTRUCTURE'
	@echo '   make docker-up           docker compose up -d'
	@echo '   make docker-down         docker compose down'
	@echo '   make docker-logs         docker compose logs -f'
	@echo ''
	@echo ' VARIABEL (semua opsional, punya default)'
	@echo '   DATA=./training_data     OUTPUT=./checkpoints/model'
	@echo '   EPOCHS=10                BATCH=32               LR=0.001'
	@echo '   HOST=127.0.0.1           PORT=8080'
	@echo '   PROMPT="Hello"           INPUT="test"'
	@echo '   FILE=src/main.rs         LANG=rust'
	@echo '   CARGO_FLAGS=             flags tambahan untuk cargo'
	@echo '   TRAIN_FLAGS=             flags tambahan untuk train'
	@echo '   SERVICES=                service filter untuk docker'
	@echo ''
	@echo ' CONTOH'
	@echo '   make train DATA=./my_data EPOCHS=3 BATCH=64 LR=1e-4'
	@echo '   make train-resume TRAIN_FLAGS=--tokenizer ./tok.json'
	@echo '   make dev PORT=3000'
	@echo '   make docker-up SERVICES="postgres redis"'
	@echo '   make chat CHAT_FLAGS="--message hello"'
	@echo '   CARGO_FLAGS="--features foo" make build'
