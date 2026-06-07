#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# Nexora AI — Production Setup
# Auto-detects environment (H200 vs dev laptop) and configures accordingly.
# Usage: bash setup.sh [--release] [--cuda] [--config-only]
#=============================================================================

BOLD='\033[1m'; RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${CYAN}→${NC} $1"; }
header() { echo -e "\n${BOLD}═══ $1 ═══${NC}\n"; }

RELEASE=false
CUDA=false
CONFIG_ONLY=false
for arg in "$@"; do
  case "$arg" in --release) RELEASE=true ;; --cuda) CUDA=true ;; --config-only) CONFIG_ONLY=true ;; esac
done

#=============================================================================
# 1. Environment detection
#=============================================================================
header "Environment Detection"

HOSTNAME=$(hostname)
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
TOTAL_MEM_GB=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "unknown")
TOTAL_MEM_MB=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "unknown")

echo "  Hostname:    $HOSTNAME"
echo "  CPU cores:   $CPU_CORES"
echo "  RAM:         ${TOTAL_MEM_GB}GB"
echo "  Disk free:   $(df -h . | awk 'NR==2 {print $4}')"

HAS_NVIDIA=false
GPU_NAME=""
CUDA_VERSION=""
DRIVER_VERSION=""
GPU_MEM_MB=0

if command -v nvidia-smi &>/dev/null; then
  if nvidia-smi &>/dev/null 2>&1; then
    HAS_NVIDIA=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | awk '{print $1}')
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1 || nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' || echo "")
    pass "NVIDIA GPU detected: $GPU_NAME (${GPU_MEM_MB}MB, driver $DRIVER_VERSION, CUDA $CUDA_VERSION)"
  else
    warn "nvidia-smi found but no GPU accessible"
  fi
else
  warn "nvidia-smi not found — no NVIDIA GPU"
fi

HAS_NVCC=false
if command -v nvcc &>/dev/null; then
  HAS_NVCC=true
  NVCC_VERSION=$(nvcc --version | grep "release" | grep -oP 'release \K[0-9.]+')
  pass "CUDA toolkit found: nvcc $NVCC_VERSION"
else
  warn "nvcc not found — CUDA toolkit tidak terinstall"
  if $CUDA; then
    fail "--cuda flag diberikan tapi nvcc tidak ditemukan"
    exit 1
  fi
fi

RUST_VERSION=$(rustc --version 2>/dev/null || echo "NOT INSTALLED")
CARGO_VERSION=$(cargo --version 2>/dev/null || echo "NOT INSTALLED")
if [ "$RUST_VERSION" != "NOT INSTALLED" ]; then
  pass "Rust: $RUST_VERSION"
  pass "Cargo: $CARGO_VERSION"
else
  fail "Rust toolchain tidak ditemukan. Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  exit 1
fi

# Determine environment type
IS_H200=false
if $HAS_NVIDIA && [ "$GPU_MEM_MB" -ge 80000 ] 2>/dev/null; then
  IS_H200=true
  echo ""
  pass "${BOLD}Detected: H200-class server${NC}"
elif $HAS_NVIDIA; then
  echo ""
  info "Detected: GPU server (non-H200)"
else
  echo ""
  info "Detected: CPU-only environment (laptop/dev)"
fi

#=============================================================================
# 2. System checks
#=============================================================================
header "System Checks"

if [ "$TOTAL_MEM_MB" -lt 4000 ] 2>/dev/null; then
  warn "RAM < 4GB — performansi akan terbatas"
fi
DISK_FREE_KB=$(df . | awk 'NR==2 {print $4}')
if [ "$DISK_FREE_KB" -lt 5000000 ] 2>/dev/null; then
  warn "Disk free < 5GB — cargo build membutuhkan ~5-10GB untuk kompilasi"
  warn "  Free: $(df -h . | awk 'NR==2 {print $4}')"
fi

if ! command -v clang &>/dev/null; then
  warn "clang tidak ditemukan (linker). Install: apt install clang lld"
fi
if ! command -v lld &>/dev/null; then
  warn "lld tidak ditemukan (default linker). Install: apt install lld"
fi
if command -v pkg-config &>/dev/null; then
  pass "pkg-config found"
else
  warn "pkg-config tidak ditemukan. Install: apt install pkg-config"
fi
if pkg-config --exists libssl 2>/dev/null || pkg-config --exists openssl 2>/dev/null; then
  pass "libssl found"
else
  warn "libssl-dev tidak ditemukan. Install: apt install libssl-dev"
fi

#=============================================================================
# 3. Generate nexora.toml
#=============================================================================
header "Generate nexora.toml"

CONFIG_FILE="nexora.toml"
if [ -f "$CONFIG_FILE" ]; then
  info "$CONFIG_FILE already exists — skipping generation"
  info "  Delete it to regenerate: rm $CONFIG_FILE && bash setup.sh"
else
  info "Generating $CONFIG_FILE..."

  if $IS_H200; then
    cat > "$CONFIG_FILE" << 'CONFIGEOF'
[core]
enable_ml_intent = true
enable_coordination = true
enable_error_recovery = true
enable_monitoring = true
max_concurrent_requests = 1000
request_timeout_ms = 120000
enable_distributed = false
distributed_listen_address = "0.0.0.0:8080"
distributed_seed_nodes = []
distributed_gossip_interval_ms = 1000

[tokenizer]
vocab_size = 50000
min_frequency = 2
enable_unicode_normalization = true
model_path = "./checkpoints/tokenizer.json"
cache_size = 10000

[models]
vocab_size = 32000
d_model = 768
n_heads = 12
n_layers = 12

[memory]
short_term_capacity = 5000
session_capacity = 20000
long_term_capacity = 100000
knowledge_capacity = 500000
enable_compression = true
compression_threshold = 0.8
enable_persistence = true
persistence_path = "./data/memory"
cleanup_interval_seconds = 300
max_age_hours = 168
eviction_strategy = "LruTtl"
max_memory_mb = 4096

[utils]
enable_crypto = true
enable_text_processing = true
enable_file_operations = true
crypto_algorithm = "aes-256-gcm"
text_processing_language = "en"
file_operations_max_size_mb = 100

[server]
host = "0.0.0.0"
port = 8080
enable_tls = false
max_connections = 10000
request_timeout_seconds = 120
enable_cors = true
cors_origins = ["*"]
api_keys = []
enable_auth = false
rate_limit_rpm = 1000

[api]
base_url = "http://0.0.0.0:8080"
timeout_seconds = 120
max_retries = 3
enable_rate_limiting = true
requests_per_minute = 1000

[logging]
level = "info"
format = "compact"
enable_file_logging = true
file_path = "./logs/nexora.log"
max_file_size_mb = 100
max_files = 30
enable_console_logging = true
enable_structured_logging = true
enable_tracing = true

[isolation.global]
cluster_name = "nexora-h200"
api_gateway_enabled = true
orchestrator_enabled = true
monitoring_enabled = true
storage_isolation = true
scheduler_isolation = true
security_core_enabled = true
service_mesh = "None"
observability_backend = "Prometheus"

[isolation.mode]
enabled = true
default_network_policy = "DenyAll"
default_memory_quota_mb = 65536

[isolation.mode.default_gpu_quota]
count = 1
memory_mb = 140000
share = true

[isolation.agent]
enabled = true
separate_pod_per_agent = false
dedicated_memory_buffer = true
dedicated_runtime = false
max_agents_per_mode = 50
agent_communication = "FirewallWithAudit"

[isolation.tool]
enabled = true
sandbox_per_tool = false
tool_gateway_enabled = true
allowed_tools = ["python", "browser", "terminal", "filesystem"]
max_tool_execution_seconds = 300
tool_network_access = false

[isolation.runtime]
enabled = true
sandbox_kind = "NamespaceOnly"
restricted_linux = true
seccomp_profile = "restricted.json"
read_only_root = false
drop_capabilities = ["ALL"]

[isolation.cognitive]
enabled = true
isolate_memory_per_agent = true
isolate_reasoning_chains = true
max_chain_isolation_depth = 10
auto_quarantine_on_anomaly = true
hallucination_spread_prevention = true

[isolation.permission]
enabled = true
rbac_enabled = true
capability_based_access = true
default_action = "Deny"
audit_all_access = true

[isolation.firewall]
enabled = true
default_egress = "AuditDeny"
default_ingress = "Deny"
rate_limit_per_agent = 100
max_message_size_bytes = 1048576
audit_all_messages = true
filter_suspicious = true

[isolation.kill_switch]
enabled = true
confirm_before_kill = true
grace_period_seconds = 30
auto_kill_on_anomaly = false
kill_scope = "SingleAgent"

[isolation.multi_cluster]
enabled = false
max_regional_clusters = 10
max_mode_clusters = 100
max_agent_clusters = 1000
max_micro_vms = 10000
auto_scaling = true
cross_cluster_sync = true
CONFIGEOF
    pass "Generated H200 production config"
  else
    cat > "$CONFIG_FILE" << 'CONFIGEOF'
[core]
enable_ml_intent = true
enable_coordination = true
enable_error_recovery = true
enable_monitoring = true
max_concurrent_requests = 4
request_timeout_ms = 60000
enable_distributed = false
distributed_listen_address = "127.0.0.1:8080"
distributed_seed_nodes = []
distributed_gossip_interval_ms = 1000

[tokenizer]
vocab_size = 50000
min_frequency = 2
enable_unicode_normalization = true
cache_size = 1000

[models]
vocab_size = 32000
d_model = 768
n_heads = 12
n_layers = 12

[memory]
short_term_capacity = 500
session_capacity = 1000
long_term_capacity = 5000
knowledge_capacity = 10000
enable_compression = true
compression_threshold = 0.8
enable_persistence = true
persistence_path = "./data/memory"
cleanup_interval_seconds = 300
max_age_hours = 24
eviction_strategy = "LruTtl"
max_memory_mb = 512

[utils]
enable_crypto = true
enable_text_processing = true
enable_file_operations = true
crypto_algorithm = "aes-256-gcm"
text_processing_language = "en"
file_operations_max_size_mb = 100

[server]
host = "127.0.0.1"
port = 8080
enable_tls = false
max_connections = 100
request_timeout_seconds = 30
enable_cors = true
cors_origins = ["*"]
api_keys = []
enable_auth = false
rate_limit_rpm = 60

[api]
base_url = "http://127.0.0.1:8080"
timeout_seconds = 30
max_retries = 3
enable_rate_limiting = true
requests_per_minute = 100

[logging]
level = "info"
format = "compact"
enable_file_logging = true
file_path = "./logs/nexora.log"
max_file_size_mb = 50
max_files = 7
enable_console_logging = true
enable_structured_logging = false
enable_tracing = false

[isolation.global]
cluster_name = "nexora-dev"
api_gateway_enabled = false
orchestrator_enabled = false
monitoring_enabled = true
storage_isolation = false
scheduler_isolation = false
security_core_enabled = true
service_mesh = "None"
observability_backend = "None"

[isolation.mode]
enabled = false
default_network_policy = "DenyAll"
default_memory_quota_mb = 1024

[isolation.mode.default_gpu_quota]
count = 0
memory_mb = 0
share = false

[isolation.agent]
enabled = false
separate_pod_per_agent = false
dedicated_memory_buffer = true
dedicated_runtime = false
max_agents_per_mode = 5
agent_communication = "FirewallWithAudit"

[isolation.tool]
enabled = true
sandbox_per_tool = false
tool_gateway_enabled = true
allowed_tools = ["python", "terminal", "filesystem"]
max_tool_execution_seconds = 60
tool_network_access = false

[isolation.runtime]
enabled = false
sandbox_kind = "NamespaceOnly"
restricted_linux = false
seccomp_profile = ""
read_only_root = false
drop_capabilities = []

[isolation.cognitive]
enabled = false
isolate_memory_per_agent = false
isolate_reasoning_chains = false
max_chain_isolation_depth = 3
auto_quarantine_on_anomaly = false
hallucination_spread_prevention = false

[isolation.permission]
enabled = false
rbac_enabled = false
capability_based_access = false
default_action = "Allow"
audit_all_access = false

[isolation.firewall]
enabled = true
default_egress = "AuditAllow"
default_ingress = "Allow"
rate_limit_per_agent = 100
max_message_size_bytes = 1048576
audit_all_messages = false
filter_suspicious = true

[isolation.kill_switch]
enabled = false
confirm_before_kill = false
grace_period_seconds = 5
auto_kill_on_anomaly = false
kill_scope = "SingleAgent"

[isolation.multi_cluster]
enabled = false
max_regional_clusters = 1
max_mode_clusters = 1
max_agent_clusters = 1
max_micro_vms = 1
auto_scaling = false
cross_cluster_sync = false
CONFIGEOF
    pass "Generated dev/laptop config"
  fi
fi

#=============================================================================
# 4. Create directories
#=============================================================================
header "Directories"

mkdir -p logs data/memory checkpoints
pass "logs/"
pass "data/memory/"
pass "checkpoints/"

#=============================================================================
# 5. Build
#=============================================================================
if $CONFIG_ONLY; then
  header "Config-only mode — skipping build"
  echo ""
  echo -e "${BOLD}Setup complete (config only)${NC}"
  echo "  Config:  $CONFIG_FILE"
  echo "  Next:    bash setup.sh [--release] [--cuda]"
  exit 0
fi

header "Build"

BUILD_CMD="cargo build"
CARGO_FEATURES="production"

if $RELEASE; then
  BUILD_CMD="cargo build --release"
  info "Release build (optimized, LTO, stripped)"
else
  info "Debug build (faster compile, unoptimized)"
  info "  Use --release for production: bash setup.sh --release"
fi

BUILD_CMD+=" --features \"$CARGO_FEATURES\""

if $HAS_NVCC && $HAS_NVIDIA; then
  info "CUDA available — will use GPU acceleration at runtime"
fi

echo ""
info "Running: $BUILD_CMD"
echo ""

RUSTFLAGS="${RUSTFLAGS:-}" eval $BUILD_CMD
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
  echo ""
  fail "Build failed with exit code $BUILD_EXIT"
  warn "Common fixes:"
  warn "  - Install dependencies: apt install clang lld pkg-config libssl-dev"
  warn "  - CUDA: install nvidia-cuda-toolkit"
  warn "  - Disk space: cargo clean; rm -rf target"
  exit $BUILD_EXIT
fi
pass "Build successful"

#=============================================================================
# 6. Verify
#=============================================================================
header "Verify"

if $RELEASE; then
  BIN_PATH="./target/release/nexora"
else
  BIN_PATH="./target/debug/nexora"
fi

if [ -f "$BIN_PATH" ]; then
  pass "Binary: $BIN_PATH"
  echo "  Version: $("$BIN_PATH" --version 2>/dev/null || echo "N/A")"
  echo "  Size:    $(du -h "$BIN_PATH" | cut -f1)"
else
  warn "Binary not found at $BIN_PATH — build mungkin gagal"
fi

echo ""
info "Running health check: $BIN_PATH health"
"$BIN_PATH" health 2>/dev/null && pass "Health check passed" || warn "Health check tidak dapat dijalankan (mungkin butuh services)"

#=============================================================================
# 7. Summary
#=============================================================================
header "Setup Complete"

if $IS_H200; then
  echo -e "  ${GREEN}Production-ready${NC} — H200 configuration active"
  echo ""
  echo "  GPU ratio:   ~80% GPU / ~20% CPU (matmul, inference, training on GPU)"
  echo "  Start:       cargo run --release --bin nexora -- --config $CONFIG_FILE start"
  echo "  Health:      cargo run --bin nexora -- --config $CONFIG_FILE health"
else
  echo -e "  ${YELLOW}Dev mode${NC} — no GPU detected, CPU-only"
  echo ""
  echo "  GPU ratio:   0% GPU / 100% CPU"
  echo "  Start:       cargo run --bin nexora -- --config $CONFIG_FILE start"
  echo "  Health:      cargo run --bin nexora -- --config $CONFIG_FILE health"
fi

echo ""
echo -e "  ${BOLD}Key env vars:${NC}"
echo "  RUSTFLAGS=\"-C target-cpu=native\"    # optimize for this CPU"
echo "  CARGO_BUILD_JOBS=8                   # limit parallel jobs"
echo ""
