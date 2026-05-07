# Nexora-AI Database Layer (L4)

## Overview

L4 menyediakan infrastruktur database lengkap untuk Nexora-AI, mencakup 6 jenis database yang dioptimasi untuk use case berbeda.

## Architecture

```
Nexora-AI Services
        │
        ├── RAG Query    ──────────► Qdrant (Vector DB)
        │                           Port 6333
        │
        ├── Session/Cache ─────────► Redis (KV Store)  
        │                           Port 6379
        │
        ├── Corpus Metadata ───────► MongoDB (Document)
        │                           Port 27017
        │
        ├── User/Auth/Billing ─────► PostgreSQL (Relational)
        │                           Port 5432
        │
        ├── Metrics/Monitoring ────► Prometheus → Grafana
        │                           Port 9090 / 3000
        │
        └── Checkpoints/Files ─────► MinIO (Object Storage)
                                    Port 9000
```

## Quick Start

### 1. Start All Databases

```bash
cd /home/whale-d/nexora/nexora-ai
docker-compose up -d
```

### 2. Initialize PostgreSQL Schema

```bash
# Schema will be auto-loaded from database/init.sql on first start
# To manually reinitialize:
docker exec -i nexora-postgres psql -U nexora_user -d nexora < database/init.sql
```

### 3. Initialize MongoDB Schema

```bash
mongosh --host localhost --port 27017 -u nexora_admin -p nexora_secure_password --authenticationDatabase admin nexora < database/mongodb_schema.js
```

### 4. Access Services

- **Qdrant Console**: http://localhost:6333/dashboard
- **Redis CLI**: `docker exec -it nexora-redis redis-cli`
- **MongoDB Shell**: `mongosh --host localhost --port 27017`
- **PostgreSQL**: `docker exec -it nexora-postgres psql -U nexora_user -d nexora`
- **Grafana**: http://localhost:3000 (admin/nexora_admin)
- **Prometheus**: http://localhost:9090
- **MinIO Console**: http://localhost:9001 (nexora_admin/nexora_secure_password)

## Database Details

### Layer 1: Qdrant (Vector Database)

**Use Case**: Semantic search for RAG pipeline

**Features**:
- Vector similarity search with cosine distance
- Payload filtering (search + filter by metadata)
- Sparse + dense hybrid search (BM25 + semantic)
- On-disk indexing for large corpora

**Port**: 6333 (HTTP), 6334 (gRPC)

**API**: HTTP REST API

### Layer 2: Redis (Key-Value Store)

**Use Case**: Session state, rate limiting, KV cache metadata

**Features**:
- Sub-millisecond latency
- TTL for automatic expiration
- LRU eviction policy (configured: `maxmemory-policy allkeys-lru`)
- Atomic operations for rate limiting

**Port**: 6379

**Max Memory**: 2GB (configurable)

### Layer 3: MongoDB (Document Store)

**Use Case**: Corpus metadata, training logs, conversation analytics

**Collections**:
- `corpus_documents`: Metadata for each training document
- `corpus_statistics`: Aggregated corpus statistics
- `deduplication_groups`: Document deduplication groups
- `training_runs`: Training run metadata
- `training_checkpoints`: Checkpoint metadata
- `training_metrics`: Detailed training metrics
- `evaluation_results`: Benchmark evaluation results
- `conversation_logs`: Anonymized conversation logs
- `conversation_analytics`: Aggregated conversation analytics
- `pipeline_jobs`: Data pipeline job metadata
- `synthetic_data`: Synthetic data generation metadata
- `system_events`: System events and alerts

**Port**: 27017

**Authentication**: nexora_admin / nexora_secure_password

### Layer 4: PostgreSQL (Relational Database)

**Use Case**: User management, billing, A/B testing, analytics

**Tables**:
- `users`: User accounts and authentication
- `subscription_tiers`: Subscription tiers (free, pro, enterprise)
- `api_keys`: API keys per user
- `monthly_usage`: Monthly usage tracking
- `usage_logs`: Detailed per-request usage logs
- `rate_limits`: Rate limiting buckets
- `checkpoints`: Model checkpoints
- `evaluations`: Evaluation results
- `ab_experiments`: A/B test experiments
- `ab_assignments`: User assignments to A/B tests
- `ab_results`: A/B test results
- `conversations`: Conversation history
- `messages`: Individual messages
- `daily_metrics`: Daily aggregated metrics
- `model_performance`: Model performance tracking

**Port**: 5432 (direct), 6432 (via PgBouncer)

**Connection Pooling**: PgBouncer (transaction mode, 25 pool size)

**Extensions**: pgvector, pg_trgm

### Layer 5: Prometheus + InfluxDB (Time-Series)

**Use Case**: Real-time metrics and long-term analytics

**Prometheus** (Port 9090):
- Inference latency (TTFT, total generation time)
- Throughput (tokens per second)
- GPU utilization
- KV cache hit rate
- Queue depth
- Retention: 30 days

**InfluxDB** (Port 8086):
- Training loss curves
- Eval scores per checkpoint
- Data pipeline statistics
- Retention: 365 days

**Grafana** (Port 3000):
- Dashboard: http://localhost:3000
- Default credentials: admin / nexora_admin
- Pre-configured Nexora-AI dashboard

### Layer 6: MinIO (Object Storage)

**Use Case**: Model checkpoints, binary files

**Buckets**:
- `checkpoints`: Model checkpoints
- `tokenizers`: Tokenizer files
- `training-data`: Training data binaries
- `eval-artifacts`: Evaluation artifacts
- `corpus`: Corpus files

**Port**: 9000 (API), 9001 (Console)

**S3 Compatible**: Yes (use any S3 client)

**Credentials**: nexora_admin / nexora_secure_password

## C API Usage

### PostgreSQL

```c
#include "database.h"

// Connect
PGConnection* pg = pg_connection_create("localhost", 5432, "nexora", "nexora_user", "password");

// Execute query
PGResult* result = pg_query(pg, "SELECT * FROM users");

// Get values
const char* email = pg_get_value(result, 0, 0);
int tier_id = pg_get_int(result, 0, 1);

// Cleanup
pg_result_free(result);
pg_connection_free(pg);
```

### Redis

```c
#include "database.h"

// Connect
RedisConnection* redis = redis_connection_create("localhost", 6379);

// Set key with TTL
redis_set(redis, "session:123", "data");
redis_expire(redis, "session:123", 3600);

// Get value
char* value = redis_get(redis, "session:123");

// Rate limiting
int64_t count = redis_incr(redis, "rate_limit:user_456");
if (count == 1) {
    redis_expire(redis, "rate_limit:user_456", 60);
}

// Cleanup
free(value);
redis_connection_free(redis);
```

### Qdrant

```c
#include "database.h"

// Connect
QdrantClient* qdrant = qdrant_client_create("localhost", 6333, NULL);

// Create collection
qdrant_create_collection(qdrant, "corpus_vectors", 4096, "Cosine");

// Insert vectors
QdrantPoint points[10];
// ... populate points
qdrant_upsert_points(qdrant, "corpus_vectors", points, 10);

// Search
QdrantSearchResult* results = qdrant_search(qdrant, "corpus_vectors", query_vector, 4096, 10, NULL);

// Cleanup
qdrant_search_results_free(results, 10);
qdrant_client_free(qdrant);
```

## Configuration

Database configuration can be loaded from a config file or environment variables:

```c
DatabaseConfig* config = db_config_load("config/database.conf");
DatabaseConnections* conns = db_connections_init(config);

// Use connections
// ...

// Cleanup
db_connections_free(conns);
db_config_free(config);
```

## Dependencies

To compile with full database support:

```bash
# PostgreSQL
sudo apt-get install libpq-dev

# Redis
sudo apt-get install libhiredis-dev

# Qdrant (HTTP client)
sudo apt-get install libcurl4-openssl-dev libcjson-dev

# Compile with flags
gcc -DUSE_POSTGRES -DUSE_REDIS -DUSE_QDRANT \
    -I/home/whale-d/nexora/nexora-ai/include \
    -lpq -lhiredis -lcurl -lcjson \
    database.c -o database.o
```

## Backup and Restore

### PostgreSQL

```bash
# Backup
docker exec nexora-postgres pg_dump -U nexora_user nexora > backup.sql

# Restore
docker exec -i nexora-postgres psql -U nexora_user nexora < backup.sql
```

### MongoDB

```bash
# Backup
docker exec nexora-mongodb mongodump --db nexora --out /backup

# Restore
docker exec nexora-mongodb mongorestore --db nexora /backup/nexora
```

### Redis

```bash
# Backup (RDB snapshot)
docker exec nexora-redis redis-cli BGSAVE
docker cp nexora-redis:/data/dump.rdb ./redis_backup.rdb

# Restore
docker cp ./redis_backup.rdb nexora-redis:/data/dump.rdb
docker restart nexora-redis
```

### MinIO

```bash
# Use mc (MinIO Client) or any S3 client
mc alias set nexora http://localhost:9000 nexora_admin nexora_secure_password
mc mirror nexora/checkpoints ./backup/checkpoints
```

## Monitoring

Access Grafana dashboard at http://localhost:3000

Pre-configured panels:
- Inference Request Rate
- Inference Latency (P50, P95)
- Training Loss
- GPU Utilization
- Database Memory Usage
- KV Cache Hit Rate

## Troubleshooting

### Database won't start

```bash
# Check logs
docker-compose logs qdrant
docker-compose logs redis
docker-compose logs postgres
docker-compose logs mongodb
```

### Connection refused

- Ensure Docker containers are running: `docker-compose ps`
- Check port conflicts: `netstat -tulpn | grep <port>`
- Verify firewall settings

### Out of memory

- Redis: Adjust `maxmemory` in docker-compose.yml
- PostgreSQL: Adjust shared_buffers in postgresql.conf
- MongoDB: Adjust wiredTigerCacheSizeGB

## Security Notes

**Default passwords should be changed in production:**

- PostgreSQL: nexora_secure_password
- MongoDB: nexora_secure_password
- Redis: (no auth by default, configure in production)
- MinIO: nexora_secure_password
- Grafana: nexora_admin

Change these in `docker-compose.yml` before deploying to production.
