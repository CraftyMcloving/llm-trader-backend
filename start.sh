#!/usr/bin/env bash
set -euo pipefail

DB_PATH="/data/ai_trade_feedback.db"

# Restore if DB missing (first deploy or after restart on fresh disk)
if [ ! -f "$DB_PATH" ]; then
  echo "Restoring SQLite from Litestream replica..."
  litestream restore -if-replica-exists -config /etc/litestream.yml "$DB_PATH" || true
fi

# Run Litestream replication in background
litestream replicate -config /etc/litestream.yml &

# Start API
exec uvicorn main:app --host 0.0.0.0 --port 8000