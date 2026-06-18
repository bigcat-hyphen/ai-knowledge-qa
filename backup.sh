#!/bin/bash
# Backup script for AI Knowledge Base Q&A
# Usage: ./backup.sh [backup_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_PATH="${SCRIPT_DIR}/vector_store.db"
BACKUP_DIR="${1:-${SCRIPT_DIR}/backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${BACKUP_DIR}/vector_store_${TIMESTAMP}.db"

if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

sqlite3 "$DB_PATH" ".backup '${BACKUP_FILE}'"

echo "Backup created: $BACKUP_FILE"

# Keep only the last 7 backups
ls -t "${BACKUP_DIR}"/vector_store_*.db 2>/dev/null | tail -n +8 | xargs -r rm --
echo "Cleanup: kept last 7 backups"
