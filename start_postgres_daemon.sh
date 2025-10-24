#!/bin/bash
# PostgreSQL Diagnostic and Recovery Script

set -e

echo "=========================================="
echo "PostgreSQL Diagnostic & Recovery"
echo "=========================================="
echo ""

PG_VERSION=16
PG_DATA="/var/lib/postgresql/$PG_VERSION/main"
PG_BIN="/usr/lib/postgresql/$PG_VERSION/bin"

# Step 1: Check if PostgreSQL is installed
echo "Step 1: Checking PostgreSQL installation..."
if [ ! -d "$PG_BIN" ]; then
    echo "✗ PostgreSQL binary not found at $PG_BIN"
    exit 1
fi
echo "✓ PostgreSQL binaries found"
echo ""

# Step 2: Check data directory
echo "Step 2: Checking data directory..."
if [ ! -d "$PG_DATA" ]; then
    echo "✗ Data directory not found at $PG_DATA"
    echo "  Initializing cluster..."
    sudo -u postgres "$PG_BIN/initdb" -D "$PG_DATA" -E UTF8 --locale=C
fi
echo "✓ Data directory: $PG_DATA"
echo ""

# Step 3: Check permissions
echo "Step 3: Checking permissions..."
OWNER=$(ls -ld "$PG_DATA" | awk '{print $3}')
echo "  Directory owner: $OWNER"
if [ "$OWNER" != "postgres" ]; then
    echo "  ✗ Wrong owner! Fixing..."
    sudo chown -R postgres:postgres "$PG_DATA"
fi
echo "✓ Permissions OK"
echo ""

# Step 4: Stop any running instances
echo "Step 4: Stopping any running PostgreSQL..."
sudo pkill -9 postgres 2>/dev/null || true
sleep 1
echo "✓ Stopped"
echo ""

# Step 5: Remove old socket
echo "Step 5: Cleaning up old sockets..."
sudo rm -f /var/run/postgresql/.s.PGSQL.5432* 2>/dev/null || true
echo "✓ Cleaned"
echo ""

# Step 6: Create socket directory
echo "Step 6: Setting up socket directory..."
sudo mkdir -p /var/run/postgresql
sudo chown postgres:postgres /var/run/postgresql
sudo chmod 700 /var/run/postgresql
echo "✓ Socket directory ready"
echo ""

# Step 7: Start PostgreSQL in foreground (for debugging)
echo "Step 7: Starting PostgreSQL..."
echo "  Running: sudo -u postgres $PG_BIN/postgres -D $PG_DATA -F"
echo ""

# Run in background and capture PID
sudo -u postgres "$PG_BIN/postgres" -D "$PG_DATA" -F > /tmp/postgres.log 2>&1 &
POSTGRES_PID=$!
echo "  PostgreSQL PID: $POSTGRES_PID"

# Wait for startup
sleep 3

# Step 8: Check if running
echo ""
echo "Step 8: Checking if PostgreSQL is running..."
if ps -p $POSTGRES_PID > /dev/null 2>&1; then
    echo "✓ PostgreSQL process is running (PID: $POSTGRES_PID)"
else
    echo "✗ PostgreSQL process died!"
    echo ""
    echo "PostgreSQL output:"
    cat /tmp/postgres.log
    exit 1
fi
echo ""

# Step 9: Test connection
echo "Step 9: Testing connection..."
for attempt in {1..5}; do
    if psql -U postgres -c "SELECT version();" 2>/dev/null; then
        echo "✓ Connection successful on attempt $attempt"
        break
    fi
    echo "  Attempt $attempt/5... waiting..."
    sleep 1
done
echo ""

# Step 10: Show status
echo "Step 10: PostgreSQL status..."
ps aux | grep postgres | grep -v grep | head -5
echo ""

echo "=========================================="
echo "✅ PostgreSQL is ready!"
echo "=========================================="
echo ""
echo "Logs: /tmp/postgres.log"
echo "To see live logs: tail -f /tmp/postgres.log"
echo ""
