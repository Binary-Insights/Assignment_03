#!/bin/bash
# PostgreSQL Connection Troubleshooting & Fix Script

echo "=========================================="
echo "PostgreSQL Connection Troubleshooting"
echo "=========================================="
echo ""

# Step 1: Check PostgreSQL status
echo "Step 1: Checking PostgreSQL status..."
sudo service postgresql status
echo ""

# Step 2: Check PostgreSQL logs
echo "Step 2: Checking recent PostgreSQL errors..."
sudo tail -20 /var/log/postgresql/postgresql-*.log | grep -i "error\|fatal" || echo "No recent errors found"
echo ""

# Step 3: Check PostgreSQL is listening
echo "Step 3: Checking if PostgreSQL is listening on port 5432..."
sudo netstat -tlnp | grep 5432 || echo "PostgreSQL not listening on 5432"
echo ""

# Step 4: Try connecting directly
echo "Step 4: Attempting direct connection as postgres user..."
echo "Run this command manually in a terminal:"
echo "  sudo -u postgres psql"
echo ""

# Step 5: Initialize PostgreSQL cluster if needed
echo "Step 5: Checking PostgreSQL cluster..."
PG_VERSION=$(psql --version | grep -oP '\d+' | head -1)
PG_DATA="/var/lib/postgresql/$PG_VERSION/main"

if [ ! -d "$PG_DATA" ]; then
    echo "PostgreSQL cluster not initialized!"
    echo "Initializing cluster..."
    sudo -u postgres /usr/lib/postgresql/$PG_VERSION/bin/initdb -D $PG_DATA
    echo "Cluster initialized. Now starting PostgreSQL..."
    sudo service postgresql start
    sleep 3
fi

echo "PostgreSQL cluster found at: $PG_DATA"
echo ""

# Step 6: Try connection again
echo "Step 6: Testing connection after potential fixes..."
if sudo -u postgres psql -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Connection successful!"
else
    echo "✗ Connection still failing. Checking PostgreSQL socket..."
    ls -la /var/run/postgresql/ 2>/dev/null || echo "Socket directory not found"
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "If still failing, try:"
echo "1. Restart PostgreSQL: sudo service postgresql restart"
echo "2. Check logs: sudo tail -50 /var/log/postgresql/postgresql-*.log"
echo "3. Reinitialize: sudo -u postgres /usr/lib/postgresql/16/bin/initdb -D /var/lib/postgresql/16/main"
echo "=========================================="
