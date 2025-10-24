#!/bin/bash
# Quick PostgreSQL Fix Script

set -e

echo "=========================================="
echo "PostgreSQL Quick Fix"
echo "=========================================="
echo ""

# Get PostgreSQL version
PG_VERSION=$(psql --version | grep -oP '\d+' | head -1)
echo "PostgreSQL Version: $PG_VERSION"
echo ""

# Step 1: Restart service
echo "Step 1: Restarting PostgreSQL service..."
sudo service postgresql restart
sleep 3
echo "✓ Service restarted"
echo ""

# Step 2: Check if cluster is initialized
PG_DATA="/var/lib/postgresql/$PG_VERSION/main"
echo "Step 2: Checking cluster at $PG_DATA..."

if [ ! -d "$PG_DATA" ]; then
    echo "⚠️  Cluster not initialized! Initializing now..."
    sudo -u postgres /usr/lib/postgresql/$PG_VERSION/bin/initdb -D $PG_DATA -E UTF8
    echo "✓ Cluster initialized"
    
    # Restart after init
    sudo service postgresql restart
    sleep 3
else
    echo "✓ Cluster directory exists"
fi
echo ""

# Step 3: Test basic connection
echo "Step 3: Testing basic PostgreSQL connection..."
if sudo -u postgres psql -c "SELECT version();" > /dev/null 2>&1; then
    echo "✓ PostgreSQL connection successful!"
    sudo -u postgres psql -c "SELECT version();"
else
    echo "✗ Connection failed. Checking socket permissions..."
    sudo chmod 777 /var/run/postgresql
    sudo service postgresql restart
    sleep 2
    
    if sudo -u postgres psql -c "SELECT 1;" > /dev/null 2>&1; then
        echo "✓ Fixed! Connection now works"
    else
        echo "✗ Still failing. Check logs: sudo tail -50 /var/log/postgresql/*.log"
        exit 1
    fi
fi
echo ""

# Step 4: Create airflow user
echo "Step 4: Setting up airflow user..."
sudo -u postgres psql -c "DROP USER IF EXISTS airflow;" 2>/dev/null || true
sudo -u postgres psql -c "CREATE USER airflow WITH PASSWORD 'airflow';"
sudo -u postgres psql -c "ALTER USER airflow CREATEDB;"
echo "✓ Airflow user created"
echo ""

# Step 5: Create concept_db
echo "Step 5: Creating concept_db database..."
sudo -u postgres dropdb concept_db 2>/dev/null || true
sudo -u postgres createdb -O airflow concept_db
echo "✓ Database created"
echo ""

# Step 6: Test application connection
echo "Step 6: Testing application connection..."
if PGPASSWORD=airflow psql -h localhost -U airflow -d concept_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Application connection works!"
else
    echo "✗ Application connection failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ PostgreSQL Setup Complete!"
echo "=========================================="
echo ""
echo "Your .env settings:"
echo "  DB_HOST=localhost"
echo "  DB_PORT=5432"
echo "  DB_NAME=concept_db"
echo "  DB_USER=airflow"
echo "  DB_PASSWORD=airflow"
echo ""
echo "Test with:"
echo "  psql -h localhost -U airflow -d concept_db"
echo "=========================================="
