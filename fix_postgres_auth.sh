#!/bin/bash
# Fix PostgreSQL Authentication - with proper sudo handling

set -e

echo "=========================================="
echo "PostgreSQL Authentication Fix (WSL)"
echo "=========================================="
echo ""

PG_VERSION=16
PG_DATA="/var/lib/postgresql/$PG_VERSION/main"
PG_HBA="$PG_DATA/pg_hba.conf"

echo "Step 1: Checking permissions..."
echo "  PostgreSQL data directory: $PG_DATA"
sudo ls -la $PG_DATA/pg_hba.conf 2>/dev/null || echo "  File not accessible"
echo ""

echo "Step 2: Creating temporary copy of pg_hba.conf..."
sudo cat $PG_HBA > /tmp/pg_hba.conf.temp
echo "✓ Copied to /tmp/pg_hba.conf.temp"
echo ""

echo "Step 3: Backing up original..."
sudo cp $PG_HBA "$PG_HBA.backup"
echo "✓ Backup created"
echo ""

echo "Step 4: Updating authentication method..."
# Update the configuration in the temporary file
sed -i 's/^\(local.*all.*all.*\)md5$/\1trust/' /tmp/pg_hba.conf.temp
sed -i 's/^\(local.*all.*all.*\)scram-sha-256$/\1trust/' /tmp/pg_hba.conf.temp
sed -i 's/^\(local.*all.*all.*\)password$/\1trust/' /tmp/pg_hba.conf.temp

# Copy back with sudo
sudo cp /tmp/pg_hba.conf.temp $PG_HBA
sudo chown postgres:postgres $PG_HBA
sudo chmod 600 $PG_HBA
echo "✓ pg_hba.conf updated"
echo ""

echo "Step 5: Showing updated authentication lines..."
sudo grep -E "^local|^host" $PG_HBA | head -10 || true
echo ""

echo "Step 6: Restarting PostgreSQL..."
sudo service postgresql restart
sleep 3
echo "✓ PostgreSQL restarted"
echo ""

echo "Step 7: Testing connection as postgres user..."
if sudo -u postgres psql -c "SELECT version();" 2>/dev/null; then
    echo "✓ Connection successful!"
    echo ""
else
    echo "✗ Connection failed - checking PostgreSQL status..."
    sudo service postgresql status
    exit 1
fi

echo "Step 8: Creating airflow user..."
sudo -u postgres psql -c "DROP USER IF EXISTS airflow;" 2>/dev/null || true
sudo -u postgres psql -c "CREATE USER airflow WITH CREATEDB;" 
echo "✓ Airflow user created"
echo ""

echo "Step 9: Creating concept_db..."
sudo -u postgres dropdb concept_db 2>/dev/null || true
sudo -u postgres createdb -O airflow concept_db
echo "✓ Database created"
echo ""

echo "Step 10: Testing connection as airflow..."
if sudo -u postgres psql -U airflow -d concept_db -c "SELECT 1;" 2>/dev/null; then
    echo "✓ Airflow connection works!"
else
    echo "✗ Airflow connection failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ PostgreSQL Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration for .env:"
echo "  DB_HOST=localhost"
echo "  DB_PORT=5432"
echo "  DB_NAME=concept_db"
echo "  DB_USER=airflow"
echo "  DB_PASSWORD="
echo ""
echo "Test connection:"
echo "  psql -U airflow -d concept_db -c 'SELECT 1;'"
echo "=========================================="
