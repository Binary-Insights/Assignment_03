#!/bin/bash
# Quick PostgreSQL setup for WSL
# Run this script inside WSL to install and configure PostgreSQL

set -e  # Exit on error

echo "=========================================="
echo "PostgreSQL WSL Quick Setup"
echo "=========================================="
echo ""

# Step 1: Update and install
echo "📦 Step 1: Installing PostgreSQL..."
sudo apt-get update -y > /dev/null 2>&1
sudo apt-get install -y postgresql postgresql-contrib > /dev/null 2>&1
echo "✓ PostgreSQL installed"

# Step 2: Start service
echo "🚀 Step 2: Starting PostgreSQL service..."
sudo service postgresql start > /dev/null 2>&1
echo "✓ PostgreSQL service started"

# Step 3: Create airflow user
echo "👤 Step 3: Creating 'airflow' user..."
sudo -u postgres psql -c "CREATE USER IF NOT EXISTS airflow WITH PASSWORD 'airflow';" > /dev/null 2>&1
echo "✓ 'airflow' user created"

# Step 4: Create concept_db
echo "🗄️  Step 4: Creating 'concept_db' database..."
sudo -u postgres dropdb concept_db 2>/dev/null || true
sudo -u postgres createdb -O airflow concept_db > /dev/null 2>&1
echo "✓ 'concept_db' database created"

# Step 5: Grant privileges
echo "🔐 Step 5: Setting up privileges..."
sudo -u postgres psql -d concept_db -c "GRANT ALL PRIVILEGES ON DATABASE concept_db TO airflow;" > /dev/null 2>&1
echo "✓ Privileges configured"

# Step 6: Test connection
echo ""
echo "🧪 Step 6: Testing connection..."
if psql -h localhost -U airflow -d concept_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Connection test passed!"
else
    echo "✗ Connection test failed"
    exit 1
fi

# Step 7: Show version
echo ""
echo "ℹ️  PostgreSQL version:"
psql --version

echo ""
echo "=========================================="
echo "✅ PostgreSQL setup complete!"
echo "=========================================="
echo ""
echo "Your .env file should have:"
echo "  DB_HOST=localhost"
echo "  DB_PORT=5432"
echo "  DB_NAME=concept_db"
echo "  DB_USER=airflow"
echo "  DB_PASSWORD=airflow"
echo ""
echo "To verify:"
echo "  psql -h localhost -U airflow -d concept_db -c 'SELECT 1;'"
echo ""
echo "To start PostgreSQL on next WSL login (optional):"
echo "  echo 'sudo service postgresql start' >> ~/.bashrc"
echo ""
echo "=========================================="
