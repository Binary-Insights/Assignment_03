#!/bin/bash
# Install and configure PostgreSQL in WSL

echo "=========================================="
echo "Installing PostgreSQL in WSL"
echo "=========================================="

# Update package list
echo "📦 Updating package list..."
sudo apt-get update -y

# Install PostgreSQL
echo "📦 Installing PostgreSQL..."
sudo apt-get install -y postgresql postgresql-contrib

# Start PostgreSQL service
echo "🚀 Starting PostgreSQL service..."
sudo service postgresql start

# Check status
echo "✓ Checking PostgreSQL status..."
sudo service postgresql status

# Get PostgreSQL version
echo ""
echo "PostgreSQL installed:"
psql --version

# Create superuser connection test
echo ""
echo "✓ PostgreSQL is ready!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Create the airflow user:"
echo "   sudo -u postgres psql -c \"CREATE USER airflow WITH PASSWORD 'airflow';\""
echo ""
echo "2. Create the concept_db database:"
echo "   sudo -u postgres createdb -O airflow concept_db"
echo ""
echo "3. Update .env file with:"
echo "   DB_HOST=localhost"
echo "   DB_PORT=5432"
echo "   DB_NAME=concept_db"
echo "   DB_USER=airflow"
echo "   DB_PASSWORD=airflow"
echo ""
echo "4. Test connection:"
echo "   psql -h localhost -U airflow -d concept_db -c 'SELECT 1;'"
echo "=========================================="
