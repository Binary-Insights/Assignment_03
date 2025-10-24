#!/bin/bash
# Quick PostgreSQL Status Check - WSL
# Simple one-liner checks

echo "🔍 PostgreSQL Status Check"
echo ""

# Check 1: Service running
echo -n "1️⃣  Service running: "
if sudo service postgresql status > /dev/null 2>&1; then
    echo "✅ YES"
else
    echo "❌ NO - Run: sudo service postgresql start"
    exit 1
fi

# Check 2: Can connect as postgres
echo -n "2️⃣  Postgres user connection: "
if sudo -u postgres psql -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ YES"
else
    echo "❌ NO"
    exit 1
fi

# Check 3: concept_db exists
echo -n "3️⃣  concept_db database: "
if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw concept_db; then
    echo "✅ EXISTS"
else
    echo "❌ NOT FOUND - Creating..."
    sudo -u postgres createdb -O airflow concept_db 2>/dev/null
    echo "✅ CREATED"
fi

# Check 4: Airflow user can connect
echo -n "4️⃣  Airflow user connection: "
if psql -h localhost -U airflow -d concept_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ YES"
else
    echo "❌ NO"
    exit 1
fi

# Check 5: Version
echo -n "5️⃣  PostgreSQL version: "
psql --version | xargs echo

echo ""
echo "✅ PostgreSQL is healthy!"
echo ""
echo "Connection string for .env:"
echo "  DB_HOST=localhost"
echo "  DB_PORT=5432"
echo "  DB_NAME=concept_db"
echo "  DB_USER=airflow"
echo "  DB_PASSWORD=airflow"
