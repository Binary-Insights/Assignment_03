#!/bin/bash
# PostgreSQL Health Check Script for WSL
# Checks if PostgreSQL is running and functioning properly

set -e

echo "=========================================="
echo "PostgreSQL Health Check - WSL"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS=0

# Function to print status
print_status() {
    local status=$1
    local message=$2
    
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
        OVERALL_STATUS=1
    fi
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_section() {
    echo ""
    echo -e "${YELLOW}========== $1 ==========${NC}"
}

# ============ Check 1: Service Status ============
print_section "1. Service Status"

if sudo service postgresql status > /dev/null 2>&1; then
    print_status 0 "PostgreSQL service is RUNNING"
else
    print_status 1 "PostgreSQL service is NOT running"
    echo -e "${YELLOW}  Try: sudo service postgresql start${NC}"
fi

# ============ Check 2: PostgreSQL Version ============
print_section "2. PostgreSQL Version"

if command -v psql &> /dev/null; then
    VERSION=$(psql --version)
    print_status 0 "$VERSION"
else
    print_status 1 "PostgreSQL client (psql) not installed"
fi

# ============ Check 3: Local Connection ============
print_section "3. Local Connection Test"

if sudo -u postgres psql -c "SELECT 1;" > /dev/null 2>&1; then
    print_status 0 "Local connection as postgres user: SUCCESS"
else
    print_status 1 "Local connection as postgres user: FAILED"
fi

# ============ Check 4: Airflow User ============
print_section "4. Airflow User Status"

if sudo -u postgres psql -c "SELECT 1;" -U airflow > /dev/null 2>&1; then
    print_status 0 "Airflow user exists and is accessible"
else
    echo -e "${YELLOW}  Creating airflow user...${NC}"
    sudo -u postgres psql -c "CREATE USER IF NOT EXISTS airflow WITH PASSWORD 'airflow';" > /dev/null 2>&1
    if sudo -u postgres psql -c "SELECT 1;" -U airflow > /dev/null 2>&1; then
        print_status 0 "Airflow user created successfully"
    else
        print_status 1 "Failed to create/access airflow user"
    fi
fi

# ============ Check 5: concept_db Database ============
print_section "5. concept_db Database Status"

if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw concept_db; then
    print_status 0 "concept_db database exists"
    
    # Check if airflow user is the owner
    OWNER=$(sudo -u postgres psql -lqt | grep concept_db | awk -F '|' '{print $3}' | xargs)
    print_info "  Database owner: $OWNER"
else
    echo -e "${YELLOW}  Creating concept_db database...${NC}"
    sudo -u postgres createdb -O airflow concept_db > /dev/null 2>&1
    print_status 0 "concept_db database created"
fi

# ============ Check 6: Connection as Airflow User ============
print_section "6. Connection Test (airflow user)"

if psql -h localhost -U airflow -d concept_db -c "SELECT 1;" > /dev/null 2>&1; then
    print_status 0 "Connection as airflow@localhost:5432/concept_db: SUCCESS"
else
    print_status 1 "Connection as airflow@localhost:5432/concept_db: FAILED"
    echo -e "${YELLOW}  Check password, user, or database exists${NC}"
fi

# ============ Check 7: Database Size ============
print_section "7. Database Information"

if sudo -u postgres psql -d concept_db -c "SELECT pg_size_pretty(pg_database_size('concept_db'));" > /dev/null 2>&1; then
    SIZE=$(sudo -u postgres psql -d concept_db -c "SELECT pg_size_pretty(pg_database_size('concept_db'));" | tail -1 | xargs)
    print_info "  Database size: $SIZE"
else
    print_info "  Could not determine database size"
fi

# ============ Check 8: Tables in concept_db ============
print_section "8. Tables in concept_db"

TABLE_COUNT=$(psql -h localhost -U airflow -d concept_db -c "\dt" 2>/dev/null | tail -1 | head -c 1)
if [ -z "$TABLE_COUNT" ]; then
    TABLE_COUNT=0
fi

print_info "  Number of tables: $TABLE_COUNT"

if [ "$TABLE_COUNT" -gt "0" ]; then
    echo -e "${BLUE}  Tables:${NC}"
    psql -h localhost -U airflow -d concept_db -c "\dt" 2>/dev/null | grep -v "^(" | grep -v "^$" | awk '{if(NF>=3) print "    - " $3}'
else
    echo -e "${YELLOW}  No tables found (database is empty)${NC}"
fi

# ============ Check 9: Connection from Application ============
print_section "9. Application Connection Test"

# Build a test connection string
TEST_RESULT=$(PGPASSWORD=airflow psql -h localhost -U airflow -d concept_db -c "SELECT version();" 2>&1)

if [ $? -eq 0 ]; then
    print_status 0 "Application connection string works"
    print_info "  $(echo $TEST_RESULT | head -1)"
else
    print_status 1 "Application connection string failed"
fi

# ============ Check 10: PostgreSQL Configuration ============
print_section "10. PostgreSQL Configuration"

LISTEN_ADDR=$(sudo -u postgres psql -c "SHOW listen_addresses;" 2>/dev/null | tail -1 | xargs)
MAX_CONNECTIONS=$(sudo -u postgres psql -c "SHOW max_connections;" 2>/dev/null | tail -1 | xargs)

print_info "  Listen addresses: $LISTEN_ADDR"
print_info "  Max connections: $MAX_CONNECTIONS"

# ============ Check 11: Disk Space ============
print_section "11. Disk Space"

POSTGRES_DATA_PATH="/var/lib/postgresql"
if [ -d "$POSTGRES_DATA_PATH" ]; then
    DISK_USAGE=$(du -sh "$POSTGRES_DATA_PATH" 2>/dev/null | awk '{print $1}')
    print_info "  PostgreSQL data directory size: $DISK_USAGE"
else
    print_info "  PostgreSQL data directory not found"
fi

# ============ Summary ============
print_section "Summary"

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! PostgreSQL is healthy.${NC}"
    echo ""
    echo "Your .env should have:"
    echo "  DB_HOST=localhost"
    echo "  DB_PORT=5432"
    echo "  DB_NAME=concept_db"
    echo "  DB_USER=airflow"
    echo "  DB_PASSWORD=airflow"
else
    echo -e "${RED}✗ Some checks failed. See details above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  1. Start PostgreSQL: sudo service postgresql start"
    echo "  2. Create user: sudo -u postgres psql -c \"CREATE USER airflow WITH PASSWORD 'airflow';\""
    echo "  3. Create database: sudo -u postgres createdb -O airflow concept_db"
    echo "  4. Restart service: sudo service postgresql restart"
fi

echo ""
echo "=========================================="

# Return overall status
exit $OVERALL_STATUS
