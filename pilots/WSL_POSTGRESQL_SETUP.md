# PostgreSQL Setup in WSL (Windows Subsystem for Linux)

## Overview

This guide shows how to install and use PostgreSQL in WSL (Windows Subsystem for Linux).

---

## Part 1: Check Your WSL Installation

### Verify WSL is Installed

Open PowerShell and run:
```powershell
wsl --list --verbose
```

Should show something like:
```
NAME            STATE           VERSION
Ubuntu          Running         2
```

If not installed, install WSL2:
```powershell
wsl --install
```

### Update WSL
```powershell
wsl --update
```

---

## Part 2: Install PostgreSQL in WSL

### Step 1: Open WSL Terminal

From PowerShell:
```powershell
wsl
```

Or open Ubuntu from Start Menu.

### Step 2: Update Package Manager

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 3: Install PostgreSQL

```bash
sudo apt install postgresql postgresql-contrib -y
```

### Step 4: Verify Installation

```bash
psql --version
```

Should show: `psql (PostgreSQL) 14.x`

### Step 5: Install Additional Tools (Optional)

```bash
sudo apt install pgadmin4 -y
```

---

## Part 3: Start PostgreSQL Service in WSL

### Check PostgreSQL Status

```bash
sudo service postgresql status
```

### Start PostgreSQL Service

```bash
sudo service postgresql start
```

### Verify It's Running

```bash
sudo service postgresql status
```

Should show: `11/main (port 5432): online`

### Make PostgreSQL Start Automatically (Optional)

Edit `/etc/wsl.conf` to auto-start PostgreSQL:

```bash
sudo nano /etc/wsl.conf
```

Add this content:
```ini
[boot]
systemd=true

[interop]
appendWindowsPath=true
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

---

## Part 4: Create Database in WSL PostgreSQL

### Connect to PostgreSQL

```bash
sudo -u postgres psql
```

You should see:
```
postgres=#
```

### Create concept_db Database

```sql
CREATE DATABASE concept_db;
```

### Create PostgreSQL User (Optional but Recommended)

```sql
CREATE USER rag_user WITH PASSWORD 'rag_password123';
```

### Grant Permissions

```sql
ALTER ROLE rag_user CREATEDB;
GRANT ALL PRIVILEGES ON DATABASE concept_db TO rag_user;
```

### Verify Database Created

```sql
\l
```

Should show:
```
 concept_db | postgres | UTF8     | C       | C       |
```

### Exit psql

```sql
\q
```

---

## Part 5: Configure PostgreSQL to Accept Remote Connections

This allows your Python code (running in Windows or WSL) to connect to PostgreSQL.

### Edit PostgreSQL Configuration

```bash
sudo nano /etc/postgresql/14/main/postgresql.conf
```

Find the line `#listen_addresses = 'localhost'` and change it to:
```
listen_addresses = '*'
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Edit pg_hba.conf

```bash
sudo nano /etc/postgresql/14/main/pg_hba.conf
```

Find the line:
```
host    all             all             127.0.0.1/32            md5
```

Change it to:
```
host    all             all             0.0.0.0/0               md5
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Restart PostgreSQL

```bash
sudo service postgresql restart
```

---

## Part 6: Find WSL PostgreSQL Host Address

Your Python code needs to know where PostgreSQL is running.

### Get WSL IP Address

In WSL bash:
```bash
ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
```

Or simply:
```bash
hostname -I
```

This gives you the WSL IP (usually something like `172.xx.xx.xx`).

---

## Part 7: Update .env File

Create or edit `.env` in your Windows project folder:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...

# PostgreSQL in WSL
DB_HOST=172.xx.xx.xx
DB_PORT=5432
DB_NAME=concept_db
DB_USER=postgres
DB_PASSWORD=postgres
```

**Replace `172.xx.xx.xx` with the IP from Part 6.**

### Alternative: Use localhost

If you're running Python code in WSL (not Windows), use:
```env
DB_HOST=localhost
```

---

## Part 8: Install Python Packages

### In WSL

```bash
pip install psycopg2-binary
pip install instructor
pip install langchain-openai
pip install pinecone-client
pip install fastapi
pip install uvicorn
pip install wikipedia
pip install python-dotenv
```

Or install from requirements:
```bash
pip install -r requirements-rag.txt
```

---

## Part 9: Test Connection from Windows Python

### Create Test Script

Create `test_db_connection.py`:

```python
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    print(f"✓ Connected successfully!")
    print(f"PostgreSQL version: {db_version[0]}")
    
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Run Test

In PowerShell (in your project directory):
```powershell
python test_db_connection.py
```

Should show:
```
✓ Connected successfully!
PostgreSQL version: PostgreSQL 14.x on ...
```

---

## Part 10: Run Enhanced RAG System

### Option 1: Run from Windows PowerShell

```powershell
python enhanced_fastapi_server.py
```

### Option 2: Run from WSL

```bash
python enhanced_fastapi_server.py
```

The system will automatically create tables on first run.

---

## Part 11: Access PostgreSQL from WSL

### Connect with PostgreSQL User

```bash
psql -U postgres -d concept_db -h localhost
```

### Connect as Another User

```bash
psql -U rag_user -d concept_db -h localhost
```

### Run Query

```sql
SELECT * FROM financial_concepts;
SELECT COUNT(*) FROM query_logs;
```

### Exit

```
\q
```

---

## Part 12: Useful WSL PostgreSQL Commands

### Start/Stop PostgreSQL Service

```bash
# Start
sudo service postgresql start

# Stop
sudo service postgresql stop

# Restart
sudo service postgresql restart

# Status
sudo service postgresql status
```

### View PostgreSQL Logs

```bash
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

### Backup Database

```bash
pg_dump -U postgres concept_db > concept_db_backup.sql
```

### Restore Database

```bash
psql -U postgres -d concept_db < concept_db_backup.sql
```

### Get WSL IP Address (for .env)

```bash
hostname -I
```

Or more specifically:
```bash
ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
```

---

## Part 13: Troubleshooting

### PostgreSQL Won't Start

```bash
# Check if port 5432 is in use
sudo lsof -i :5432

# Check logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log

# Try restarting
sudo service postgresql restart
```

### Connection Refused

```bash
# Verify service is running
sudo service postgresql status

# Make sure port is listening
sudo netstat -tlnp | grep postgres

# Check pg_hba.conf allows connections
sudo grep "host" /etc/postgresql/14/main/pg_hba.conf
```

### Can't Find Database

```bash
# List all databases
sudo -u postgres psql -l

# Create it if missing
sudo -u postgres createdb concept_db
```

### Permission Denied

```bash
# Make sure you're using sudo for postgres commands
sudo -u postgres psql

# Or grant permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE concept_db TO postgres;"
```

### WSL IP Changes

The WSL IP address can change when you restart. If connection fails:
1. Run `hostname -I` to get new IP
2. Update .env file with new IP
3. Test connection again

---

## Part 14: Permanent Setup Script

Create `setup_pg_wsl.sh` in WSL:

```bash
#!/bin/bash

echo "Setting up PostgreSQL in WSL..."

# Update packages
sudo apt update
sudo apt upgrade -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Start service
sudo service postgresql start

# Create database
sudo -u postgres createdb concept_db

# Create user
sudo -u postgres psql -c "CREATE USER rag_user WITH PASSWORD 'rag_password123';"
sudo -u postgres psql -c "ALTER ROLE rag_user CREATEDB;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE concept_db TO rag_user;"

echo "✓ PostgreSQL setup complete!"
echo "Database: concept_db"
echo "User: rag_user / postgres"
echo ""
echo "To get WSL IP for .env file, run:"
echo "hostname -I"
```

### Run the Setup Script

```bash
chmod +x setup_pg_wsl.sh
./setup_pg_wsl.sh
```

---

## Part 15: Quick Start Checklist

- [ ] WSL installed and updated
- [ ] PostgreSQL installed in WSL
- [ ] PostgreSQL service started
- [ ] `concept_db` database created
- [ ] PostgreSQL configured to accept remote connections
- [ ] WSL IP address obtained
- [ ] .env file updated with WSL IP
- [ ] Python packages installed
- [ ] Test connection script ran successfully
- [ ] Enhanced RAG server started

---

## Part 16: Common Connection Strings

### From Windows to WSL PostgreSQL

```env
DB_HOST=172.31.42.23         # Your WSL IP (changes on restart)
DB_PORT=5432
DB_NAME=concept_db
DB_USER=postgres
DB_PASSWORD=postgres
```

### From WSL to WSL PostgreSQL

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=concept_db
DB_USER=postgres
DB_PASSWORD=postgres
```

### Python Connection Example

```python
import psycopg2

conn = psycopg2.connect(
    host="172.31.42.23",      # WSL IP
    port=5432,
    database="concept_db",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()
cursor.execute("SELECT version();")
print(cursor.fetchone())
conn.close()
```

---

## WSL PostgreSQL vs Windows PostgreSQL

| Feature | WSL PostgreSQL | Windows PostgreSQL |
|---------|----------------|--------------------|
| Installation | In WSL terminal | Windows installer |
| Start Service | `sudo service postgresql start` | Windows Services |
| Access from Windows | Use WSL IP (172.x.x.x) | Use localhost |
| Access from WSL | Use localhost | Use Windows IP |
| Performance | Good for Linux development | Native Windows |
| Recommended For | Linux developers, Docker testing | Windows-native apps |

---

## Next Steps

1. Set up PostgreSQL in WSL following this guide
2. Update your .env file with the WSL IP
3. Run the test connection script
4. Start the enhanced RAG server
5. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is volatility?", "use_vector_db": true}'
```

Enjoy your WSL PostgreSQL setup! 🚀
