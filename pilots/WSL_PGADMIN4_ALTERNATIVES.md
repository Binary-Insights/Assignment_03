# pgAdmin4 Installation Alternatives for WSL

## Problem
pgAdmin4 package is not available in standard Ubuntu repositories for WSL.

## Solution Options

### Option 1: Use Docker (Recommended for WSL)
The cleanest way to run pgAdmin4 is via Docker container:

```bash
# Install Docker in WSL (if not already installed)
sudo apt update
sudo apt install docker.io -y
sudo usermod -aG docker $USER
newgrp docker

# Run pgAdmin4 in Docker
docker run --name pgadmin \
  -e PGADMIN_DEFAULT_EMAIL=admin@example.com \
  -e PGADMIN_DEFAULT_PASSWORD=admin \
  -p 5050:80 \
  -d dpage/pgadmin4:latest
```

Then access it at: http://localhost:5050

**Connection Details in pgAdmin4:**
- Hostname/address: `172.17.0.1` (Docker gateway - reaches host WSL IP)
- Port: `5432`
- Username: `postgres`
- Password: `(your postgres password)`

---

### Option 2: Install from PostgreSQL Repository
Add the official PostgreSQL repository which has pgAdmin4:

```bash
# Add PostgreSQL repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Import PostgreSQL repository key
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Update and install pgAdmin4
sudo apt update
sudo apt install pgadmin4 -y
```

Then access via: http://localhost/pgadmin4

---

### Option 3: Use psql Command Line (Lightweight Alternative)
No installation needed - use PostgreSQL's built-in command-line client:

```bash
# Connect to local PostgreSQL
psql -U postgres -d concept_db

# Useful psql commands:
\dt                    # List tables
\d financial_concepts  # Describe table structure
SELECT * FROM financial_concepts;  # View data
\q                     # Quit

# Or connect from Windows Python directly with psycopg2:
```

**Python example to query from Windows:**
```python
import psycopg2

conn = psycopg2.connect(
    host="172.xx.xx.xx",  # WSL IP
    port=5432,
    database="concept_db",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM financial_concepts;")
for row in cursor.fetchall():
    print(row)

conn.close()
```

---

### Option 4: Use DBeaver (Desktop GUI - Windows)
Install DBeaver on Windows and connect directly to WSL PostgreSQL:

1. Download from: https://dbeaver.io/download/
2. Install on Windows
3. New Database Connection → PostgreSQL
4. **Connection Details:**
   - Server Host: `172.xx.xx.xx` (your WSL IP)
   - Port: `5432`
   - Database: `concept_db`
   - Username: `postgres`
   - Password: `(your password)`
5. Test Connection
6. Browse data graphically

**Advantages:**
- Professional GUI on Windows (no WSL GUI issues)
- Direct connection to WSL PostgreSQL
- SQL query editor with autocomplete
- Data visualization tools

---

### Option 5: VS Code SQL Tools Extension (Lightweight)
Use VS Code extension for database browsing:

1. Install "SQL Tools" extension in VS Code
2. Install "SQLTools PostgreSQL Driver"
3. Create connection:
   - **Connection Type:** PostgreSQL
   - **Server:** `172.xx.xx.xx` (WSL IP)
   - **Port:** `5432`
   - **User:** `postgres`
   - **Password:** `postgres`
   - **Database:** `concept_db`
4. Browse tables and run queries directly in VS Code

---

## Recommendation for Your Setup

**Best Option: Option 3 (psql) + Option 4 (DBeaver)**

1. **For Command-Line Work:** Use `psql` in WSL terminal
   ```bash
   psql -U postgres -d concept_db
   ```

2. **For Visual Browsing:** Install DBeaver on Windows
   - Point to WSL IP: `172.xx.xx.xx:5432`
   - No additional WSL installation needed
   - Professional GUI interface

---

## Verify PostgreSQL is Ready

Before trying any GUI tool, ensure PostgreSQL is running and accessible:

```bash
# In WSL terminal:

# 1. Check if PostgreSQL is running
sudo service postgresql status

# 2. Check if listening on all interfaces
sudo -u postgres psql -c "SHOW listen_addresses;"

# 3. Get your WSL IP address
hostname -I

# 4. Test local connection
psql -U postgres -d concept_db -c "SELECT 1 as connected;"
```

---

## Connection String Reference

**From Windows to WSL PostgreSQL:**
```
psycopg2: postgresql://postgres:postgres@172.xx.xx.xx:5432/concept_db
Connection URL: postgresql://172.xx.xx.xx:5432/concept_db?user=postgres&password=postgres
```

**Replace `172.xx.xx.xx` with your actual WSL IP address.**

---

## Troubleshooting

**Can't find WSL IP?**
```bash
# In WSL terminal
hostname -I
# Example output: 172.31.249.101
```

**PostgreSQL not listening on all interfaces?**
```bash
# In WSL terminal
sudo nano /etc/postgresql/14/main/postgresql.conf

# Find line with "listen_addresses" and ensure it's set to:
# listen_addresses = '*'

# Then restart PostgreSQL:
sudo service postgresql restart
```

**Connection refused?**
```bash
# Ensure pg_hba.conf allows remote connections
sudo nano /etc/postgresql/14/main/pg_hba.conf

# Add these lines if not present:
# TYPE  DATABASE        USER            ADDRESS                 METHOD
# host  all             all             172.0.0.0/8             md5
# host  all             all             0.0.0.0/0               md5

sudo service postgresql restart
```

---

## Next Steps

1. **Verify PostgreSQL is running and accessible** (use psql from WSL)
2. **Choose your GUI tool** (recommend DBeaver for Windows)
3. **Test connection** with your chosen tool
4. **Verify your enhanced RAG system can connect** (run test script)
5. **Start querying!**

The enhanced RAG system doesn't require pgAdmin4 to work - it connects directly via Python psycopg2 driver.
