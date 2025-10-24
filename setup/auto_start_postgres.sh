#!/bin/bash
# Auto-start PostgreSQL daemon on WSL startup
# Add this line to your ~/.bashrc file:
# nohup sudo -u postgres /usr/lib/postgresql/16/bin/postgres -D /var/lib/postgresql/16/main > /tmp/postgres.log 2>&1 &

if ! pgrep -x "postgres" > /dev/null; then
    echo "Starting PostgreSQL daemon..."
    nohup sudo -u postgres /usr/lib/postgresql/16/bin/postgres -D /var/lib/postgresql/16/main > /tmp/postgres.log 2>&1 &
    sleep 2
    if pgrep -x "postgres" > /dev/null; then
        echo "✅ PostgreSQL started successfully"
    else
        echo "❌ Failed to start PostgreSQL"
        cat /tmp/postgres.log
    fi
else
    echo "✅ PostgreSQL is already running"
fi
