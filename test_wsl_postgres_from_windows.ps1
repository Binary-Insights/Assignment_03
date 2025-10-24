# Test PostgreSQL Connection from Windows to WSL
# Run this in PowerShell to verify connectivity

Write-Host "🔍 Testing PostgreSQL Connection from Windows to WSL" -ForegroundColor Cyan
Write-Host ""

# Step 1: Get WSL IP
Write-Host "1️⃣  Getting WSL IP address..." -ForegroundColor Yellow
$wslIp = wsl hostname -I | Select-Object -First 1 | ForEach-Object { $_.Split()[0] }
Write-Host "   WSL IP: $wslIp" -ForegroundColor Green
Write-Host ""

# Step 2: Test port connectivity
Write-Host "2️⃣  Testing port 5432 connectivity..." -ForegroundColor Yellow
$tcpClient = New-Object System.Net.Sockets.TcpClient
try {
    $tcpClient.Connect($wslIp, 5432)
    Write-Host "   ✅ Port 5432 is open and accessible" -ForegroundColor Green
    $tcpClient.Close()
} catch {
    Write-Host "   ❌ Cannot connect to port 5432" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Display connection info
Write-Host "3️⃣  Connection Details for DBeaver:" -ForegroundColor Yellow
Write-Host "   Host:     $wslIp" -ForegroundColor Cyan
Write-Host "   Port:     5432" -ForegroundColor Cyan
Write-Host "   Database: concept_db" -ForegroundColor Cyan
Write-Host "   Username: airflow" -ForegroundColor Cyan
Write-Host "   Password: airflow" -ForegroundColor Cyan
Write-Host ""

# Step 4: Test with psql if installed
Write-Host "4️⃣  Testing with psql (if installed)..." -ForegroundColor Yellow
$psqlPath = Get-Command psql -ErrorAction SilentlyContinue
if ($psqlPath) {
    try {
        $result = psql -h $wslIp -U airflow -d concept_db -c "SELECT 1;" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ psql connection successful!" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  psql connection failed" -ForegroundColor Yellow
            Write-Host "   Response: $result" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⚠️  psql test skipped" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ℹ️  psql not installed (optional)" -ForegroundColor Gray
}
Write-Host ""

# Step 5: Summary
Write-Host "✅ Ready to connect from DBeaver!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Open DBeaver" -ForegroundColor White
Write-Host "2. Create new PostgreSQL connection" -ForegroundColor White
Write-Host "3. Use host: $wslIp" -ForegroundColor White
Write-Host "4. Use credentials above" -ForegroundColor White
Write-Host "5. Click 'Test Connection'" -ForegroundColor White
Write-Host ""
