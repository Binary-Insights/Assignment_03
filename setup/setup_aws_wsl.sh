#!/bin/bash
# AWS Configuration Setup for WSL
# This script helps set up AWS CLI and credentials

set -e

echo "🔧 AWS Configuration Setup for WSL"
echo "===================================="
echo ""

# Step 1: Install AWS CLI
echo "1️⃣  Installing AWS CLI v2..."
if command -v aws &> /dev/null; then
    echo "   ✅ AWS CLI already installed"
    aws --version
else
    sudo apt update
    sudo apt install -y awscli
    echo "   ✅ AWS CLI installed"
fi
echo ""

# Step 2: Check/Create AWS credentials directory
echo "2️⃣  Setting up AWS credentials directory..."
mkdir -p ~/.aws
chmod 700 ~/.aws
echo "   ✅ ~/.aws directory ready"
echo ""

# Step 3: Test existing credentials
echo "3️⃣  Testing AWS credentials..."
if [ -f ~/.aws/credentials ]; then
    echo "   ✅ ~/.aws/credentials found"
    if aws sts get-caller-identity &> /dev/null; then
        echo "   ✅ Credentials are valid"
        aws sts get-caller-identity
    else
        echo "   ❌ Credentials invalid or expired"
        echo "   Run: aws configure"
    fi
else
    echo "   ⚠️  ~/.aws/credentials not found"
    echo "   Run: aws configure"
fi
echo ""

# Step 4: Check boto3
echo "4️⃣  Checking boto3 (Python AWS SDK)..."
if python3 -c "import boto3" 2> /dev/null; then
    echo "   ✅ boto3 is installed"
    python3 -c "import boto3; print('   Version:', boto3.__version__)"
else
    echo "   ⚠️  boto3 not installed"
    echo "   Install with: pip install boto3"
fi
echo ""

# Step 5: Display configuration
echo "5️⃣  Current AWS Configuration:"
if [ -f ~/.aws/config ]; then
    echo "   Config file exists: ~/.aws/config"
    grep -E "^\[|region" ~/.aws/config 2>/dev/null || echo "   (empty or not configured)"
fi
if [ -f ~/.aws/credentials ]; then
    echo "   Credentials file exists: ~/.aws/credentials (protected)"
fi
echo ""

# Step 6: Quick commands reference
echo "6️⃣  Quick Commands:"
echo "   • Configure credentials: aws configure"
echo "   • List S3 buckets: aws s3 ls"
echo "   • Test credentials: aws sts get-caller-identity"
echo "   • View config: aws configure list"
echo ""

echo "✅ AWS setup verification complete!"
echo ""
echo "Next steps:"
echo "1. If credentials not configured, run: aws configure"
echo "2. Provide your AWS Access Key ID and Secret Access Key"
echo "3. Set default region (e.g., us-east-1)"
echo "4. Test with: aws s3 ls"
echo ""
