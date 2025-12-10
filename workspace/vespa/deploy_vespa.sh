#!/bin/bash
set -e

echo "---------------------------------------------------------"
echo " 🚀 Vespa Application Package Deployment"
echo "---------------------------------------------------------"

# ---------------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------------
ENV_PATH="/home/vscode/workspace/.env"

echo -e "\n[Step 1] Environment Setup..."
if [ -f "$ENV_PATH" ]; then
    echo -e "\n✅ Environment file found. Loading environment variables..."
    export $(grep -v '^#' "$ENV_PATH" | xargs)
else
    echo -e "\n❌ Environment file not found."
    exit 1
fi

APP_PACKAGE_DIR=${APP_PACKAGE_DIR:-/home/vscode/workspace/vespa/app_package_out}
SCRIPT_DIR=$(dirname "$0")

# ---------------------------------------------------------
# 2. Set Vespa Configuration
# ---------------------------------------------------------
echo -e "\n[Step 2] Setting Vespa Configuration..."
vespa config set target http://vespa:19071

echo -e "\n✅ Vespa Configuration set successfully."

# ---------------------------------------------------------
# 3. Generate Application Package
# ---------------------------------------------------------
echo -e "\n[Step 3] Generating Application Package..."
python $SCRIPT_DIR/create_package.py

echo -e "\n✅ Application Package generated successfully."

# ---------------------------------------------------------
# 4. Deploy to Config Server
# ---------------------------------------------------------
echo -e "\n[Step 4] Deploying to Config Server..."
vespa deploy $APP_PACKAGE_DIR

echo -e "\n✅ Deployed to Config Server successfully."

# ---------------------------------------------------------
# 5. Reset Vespa Configuration
# ---------------------------------------------------------
echo -e "\n[Step 5] Resetting Vespa Configuration..."
vespa config set target http://vespa:8080

echo -e "\n✅ Vespa Configuration reset successfully."

echo -e "\n✅ Deployment Pipeline finished successfully!"