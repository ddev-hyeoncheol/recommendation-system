#!/bin/bash
set -e

echo "---------------------------------------------------------"
echo " 🚀 Vespa Deployment Pipeline"
echo "---------------------------------------------------------"

VESPA_SHARED_FS_PATH="/opt/vespa/var/shared"

# ---------------------------------------------------------
# 1. Generate Application Package
# ---------------------------------------------------------
echo -e "\n[Step 1] Generating Application Package..."
docker exec -u vscode:vscode develop python /home/vscode/workspace/vespa/create_package.py

# ---------------------------------------------------------
# 2. Deploy to Config Server
# ---------------------------------------------------------
echo -e "\n[Step 2] Deploying to Config Server..."
docker exec vespa vespa deploy $VESPA_SHARED_FS_PATH/app_package_out

echo -e "\n✅ Deployment Pipeline Finished Successfully!"
