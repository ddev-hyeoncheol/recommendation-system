#!/bin/bash
set -e

echo "---------------------------------------------------------"
echo " 🚚 Vespa Data Feeding Pipeline"
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

FEED_DATA_DIR=${FEED_DATA_DIR:-/home/vscode/workspace/data/feed_data}

# ---------------------------------------------------------
# 2. Set Vespa Configuration
# ---------------------------------------------------------
echo -e "\n[Step 2] Setting Vespa Configuration..."
vespa config set target http://vespa:8080

echo -e "\n✅ Vespa Configuration set successfully."

# ---------------------------------------------------------
# 3. Feed User Data
# ---------------------------------------------------------
echo -e "\n[Step 3] Feeding User Data..."
vespa feed $FEED_DATA_DIR/vespa_user_feed.jsonl
vespa feed $FEED_DATA_DIR/vespa_user_vector_feed.jsonl
# vespa feed $FEED_DATA_DIR/vespa_user_cold_start_feed.jsonl

echo -e "\n✅ Fed User & User Vector & User Cold Start successfully."

# ---------------------------------------------------------
# 4. Feed Product Data
# ---------------------------------------------------------
echo -e "\n[Step 4] Feeding Product Data..."
vespa feed $FEED_DATA_DIR/vespa_product_feed.jsonl
vespa feed $FEED_DATA_DIR/vespa_product_vector_feed.jsonl
vespa feed $FEED_DATA_DIR/vespa_product_cold_start_feed.jsonl

echo -e "\n✅ Fed Product & Product Vector & Product Cold Start successfully."

echo -e "\n✅ Data Feeding Pipeline finished successfully!"