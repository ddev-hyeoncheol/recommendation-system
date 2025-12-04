#!/bin/bash
set -e

echo "---------------------------------------------------------"
echo " 🚚 Vespa Data Feeding"
echo "---------------------------------------------------------"

VESPA_SHARED_FS_PATH="/opt/vespa/var/shared"

# ---------------------------------------------------------
# 1. Feed User Data
# ---------------------------------------------------------
echo -e "\n[Step 1] Feeding Users..."
docker exec vespa vespa feed $VESPA_SHARED_FS_PATH/vespa_feed/vespa_user_feed.jsonl

# ---------------------------------------------------------
# 2. Feed Product Data
# ---------------------------------------------------------
echo -e "\n[Step 2] Feeding Products..."
docker exec vespa vespa feed $VESPA_SHARED_FS_PATH/vespa_feed/vespa_product_feed.jsonl

echo -e "\n✅ Data Feeding Complete!"
