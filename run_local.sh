#!/bin/bash
# run_local.sh - Robust Local Launcher for RLAIF Dashboard

# 1. Configuration
# Use 0.0.0.0 to allow access from other devices (like phone) or network IPs
ADDRESS="0.0.0.0"
PORT=8502

echo "🚀 Starting RLAIF Dashboard (Safe Mode)..."
echo "If browser doesn't open, go to: http://localhost:$PORT"

# 2. Launch with Security Relaxations
# - enableCORS=false: Allows access from network/external URLs
# - enableXsrfProtection=false: Prevents 'stuck looking for connection' errors
/usr/bin/python3 -m streamlit run app.py \
    --server.port $PORT \
    --server.address $ADDRESS \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
