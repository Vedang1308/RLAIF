#!/bin/bash

# Configuration
PORT=${1:-8502} # Default to 8502 to avoid collision with other Streamlit apps
HOST=$(hostname)
USER=$(whoami)

echo "=========================================================="
echo "🚀 Creating Streamlit Dashboard..."
echo "To view this dashboard on your LOCAL machine, you must set up a secure tunnel."
echo ""
echo "👇 RUN THIS COMMAND IN A NEW TERMINAL ON YOUR LAPTOP/LOCAL MACHINE:"
echo ""
echo "  ssh -L ${PORT}:localhost:${PORT} ${USER}@${HOST}"
echo ""
echo "(If that doesn't work, replace '${HOST}' with the full address you use to log in, e.g. login.rc.asu.edu)"
echo ""
echo "Then open your browser to: http://localhost:${PORT}"
echo "=========================================================="
echo ""

# Launch Streamlit
# --server.address 0.0.0.0: Bind to all IPs (required for tunnel)
# --server.headless true: Don't open browser on server
# --server.enableCORS false: Allow tunnel traffic (Fixes "Stuck on loading")
# --server.enableXsrfProtection false: Fixes WebSocket issues over tunnel
streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
