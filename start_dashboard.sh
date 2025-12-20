#!/bin/bash

# Configuration
PORT=8501
HOST=$(hostname)
USER=$(whoami)

echo "=========================================================="
echo "🚀 Creating Streamlit Dashboard..."
echo "To view this dashboard on your LOCAL machine, you must set up a secure tunnel."
echo ""
echo "👇 RUN THIS COMMAND IN A NEW TERMINAL ON YOUR LAPTOP/LOCAL MACHINE:"
echo ""
echo "ssh -L ${PORT}:localhost:${PORT} ${USER}@${HOST}.sol.supercomputer.edu" 
echo "(Note: Replace .sol.supercomputer.edu with the actual full domain of this login node if different)"
echo ""
echo "Then open your browser to: http://localhost:${PORT}"
echo "=========================================================="
echo ""

# Launch Streamlit
# --server.address 0.0.0.0 ensures it binds to all interfaces (required for tunneling sometimes)
# --server.headless true prevents it from trying to pop up a browser window on the server
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
