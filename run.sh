#!/bin/bash
# FastAPI Backend Startup Script (Linux/Mac)

echo "========================================="
echo "HerHealth.AI - FastAPI Backend"
echo "========================================="
echo ""

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start FastAPI server
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo ""

python main.py
