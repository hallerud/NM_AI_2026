#!/bin/bash
echo "=== Tripletex AI Agent ==="
echo ""
echo "Step 1: Install dependencies"
pip install -r requirements.txt
echo ""
echo "Step 2: Starting server on port 8000..."
echo "Step 3: In another terminal, run: ngrok http 8000"
echo "Step 4: Copy the ngrok HTTPS URL to the submission form"
echo ""
echo "Make sure ANTHROPIC_API_KEY is set:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000
