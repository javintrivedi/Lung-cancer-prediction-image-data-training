#!/bin/bash
# ──────────────────────────────────────────────────────────────
# LungAI — Start Script
# Launches the Flask backend + serves the frontend via HTTP
# ──────────────────────────────────────────────────────────────

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
FRONTEND_PORT=8080
BACKEND_PORT=5001

echo "🫁  LungAI — Starting up..."

# 1. Start Flask backend
cd "$BACKEND_DIR"

# Check if venv is broken or missing
if [ ! -d "venv" ] || [ ! -f "venv/bin/python3" ]; then
  echo "📦  Recreating virtual environment..."
  rm -rf venv
  python3 -m venv venv
  source venv/bin/activate
  echo "pip install -r requirements.txt"
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

echo "🚀  Starting Flask backend on http://localhost:$BACKEND_PORT"
python3 app.py &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"

# Wait for Flask to be ready
sleep 3

# 2. Serve frontend
cd "$FRONTEND_DIR"
echo "🌐  Serving frontend on http://localhost:$FRONTEND_PORT"
# Use a simple check to see if port is in use
if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port $FRONTEND_PORT is already in use. Skipping frontend server."
    FRONTEND_PID=""
else
    python3 -m http.server $FRONTEND_PORT &
    FRONTEND_PID=$!
    echo "    Frontend PID: $FRONTEND_PID"
fi

# 3. Open in browser (if on Mac)
if [[ "$OSTYPE" == "darwin"* ]]; then
  sleep 1
  open "http://localhost:$FRONTEND_PORT"
fi

echo ""
echo "✅  LungAI is running!"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   Backend:  http://localhost:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop all servers."

# Wait and clean up on exit
trap "echo ''; echo '🛑 Shutting down...' && kill $BACKEND_PID $FRONTEND_PID 2>/dev/null && echo 'Done.'" EXIT
wait
