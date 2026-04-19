#!/bin/bash
# ──────────────────────────────────────────────────────────────
# LungAI — Start Script
# Launches the Flask backend + serves the frontend via HTTP
# ──────────────────────────────────────────────────────────────

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
FRONTEND_PORT=8080
BACKEND_PORT=5000

echo "🫁  LungAI — Starting up..."

# 1. Start Flask backend
cd "$BACKEND_DIR"
if [ ! -d "venv" ]; then
  echo "📦  Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install flask flask-cors pillow numpy
else
  source venv/bin/activate
fi

echo "🚀  Starting Flask backend on http://localhost:$BACKEND_PORT"
python app.py &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"

# Wait for Flask to be ready
sleep 2

# 2. Serve frontend
cd "$FRONTEND_DIR"
echo "🌐  Serving frontend on http://localhost:$FRONTEND_PORT"
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!
echo "    Frontend PID: $FRONTEND_PID"

# 3. Open in browser
sleep 1
open "http://localhost:$FRONTEND_PORT"

echo ""
echo "✅  LungAI is running!"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   Backend:  http://localhost:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop all servers."

# Wait and clean up on exit
trap "echo ''; echo '🛑 Shutting down...' && kill $BACKEND_PID $FRONTEND_PID 2>/dev/null && echo 'Done.'" EXIT
wait
