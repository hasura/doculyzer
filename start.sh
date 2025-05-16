#!/bin/bash

# Path to the project root directory
PROJECT_DIR=$(dirname "$(readlink -f "$0")")

# Install required packages if not already installed
if ! pip list | grep -q "gunicorn"; then
    echo "Installing Gunicorn and Gevent..."
    pip install gunicorn gevent
fi

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
export LOG_LEVEL=INFO

# Increase system limits for the process
# This allows more open files and connections
ulimit -n 65536

# Start Gunicorn with optimal settings
echo "Starting Doculyzer API server with Gunicorn..."
gunicorn \
    --workers=4 \
    --worker-class=gevent \
    --worker-connections=1000 \
    --timeout=60 \
    --max-requests=1000 \
    --max-requests-jitter=50 \
    --backlog=2048 \
    --bind=0.0.0.0:5000 \
    --access-logfile=- \
    --error-logfile=- \
    --log-level=info \
    --keep-alive=5 \
    --graceful-timeout=10 \
    doculyzer.server:app
