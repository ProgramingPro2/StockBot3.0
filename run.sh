#!/bin/bash
# Run The Boardroom from project root
cd "$(dirname "$0")"
python -m trading_system "$@"
