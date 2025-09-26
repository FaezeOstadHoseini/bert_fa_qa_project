#!/bin/bash

# Script for training Persian BERT QA model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_FILE="$PROJECT_DIR/configs/training_config.yaml"
DEBUG=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config CONFIG_FILE    Path to config file (default: configs/training_config.yaml)"
            echo "  --debug                 Enable debug mode"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Activate virtual environment if exists
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "Activated virtual environment"
elif [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    echo "Activated virtual environment"
fi

echo "Starting Persian BERT QA Training..."
echo "Config: $CONFIG_FILE"
echo "Project directory: $PROJECT_DIR"

# Run training
cd "$PROJECT_DIR"
python train.py

echo "Training completed!"
