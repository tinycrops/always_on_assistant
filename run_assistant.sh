#!/bin/bash
# Launcher script for Always-On Assistant

echo "Starting Always-On Voice Assistant..."
echo "======================================="
echo ""

# Check if running in conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "📦 Conda environment: $CONDA_DEFAULT_ENV"
fi

# Check API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set!"
    echo "   Set it with: export GEMINI_API_KEY='your-key-here'"
    exit 1
fi

echo "✅ GEMINI_API_KEY found"
echo ""

# Run verification
echo "Running setup verification..."
python verify_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Some checks failed. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 1
    fi
fi

echo ""
echo "======================================="
echo "Starting assistant..."
echo "Press Ctrl+C to stop"
echo "======================================="
echo ""

# Run the assistant
python always_on_assistant.py

