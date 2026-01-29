#!/bin/bash
# Quick-start visualization generator
# Run this to generate your first RSS map visualizations

set -e

echo "=========================================="
echo "RSS Map Visualization Generator"
echo "=========================================="
echo ""

# Check if running in correct directory
if [ ! -f "visualize_rss_maps.py" ]; then
    echo "ERROR: visualize_rss_maps.py not found"
    echo "Please run this script from the Diffusion project root directory"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; print('  ✓ PyTorch available')" || { echo "  ✗ PyTorch missing"; exit 1; }
python -c "import matplotlib; print('  ✓ Matplotlib available')" || { echo "  ✗ Matplotlib missing"; exit 1; }
python -c "import numpy; print('  ✓ NumPy available')" || { echo "  ✗ NumPy missing"; exit 1; }

echo ""
echo "Available commands:"
echo "  1. Quick test (1 scene, ~30 sec):"
echo "     python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20"
echo ""
echo "  2. Standard (3 scenes, ~2 min):"
echo "     python batch_visualize.py --config standard"
echo ""
echo "  3. All scenes high-quality (5 scenes, ~5 min):"
echo "     python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50"
echo ""
echo "  4. View batch preset options:"
echo "     python batch_visualize.py --list"
echo ""
echo "Running quick test..."
echo "=========================================="
echo ""

# Run quick test
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

echo ""
echo "=========================================="
echo "✓ Visualization complete!"
echo "=========================================="
echo ""
echo "Output files saved to: rss_visualizations/"
echo ""
echo "View the PNG files:"
echo "  - scene0_rss_comparison.png  (6-panel comparison)"
echo "  - scene0_predicted.png       (predicted map)"
echo "  - scene0_groundtruth.png     (reference map)"
echo ""
echo "For more options, see VISUALIZATION_TOOLKIT_GUIDE.md"
echo ""
