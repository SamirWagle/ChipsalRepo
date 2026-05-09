#!/bin/bash
# Download pre-trained model weights and checkpoints
#
# Usage:
#   bash download_models.sh [--all] [--model MODEL_NAME]
#
# Options:
#   --all          Download all models (M1-M8)
#   --model NAME   Download specific model (e.g., m7_hybrid_fusion)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base URL for model weights (update with actual hosting location)
BASE_URL="https://example.com/chipsal-models"  # PLACEHOLDER: Update with actual URL

# Output directory
OUTPUT_DIR="models/pretrained"

# Available models
declare -A MODELS
MODELS["m1_text_only"]="m1_text_only_best.pt"
MODELS["m2_image_only"]="m2_image_only_best.pt"
MODELS["m3_image_text_removed"]="m3_image_text_removed_best.pt"
MODELS["m4_early_fusion"]="m4_early_fusion_best.pt"
MODELS["m5_late_voting"]="m5_late_voting_best.pt"
MODELS["m6_late_bagging"]="m6_late_bagging_best.pt"
MODELS["m7_hybrid_fusion"]="m7_hybrid_fusion_best.pt"
MODELS["m8_hybrid_text_removed"]="m8_hybrid_text_removed_best.pt"

# Model sizes (approximate, in MB)
declare -A MODEL_SIZES
MODEL_SIZES["m1_text_only"]=420
MODEL_SIZES["m2_image_only"]=350
MODEL_SIZES["m3_image_text_removed"]=350
MODEL_SIZES["m4_early_fusion"]=780
MODEL_SIZES["m5_late_voting"]=770
MODEL_SIZES["m6_late_bagging"]=2300
MODEL_SIZES["m7_hybrid_fusion"]=780
MODEL_SIZES["m8_hybrid_text_removed"]=780

# Parse arguments
DOWNLOAD_ALL=false
SPECIFIC_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash download_models.sh [--all] [--model MODEL_NAME]"
            echo ""
            echo "Options:"
            echo "  --all          Download all models (M1-M8)"
            echo "  --model NAME   Download specific model"
            echo ""
            echo "Available models:"
            for model in "${!MODELS[@]}"; do
                echo "  - $model (${MODEL_SIZES[$model]} MB)"
            done
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}ChiPSAL Model Downloader${NC}"
echo "======================================"

# Function to download a single model
download_model() {
    local model_name=$1
    local filename=${MODELS[$model_name]}
    local size=${MODEL_SIZES[$model_name]}
    
    echo -e "${YELLOW}Downloading $model_name (${size} MB)...${NC}"
    
    local url="${BASE_URL}/${filename}"
    local output_path="${OUTPUT_DIR}/${filename}"
    
    # Check if already downloaded
    if [ -f "$output_path" ]; then
        echo -e "${GREEN}✓ $model_name already downloaded${NC}"
        return 0
    fi
    
    # Download with wget or curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$url" -O "$output_path"
    elif command -v curl &> /dev/null; then
        curl -# -L "$url" -o "$output_path"
    else
        echo -e "${RED}Error: Neither wget nor curl found${NC}"
        exit 1
    fi
    
    # Verify download
    if [ -f "$output_path" ]; then
        echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Failed to download $model_name${NC}"
        return 1
    fi
}

# Download models
if [ "$DOWNLOAD_ALL" = true ]; then
    echo "Downloading all models..."
    echo ""
    
    total_size=0
    for model in "${!MODELS[@]}"; do
        total_size=$((total_size + MODEL_SIZES[$model]))
    done
    
    echo -e "${YELLOW}Total download size: ${total_size} MB${NC}"
    echo ""
    
    for model in "${!MODELS[@]}"; do
        download_model "$model"
        echo ""
    done
    
elif [ -n "$SPECIFIC_MODEL" ]; then
    if [ -z "${MODELS[$SPECIFIC_MODEL]}" ]; then
        echo -e "${RED}Error: Unknown model '$SPECIFIC_MODEL'${NC}"
        echo "Available models:"
        for model in "${!MODELS[@]}"; do
            echo "  - $model"
        done
        exit 1
    fi
    
    download_model "$SPECIFIC_MODEL"
    
else
    echo -e "${YELLOW}No download option specified${NC}"
    echo "Use --all to download all models or --model NAME for specific model"
    echo "Run with --help for more information"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================"
echo "Download complete!${NC}"
echo ""
echo "Models saved to: $OUTPUT_DIR"
echo ""
echo "To use these models:"
echo "  python abiral/inference.py --model $OUTPUT_DIR/m7_hybrid_fusion_best.pt --image path/to/image.jpg --text 'your text'"
echo ""

# Note about model hosting
echo -e "${YELLOW}Note:${NC} This script downloads from: $BASE_URL"
echo "Update BASE_URL variable if models are hosted elsewhere"
echo ""
echo "Alternative: Download manually from:"
echo "  - GitHub Releases: https://github.com/yourusername/chipsal/releases"
echo "  - Hugging Face Hub: https://huggingface.co/yourusername/chipsal-models"
echo "  - Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX"
