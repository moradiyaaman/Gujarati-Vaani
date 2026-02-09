#!/bin/bash
# ============================================================================
# GUJARATI VAANI - AZURE ML SETUP SCRIPT
# ============================================================================
#
# This script prepares the Azure ML Compute Instance for training.
# Target: Standard_NC6 (Tesla T4 GPU, 16GB VRAM)
#
# Usage:
#   chmod +x azure_setup.sh
#   ./azure_setup.sh
#
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "GUJARATI VAANI - Azure ML Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# 1. SYSTEM INFO
# ============================================================================
echo -e "\n${YELLOW}[1/7] System Information${NC}"

echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}Warning: nvidia-smi not found. GPU might not be available.${NC}"
fi

# ============================================================================
# 2. CREATE DIRECTORY STRUCTURE
# ============================================================================
echo -e "\n${YELLOW}[2/7] Creating Directory Structure${NC}"

# These paths match the README.md structure
mkdir -p model_weights/finetuned
mkdir -p model_weights/original
mkdir -p model_weights/tokenizer
mkdir -p samples
mkdir -p logs
mkdir -p data

echo "Created directories:"
echo "  ├── model_weights/"
echo "  │   ├── finetuned/    (training output)"
echo "  │   ├── original/     (base model backup)"
echo "  │   └── tokenizer/    (tokenizer files)"
echo "  ├── samples/          (validation audio samples)"
echo "  ├── logs/             (TensorBoard logs)"
echo "  └── data/             (local data cache)"

# ============================================================================
# 3. INSTALL DEPENDENCIES
# ============================================================================
echo -e "\n${YELLOW}[3/7] Installing Python Dependencies${NC}"

pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install training requirements
pip install -r training/requirements_train.txt

echo -e "${GREEN}Dependencies installed successfully!${NC}"

# ============================================================================
# 4. VERIFY GPU ACCESS
# ============================================================================
echo -e "\n${YELLOW}[4/7] Verifying GPU Access${NC}"

python << 'EOF'
import torch
import sys

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    print("Please ensure you're running on a GPU compute instance.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"✓ GPU: {gpu_name}")
print(f"✓ VRAM: {gpu_memory:.1f} GB")
print(f"✓ CUDA Version: {torch.version.cuda}")
print(f"✓ PyTorch Version: {torch.__version__}")

# Quick CUDA test
x = torch.randn(100, 100).cuda()
y = x @ x.T
print(f"✓ CUDA tensor operations working!")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}GPU verification failed!${NC}"
    exit 1
fi

echo -e "${GREEN}GPU verification passed!${NC}"

# ============================================================================
# 5. DOWNLOAD BASE MODEL (Optional - for offline backup)
# ============================================================================
echo -e "\n${YELLOW}[5/7] Downloading Base Model for Offline Backup${NC}"

python << 'EOF'
from transformers import VitsModel, AutoTokenizer
from pathlib import Path

model_id = "facebook/mms-tts-guj"
original_dir = Path("model_weights/original")
tokenizer_dir = Path("model_weights/tokenizer")

print(f"Downloading {model_id}...")

# Download and save model
model = VitsModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(original_dir)
tokenizer.save_pretrained(tokenizer_dir)

print(f"✓ Model saved to: {original_dir}")
print(f"✓ Tokenizer saved to: {tokenizer_dir}")
EOF

echo -e "${GREEN}Base model downloaded!${NC}"

# ============================================================================
# 6. MOUNT DATA STORAGE (Azure Blob)
# ============================================================================
echo -e "\n${YELLOW}[6/7] Data Storage Configuration${NC}"

# Check if data is already mounted (Azure ML auto-mounts datasets)
if [ -d "/data/indic_tts/guj" ]; then
    echo -e "${GREEN}✓ Data storage already mounted at /data/indic_tts/guj${NC}"
    
    # Verify dataset structure
    echo "Dataset structure:"
    ls -la /data/indic_tts/guj/ 2>/dev/null || echo "  (checking...)"
    
    if [ -f "/data/indic_tts/guj/train/metadata.csv" ]; then
        TRAIN_COUNT=$(wc -l < /data/indic_tts/guj/train/metadata.csv)
        echo "  ├── train/metadata.csv: ${TRAIN_COUNT} samples"
    fi
    
    if [ -f "/data/indic_tts/guj/valid/metadata.csv" ]; then
        VALID_COUNT=$(wc -l < /data/indic_tts/guj/valid/metadata.csv)
        echo "  └── valid/metadata.csv: ${VALID_COUNT} samples"
    fi
else
    echo -e "${YELLOW}Data not mounted. Configure Azure Blob Storage:${NC}"
    echo ""
    echo "Option 1: Mount via Azure ML Studio"
    echo "  1. Go to Azure ML Studio > Data"
    echo "  2. Create a new Datastore pointing to your Blob container"
    echo "  3. Mount it to /data/indic_tts/guj in your compute instance"
    echo ""
    echo "Option 2: Use azcopy (manual download)"
    echo "  azcopy copy 'https://<storage>.blob.core.windows.net/indic-tts/guj/*' ./data/guj/ --recursive"
    echo ""
    echo "Then update train.py: --data-dir ./data/guj"
fi

# ============================================================================
# 7. VERIFY INSTALLATION
# ============================================================================
echo -e "\n${YELLOW}[7/7] Final Verification${NC}"

python << 'EOF'
import sys
from pathlib import Path

# Add parent for utils import
sys.path.insert(0, str(Path(__file__).parent))

print("Checking all imports...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers: {e}")

try:
    import librosa
    print(f"✓ librosa {librosa.__version__}")
except ImportError as e:
    print(f"✗ librosa: {e}")

try:
    import soundfile
    print(f"✓ soundfile")
except ImportError as e:
    print(f"✗ soundfile: {e}")

try:
    from utils.text_utils import normalize_text, filter_gujarati_text
    print(f"✓ utils.text_utils (normalize_text, filter_gujarati_text)")
except ImportError as e:
    print(f"✗ utils.text_utils: {e}")

try:
    from torch.utils.tensorboard import SummaryWriter
    print(f"✓ tensorboard")
except ImportError as e:
    print(f"✗ tensorboard: {e}")

print("\n✓ All imports successful!")
EOF

echo ""
echo "=============================================="
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "=============================================="
echo ""
echo "To start training, run:"
echo ""
echo "  python training/train.py \\"
echo "    --data-dir /data/indic_tts/guj \\"
echo "    --output-dir ./model_weights/finetuned \\"
echo "    --epochs 50 \\"
echo "    --batch-size 4"
echo ""
echo "To monitor training with TensorBoard:"
echo ""
echo "  tensorboard --logdir ./logs --port 6006"
echo ""
echo "=============================================="
