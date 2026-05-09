# Troubleshooting Guide

Common issues and solutions for the CHiPSAL multimodal hate detection system.

## Table of Contents

- [Installation Issues](#installation-issues)
- [CUDA / GPU Issues](#cuda--gpu-issues)
- [Memory Issues](#memory-issues)
- [Training Issues](#training-issues)
- [Data Loading Issues](#data-loading-issues)
- [Model Loading Issues](#model-loading-issues)
- [Inference Issues](#inference-issues)
- [Performance Issues](#performance-issues)

---

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install transformers==4.30.2
```

If still failing:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

### Issue: `torch.cuda.is_available()` returns `False`

**Solution:**

1. **Check CUDA installation:**
```bash
nvidia-smi
```

2. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify CUDA version:**
```python
import torch
print(torch.version.cuda)  # Should match your CUDA version
```

---

### Issue: `ImportError: libcudnn.so.8: cannot open shared object file`

**Solution (Linux):**
```bash
# Install cuDNN
conda install -c conda-forge cudnn

# Or add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Solution (Windows):**
- Download cuDNN from NVIDIA website
- Extract and copy files to CUDA installation directory

---

## CUDA / GPU Issues

### Issue: `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
```python
# In config or command line
--batch_size 16  # Instead of 32
```

2. **Enable gradient checkpointing:**
```python
# In model config
model.gradient_checkpointing_enable()
```

3. **Clear CUDA cache:**
```python
import torch
torch.cuda.empty_cache()
```

4. **Use mixed precision training:**
```python
# Already enabled by default
trainer = Trainer(..., use_amp=True)
```

5. **Monitor GPU memory:**
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
```

---

### Issue: `RuntimeError: Expected all tensors to be on the same device`

**Solution:**

Ensure all tensors are on GPU:
```python
# Move model to GPU
model = model.to('cuda')

# Move data to GPU
images = images.to('cuda')
input_ids = input_ids.to('cuda')
attention_mask = attention_mask.to('cuda')
labels = labels.to('cuda')
```

Or specify device in config:
```python
trainer = Trainer(..., device='cuda')
```

---

## Memory Issues

### Issue: `RuntimeError: DataLoader worker (pid XXXX) is killed by signal: Killed`

**Cause:** System runs out of RAM during data loading.

**Solutions:**

1. **Reduce num_workers:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,  # Reduce from 4 or 8
    pin_memory=True
)
```

2. **Disable pin_memory:**
```python
train_loader = DataLoader(..., pin_memory=False)
```

3. **Reduce image resolution:**
```python
# In dataset transform
Resize(size=(224, 224))  # Already default for CLIP
```

---

### Issue: Training crashes with `SIGKILL`

**Solutions:**

1. **Check system memory:**
```bash
free -h  # Linux
# Ensure sufficient free RAM (16GB+ recommended)
```

2. **Enable swap space:**
```bash
# Linux: Create 8GB swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **Reduce dataset in-memory caching:**
```python
# Disable caching if enabled
dataset = MultimodalDataset(..., cache_in_memory=False)
```

---

## Training Issues

### Issue: Loss becomes `nan` or `inf`

**Solutions:**

1. **Reduce learning rate:**
```python
--learning_rate 1e-5  # Instead of 2e-5
```

2. **Enable gradient clipping:**
```python
# In trainer
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **Check for invalid labels:**
```python
# Ensure labels are 0 or 1
assert set(labels.unique().tolist()) <= {0, 1}
```

4. **Disable mixed precision temporarily:**
```python
trainer = Trainer(..., use_amp=False)
```

---

### Issue: Model not improving (val F1 stuck)

**Solutions:**

1. **Check learning rate:**
```python
# May be too high or too low
--learning_rate 2e-5  # Recommended
```

2. **Increase patience:**
```python
--early_stopping_patience 15  # Instead of 10
```

3. **Verify class weights:**
```python
from abiral.utils import compute_class_weights
weights = compute_class_weights(labels, gamma=0.3)
print(weights)  # Should not be extreme (e.g., [0.01, 100])
```

4. **Check for data leakage:**
```python
# Ensure validation set is separate
assert len(set(train_indices) & set(val_indices)) == 0
```

5. **Try different random seed:**
```bash
--seed 123  # Try multiple seeds
```

---

### Issue: `KeyError: 'label'` during training

**Solution:**

Check CSV column names:
```python
import pandas as pd
df = pd.read_csv('train.csv')
print(df.columns)  # Should include 'label'

# If different, specify in dataset
dataset = MultimodalDataset(
    ...,
    label_column='target'  # Or whatever your column is named
)
```

---

## Data Loading Issues

### Issue: `FileNotFoundError: [Errno 2] No such file or directory: 'train/train_images/12345.jpg'`

**Solutions:**

1. **Check file paths:**
```python
import os
assert os.path.exists('train/train_images/12345.jpg')
```

2. **Verify CSV has correct filenames:**
```python
import pandas as pd
df = pd.read_csv('train.csv')
print(df['index'].head())  # Should match actual filenames
```

3. **Use absolute paths:**
```python
import os
image_dir = os.path.abspath('train/train_images/')
dataset = MultimodalDataset(..., image_dir=image_dir)
```

---

### Issue: `OSError: image file is truncated`

**Cause:** Corrupted or incomplete image file.

**Solution:**

1. **Skip corrupted images:**
```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

2. **Find corrupted images:**
```python
from PIL import Image
import os

for img_file in os.listdir('train/train_images/'):
    try:
        img = Image.open(f'train/train_images/{img_file}')
        img.verify()
    except Exception as e:
        print(f"Corrupted: {img_file}")
```

3. **Re-download dataset:**
```bash
# Download fresh copy from CHiPSAL 2026
```

---

### Issue: Images not loading or all black

**Solution:**

Check image transformations:
```python
from PIL import Image
import matplotlib.pyplot as plt

# Load and display
img = Image.open('train/train_images/12345.jpg')
plt.imshow(img)
plt.show()

# Check mode
print(img.mode)  # Should be 'RGB'
if img.mode != 'RGB':
    img = img.convert('RGB')
```

---

## Model Loading Issues

### Issue: `RuntimeError: Error(s) in loading state_dict`

**Cause:** Model architecture changed or checkpoint incompatible.

**Solutions:**

1. **Load with strict=False:**
```python
model.load_state_dict(
    torch.load('model.pt', map_location='cuda'),
    strict=False
)
```

2. **Check model config matches:**
```python
# Ensure same config used for training and loading
config = {
    'model_type': 'hybrid',
    'hidden_dim': 256,
    'attention_heads': 4,
    ...
}
model = MultimodalClassifier(**config)
model.load_state_dict(torch.load('model.pt'))
```

3. **Re-train model if architecture changed**

---

### Issue: `ModuleNotFoundError: No module named 'abiral'`

**Solution:**

Install package in editable mode:
```bash
cd ChipsalRepo/
pip install -e .
```

Or add to Python path:
```python
import sys
sys.path.append('/path/to/ChipsalRepo')
from abiral.model import MultimodalClassifier
```

---

## Inference Issues

### Issue: Predictions all the same class

**Solutions:**

1. **Check model is in eval mode:**
```python
model.eval()
with torch.no_grad():
    predictions = model(...)
```

2. **Verify model loaded correctly:**
```python
# Check model weights
print(list(model.parameters())[0])  # Should not be all zeros
```

3. **Check class imbalance:**
```python
# Model may be predicting majority class
# Ensure using probability threshold or F1-optimized threshold
```

4. **Try different checkpoint:**
```python
# Load best_model.pt instead of last_epoch.pt
model.load_state_dict(torch.load('best_model.pt'))
```

---

### Issue: `ValueError: Expected input batch_size (64) to match target batch_size (32)`

**Solution:**

Ensure batch sizes match:
```python
# In inference
predictions = model.predict(
    images=images[:32],       # Match batch size
    input_ids=input_ids[:32],
    attention_mask=attention_mask[:32]
)
```

---

## Performance Issues

### Issue: Training is very slow

**Solutions:**

1. **Enable mixed precision:**
```python
trainer = Trainer(..., use_amp=True)  # 2x speedup
```

2. **Increase num_workers:**
```python
train_loader = DataLoader(..., num_workers=8)
```

3. **Enable pin_memory:**
```python
train_loader = DataLoader(..., pin_memory=True)
```

4. **Use persistent workers:**
```python
train_loader = DataLoader(..., persistent_workers=True)
```

5. **Profile bottlenecks:**
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Training loop
    for batch in train_loader:
        ...

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

### Issue: Inference is slow

**Solutions:**

1. **Use batch inference:**
```python
# Instead of one-by-one
results = detector.predict_batch(images, texts, batch_size=32)
```

2. **Convert to TorchScript:**
```python
model.eval()
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_scripted.pt')
```

3. **Use ONNX Runtime:**
```python
# Export to ONNX
torch.onnx.export(model, ...)

# Load with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
```

4. **Quantize model (CPU):**
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Advanced Issues

### Issue: Distributed training fails

**Solution:**

Use `torch.distributed.launch`:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    abiral/train.py \
    --distributed
```

---

### Issue: Wandb logging fails

**Solutions:**

1. **Login to Wandb:**
```bash
wandb login
```

2. **Disable Wandb:**
```python
import os
os.environ['WANDB_MODE'] = 'disabled'
```

3. **Use offline mode:**
```bash
export WANDB_MODE=offline
```

---

### Issue: Results not reproducible

**Solutions:**

1. **Set all random seeds:**
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

2. **Use same hardware:**
- Different GPUs may have slightly different results
- Use exact same CUDA/cuDNN versions

3. **Disable data augmentation during testing:**
```python
test_dataset = MultimodalDataset(..., transform=None)
```

---

## Getting Help

If your issue is not covered here:

1. **Check existing GitHub issues:** [Link to issues page]
2. **Search discussions:** [Link to discussions]
3. **Open a new issue:** [Link to new issue]
   - Include full error traceback
   - Specify environment (OS, Python version, CUDA version, GPU model)
   - Provide minimal reproducible example

4. **Contact:** samir@example.com

---

## Debugging Tips

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check tensor shapes

```python
def debug_forward(model, batch):
    print(f"Images shape: {batch['image'].shape}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    
    with torch.no_grad():
        outputs = model(
            batch['image'],
            batch['input_ids'],
            batch['attention_mask']
        )
    print(f"Outputs shape: {outputs.shape}")
    return outputs
```

### Profiling memory

```python
import tracemalloc

tracemalloc.start()
# Your code here
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

---

## Common Error Messages

### `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`

**Solution:**
```python
torch.backends.cudnn.enabled = False
```

### `RuntimeError: CUDA error: device-side assert triggered`

**Solution:**
```python
# Run on CPU to see actual error
model = model.cpu()
# Then fix the underlying issue (usually invalid labels/indices)
```

### `OSError: [Errno 28] No space left on device`

**Solution:**
```bash
# Free up disk space
# Clear cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Check disk usage
df -h
```

---

## Performance Benchmarks

Expected performance on standard hardware:

### Training Time (per epoch)

| Model | RTX 3090 | RTX 4090 | A100 |
|-------|----------|----------|------|
| M1 (Text-Only) | ~3.6 min | ~2.5 min | ~2.0 min |
| M7 (Hybrid) | ~10.4 min | ~7.2 min | ~5.5 min |

### Inference Speed

| Batch Size | RTX 3090 | RTX 4090 | CPU (32 cores) |
|------------|----------|----------|----------------|
| 1 | 45 ms | 32 ms | 180 ms |
| 32 | 850 ms | 600 ms | 6.2 s |
| 64 | 1.6 s | 1.1 s | 12.5 s |

If your performance is significantly worse, check for:
- Mixed precision disabled
- Inefficient data loading
- CPU bottlenecks
- Thermal throttling

---

**Last Updated:** March 1, 2026
