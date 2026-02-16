# GPU Setup Instructions

## Current Status
Your system is configured to **auto-detect** GPU availability and use it if CUDA-enabled PyTorch is installed.

## To Enable GPU Acceleration

### 1. Free Up Disk Space
You need at least **3GB free** to install PyTorch with CUDA support.

### 2. Install CUDA-enabled PyTorch

**For CUDA 11.8** (Compatible with most GPUs):
```powershell
D:/Local-ollama/.venv/Scripts/pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1** (Newer GPUs):
```powershell
D:/Local-ollama/.venv/Scripts/pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verify GPU is Detected
```powershell
D:/Local-ollama/.venv/Scripts/python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 4. Restart the Server
```powershell
python server.py
```

You should see: `✅ Embedding model loaded successfully with CUDA`

## Manual Device Selection

If you want to force GPU or CPU, create/edit `.env` file:

```bash
# Force GPU
EMBEDDING_DEVICE=cuda

# Force CPU
EMBEDDING_DEVICE=cpu

# Auto-detect (default)
EMBEDDING_DEVICE=auto
```

## Current Setup (CPU Only)
Right now you're using **CPU** because:
- Current PyTorch: CPU-only version (114MB)
- GPU PyTorch: Requires 2.8GB download

## Performance Comparison

| Device | Speed | Memory |
|--------|-------|--------|
| CPU | Slower | 114MB |
| GPU | **5-10x faster** | 2.8GB |

## Disk Space Tips

To free up space:
1. Delete temporary files: `%TEMP%`
2. Clean browser cache
3. Remove old downloads
4. Run Disk Cleanup
5. Move files to external drive

Once you have space, run the CUDA PyTorch installation command above!
