# Installation

TorchWM supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
# Core dependencies (torch, torchvision, torchaudio, gym, gymnasium, etc.)
pip install torchwm

# With specific extras
pip install torchwm[gym]       # Additional gym environments (huggingface-hub, pygame, autorom)
pip install torchwm[ml-agents] # Unity ML-Agents support
pip install torchwm[ml]        # TensorBoard, Weights & Biases, logging tools
pip install torchwm[viz]       # FastAPI, Uvicorn, documentation tools
pip install torchwm[docs]      # Sphinx and documentation tools
pip install torchwm[dev]       # Testing and development tools (pytest, mypy, pre-commit)

# Install multiple extras
pip install torchwm[gym,ml-agents,dev]
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `gym` | Additional Gym environment dependencies (huggingface-hub, pygame, autorom) |
| `ml-agents` | Unity ML-Agents support |
| `ml` | TensorBoard, Weights & Biases, and logging tools |
| `viz` | FastAPI, Uvicorn for visualization server |
| `docs` | Sphinx and documentation building tools |
| `dev` | pytest, mypy, pre-commit for development |

### Core Dependencies

The minimal installation includes: torch, torchvision, torchaudio, einops, pyyaml, tqdm, opencv-python, requests, gym, gymnasium, moviepy, h5py, plotly, ale-py, selenium, scikit-learn, umap-learn.

## From Source

For the latest development version:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm

# Core dependencies
pip install -e .

# With extras
pip install -e ".[gym,ml-agents,ml,viz,dev,docs]"
```

## PyTorch Build Selection

TorchWM does not pin a single PyTorch wheel index. Install the PyTorch build that matches your platform (CPU, macOS, CUDA, ROCm, etc.) using the index recommended by the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
# Example: CUDA 12.1 wheels. Replace the index for CPU, ROCm, or other CUDA versions.
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121

# Or using pip.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Docker

Build the default CPU image and run the TorchWM CLI:

```bash
# Build
docker build -t torchwm .

# Show the CLI help
docker run --rm torchwm

# Run a specific command
docker run --rm torchwm models list
```

The Dockerfile installs PyTorch explicitly before installing TorchWM so the wheel
source is controlled by the `PYTORCH_INDEX_URL` build argument. The default uses
CPU wheels. To build against a CUDA wheel index, pass the matching PyTorch index
and run the container with the NVIDIA runtime:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t torchwm:cu121 .

docker run --rm --gpus all torchwm:cu121 models list
```

Additional optional dependency groups can be installed at build time with
`TORCHWM_EXTRAS`, for example `--build-arg TORCHWM_EXTRAS=viz,ml`. Runtime data is
stored under `/data/torchwm`, which you can persist with a bind mount or volume.

## Verification

Verify your installation:

```python
import torch
import torchwm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("TorchWM imported successfully!")
```