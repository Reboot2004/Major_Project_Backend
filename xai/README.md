# XAI Module - Enhanced CAM with Quality Assessment

This module provides comprehensive explainability and quality assessment for cervical cytology images.

## Features Implemented

### 1. **Sample Quality Assessment**
- Blur detection using Laplacian variance
- Contrast measurement
- Saturation analysis
- Brightness validation
- Returns quality score [0-1] and specific flags

### 2. **Stain Normalization** (Macenko Method)
- Normalizes H&E staining across different scanners
- Ensures consistent color appearance
- Uses optical density transformation
- Optional - enable with `--enable-stain-norm`

### 3. **Multi-Cell Detection**
- Adaptive thresholding
- Morphological operations
- Distance transform for separation
- Returns bounding boxes and cell count

### 4. **Uncertainty Estimation** (Monte Carlo Dropout)
- Runs multiple forward passes with dropout enabled
- Computes prediction confidence
- Calculates entropy and variance
- Provides reliability metrics

### 5. **Score-CAM** (Optimized)
- Top-K channel selection (96 channels)
- Batched forward passes (batch size 64)
- FP16 inference on GPU
- ~3-5x faster than standard implementation

### 6. **Layer-CAM**
- Gradient-weighted activation maps
- Element-wise multiplication with gradients
- ReLU applied to gradients

## Installation

```powershell
cd backend/xai
pip install -r requirements.txt
```

## Usage

### Basic (CAM only)
```powershell
python compute_cam.py --image path/to/image.jpg --class 0
```

### With All Features
```powershell
python compute_cam.py --image path/to/image.jpg --class 0 \
    --enable-quality \
    --enable-stain-norm \
    --enable-multi-cell \
    --enable-uncertainty
```

### Output Format
```json
{
    "quality": {
        "score": 0.85,
        "flags": {
            "is_blurry": false,
            "low_contrast": false,
            "overexposed": false,
            "underexposed": false
        }
    },
    "cells": {
        "count": 3,
        "bounding_boxes": [
            {"x": 120, "y": 80, "width": 50, "height": 60, "area": 2500}
        ]
    },
    "uncertainty": {
        "confidence": 0.87,
        "entropy": 0.34,
        "std": 0.02,
        "prediction_variance": 0.05
    },
    "scorecam_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "layercam_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "normalized_image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Performance

- **Score-CAM**: ~1-2s on GPU (with optimizations)
- **Layer-CAM**: ~0.3-0.5s on GPU
- **Quality Assessment**: ~0.05s
- **Multi-Cell Detection**: ~0.1-0.2s
- **Uncertainty Estimation**: ~1-2s (20 samples)

## Integration with Java Backend

The Java `ModelService` automatically calls this script with enhanced features enabled:
- Quality assessment: ON
- Multi-cell detection: ON
- Uncertainty estimation: ON
- Stain normalization: OFF (optional, enable if needed)

## GPU Support

- Automatically uses CUDA if available
- Falls back to CPU if no GPU detected
- FP16 precision on GPU for faster inference
- torch.compile() optimization if PyTorch 2.0+

## Troubleshooting

### OpenCV Import Error
```powershell
pip install opencv-python-headless
```

### CUDA Out of Memory
Reduce batch size in `score_cam_fast()`:
```python
score_cam_fast(model, x, args.cls, device=device, top_k=64, batch=32)
```

### Stain Normalization Fails
Image might have insufficient pixels or unusual staining. The script automatically falls back to original image.

## References

- Score-CAM: Wang et al. (2020)
- Layer-CAM: Jiang et al. (2021)
- Macenko Normalization: Macenko et al. (2009)
