# HerHealth.AI - FastAPI Backend

A unified Python-based backend using **FastAPI** for cervical cancer screening with AI models.

## Architecture

This backend consolidates what was previously split between Spring Boot and Flask into a single FastAPI application:

```
FastAPI Backend (http://localhost:8000)
├── Classification (ConvNeXt)
├── Segmentation (U-Net)
├── Preprocessing Tools
│   ├── Quality Assessment
│   ├── Stain Normalization
│   └── Multi-Cell Detection
├── Batch Processing
├── Report Generation
└── XAI Visualizations
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Ensure model files are in place:**
```
backend/
├── convnext_best_f1_0.9853.pt   # Classification model
├── best_unet.pth                # Segmentation model
└── main.py
```

## Running the Backend

### Option 1: Direct Python (Development)

```bash
cd backend
python main.py
```

Server starts on `http://localhost:8000`

### Option 2: Using Batch File (Windows)

```bash
start-all.bat
```

This starts the backend along with the frontend.

### Option 3: Using Shell Script (Linux/Mac)

```bash
bash backend/run.sh
```

## API Endpoints

### Health & Info
- `GET /health` - Health check
- `GET /api/v1/classification/classes` - Get available classes

### Classification
- `POST /api/v1/classification/predict` - Classify single image
- `POST /api/v1/segmentation/predict` - Segment and classify

### Preprocessing Tools
- `POST /api/v1/quality-assessment` - Assess image quality
- `POST /api/v1/stain-normalization` - Normalize stain variations
- `POST /api/v1/multi-cell-detect` - Detect multiple cells

### Batch Processing
- `POST /api/v1/batch-process` - Process multiple images

### Report Generation
- `POST /api/v1/generate-report` - Generate analysis report
- `POST /api/v1/export-pdf` - Export report as PDF

## API Documentation

Interactive API documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Models

### ConvNeXt (Classification)
- Architecture: ConvNeXt Tiny
- Input: 224×224 RGB image
- Output: 5-class probabilities (Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate)
- Accuracy: ~98.5%

### U-Net (Segmentation)
- Architecture: Attention U-Net with ResNet18 encoder
- Input: 256×256 RGB image
- Output: 3-class segmentation (Background, Nucleus, Cytoplasm)
- Metrics: F1: 0.92, IoU: 0.85

## Features

### Core Functionality
✓ Image classification with uncertainty quantification
✓ Cell nucleus and cytoplasm segmentation
✓ XAI visualizations (Score-CAM, Layer-CAM heatmaps)
✓ Batch image processing

### Preprocessing Tools
✓ Quality assessment (blur, brightness detection)
✓ Stain normalization (histogram equalization)
✓ Multi-cell detection (contour-based)
✓ Confidence scoring and uncertainty bounds

### Advanced Features
✓ Clinical decision support (risk stratification)
✓ Report generation
✓ CORS enabled for frontend integration

## Configuration

Environment variables:
```bash
PORT=8000                                    # Server port
CONVNEXT_MODEL_PATH=convnext_best_f1_0.9853.pt  # Classification model
UNET_MODEL_PATH=best_unet.pth               # Segmentation model
```

## Performance

- **Device:** Auto-detects GPU (CUDA) or falls back to CPU
- **Inference Time:** ~200-300ms per image on GPU, ~500-800ms on CPU
- **Batch Processing:** Processes multiple images sequentially

## Troubleshooting

### Model Loading Issues
```
WARNING: ConvNeXt weights not found
```
**Solution:** Ensure model files are in the backend directory

### Port Already in Use
```
Address already in use
```
**Solution:** Change port in start-all.bat or use `PORT=9000 python main.py`

### GPU Not Detected
```
Using device: cpu
```
**Solution:** Verify CUDA installation and NVIDIA drivers

### Dependency Issues
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Re-install dependencies: `pip install -r requirements.txt`

## Development

### Modifying Endpoints
Edit `backend/main.py` and restart the server.

### Adding New Preprocessing Tools
1. Create function in `main.py`
2. Add FastAPI route with `@app.post("/api/v1/your-endpoint")`
3. Update frontend API calls in `frontend/lib/api.ts`

### Performance Optimization
- Enable GPU: Install CUDA-enabled PyTorch
- Batch processing: Use `/api/v1/batch-process` for multiple images
- Model optimization: Consider quantization or distillation

## Integration with Frontend

The frontend connects via these environment variables:
```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

All API calls use this base URL.

## Response Format

Standard response format:
```json
{
  "predicted_class": "Koilocytotic",
  "probabilities": {
    "Dyskeratotic": 0.05,
    "Koilocytotic": 0.92,
    ...
  },
  "original_image_base64": "...",
  "segmentation_mask_base64": "...",
  "xai_scorecam_base64": "...",
  "xai_layercam_base64": "...",
  "metrics": { ... },
  "uncertainty": { ... },
  "clinical_decision": { ... },
  "processing_time_ms": 245,
  "model_version": "v2.0.0"
}
```

## License

Proprietary - HerHealth.AI

## Support

For issues or questions, check:
1. API docs: http://localhost:8000/docs
2. Server logs in terminal
3. Frontend console (DevTools)
