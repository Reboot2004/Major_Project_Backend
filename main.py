"""
HerHealth.AI - FastAPI Backend
Unified Python backend for cervical cancer screening
"""
import os
import io
import base64
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
from starlette.concurrency import run_in_threadpool

# MongoDB
from pymongo import MongoClient
from bson import ObjectId

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.graphics import renderPM

try:
    from svglib.svglib import svg2rlg
except ImportError:
    svg2rlg = None

# Import robust preprocessing and CAM functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xai'))
from compute_cam import (
    assess_image_quality,
    normalize_stain_macenko,
    detect_multiple_cells,
    estimate_uncertainty,
    layer_cam_convnext,
    score_cam_fast,
    cam_to_base64
)

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[BACKEND] Using device: {DEVICE}')

CLASSES = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate'
]

# Model paths
CONVNEXT_PATH = os.environ.get('CONVNEXT_MODEL_PATH', 'convnext_best_f1_0.9853.pt')
UNET_PATH = os.environ.get('UNET_MODEL_PATH', 'best_unet.pth')

# Create FastAPI app
app = FastAPI(
    title="HerHealth.AI Backend",
    description="Cervical Cancer Screening with AI",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATABASE (MongoDB)
# ============================================================================

# Allow env override; default points directly at the 'herhealth' database
MONGO_URI = os.environ.get(
    'MONGO_URI',
    'mongodb+srv://knssriharshith:c3VY2EmB8RRUghE6@cluster0.lz5ln4u.mongodb.net/herhealth?retryWrites=true&w=majority&appName=Cluster0'
)
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'herhealth')

mongo_client: Optional[MongoClient] = None
db = None
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    print(f"[BACKEND] ✓ Connected to MongoDB (db='{MONGO_DB_NAME}')")
except Exception as e:
    print(f"[BACKEND] ✗ WARNING: MongoDB connection failed: {e}")

# ============================================================================
# MODEL LOADING
# ============================================================================

def build_convnext(num_classes=5):
    """Build ConvNeXt model for classification"""
    model = torchvision.models.convnext_tiny(pretrained=False)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def build_unet():
    """Build Attention U-Net for segmentation"""
    return smp.Unet(
        encoder_name='resnet18',
        encoder_weights=None,
        in_channels=3,
        classes=3,
        decoder_attention_type='scse'
    )

print('[BACKEND] Loading ConvNeXt model...')
convnext_model = build_convnext(num_classes=5).to(DEVICE)
convnext_candidates = [
    CONVNEXT_PATH,
    os.path.join(os.path.dirname(__file__), '..', 'convnext_best_f1_0.9853.pt'),
    os.path.join(os.path.dirname(__file__), '..', 'inference-service', 'convnext_best_f1_0.9853.pt')
]
convnext_loaded = False
for path in convnext_candidates:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        convnext_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f'[BACKEND] ✓ ConvNeXt loaded from {path}')
        convnext_loaded = True
        break
if not convnext_loaded:
    print(f"[BACKEND] ✗ WARNING: ConvNeXt weights not found. Tried: {convnext_candidates}")

print('[BACKEND] Loading U-Net model...')
unet_model = build_unet().to(DEVICE)
unet_candidates = [
    UNET_PATH,
    os.path.join(os.path.dirname(__file__), '..', 'best_unet.pth'),
    os.path.join(os.path.dirname(__file__), '..', 'inference-service', 'best_unet.pth')
]
unet_loaded = False
for path in unet_candidates:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                unet_model.load_state_dict(ckpt['model'])
            elif 'model_state_dict' in ckpt:
                unet_model.load_state_dict(ckpt['model_state_dict'])
            else:
                unet_model.load_state_dict(ckpt)
        else:
            unet_model.load_state_dict(ckpt)
        print(f'[BACKEND] ✓ U-Net loaded from {path}')
        unet_loaded = True
        break
if not unet_loaded:
    print(f"[BACKEND] ✗ WARNING: U-Net weights not found. Tried: {unet_candidates}")

convnext_model.eval()
unet_model.eval()

# Expose load status for downstream logic
CONVNEXT_LOADED = convnext_loaded
UNET_LOADED = unet_loaded

# ============================================================================
# PREPROCESSING
# ============================================================================

cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

seg_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def image_to_base64(img: np.ndarray) -> str:
    """Convert image array to base64 string"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def softmax(x):
    """Compute softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ============================================================================
# PROPER IMAGE NORMALIZATION
# ============================================================================

# ============================================================================
# IMAGE PREPROCESSING PIPELINE
# ============================================================================

# In-memory storage for stain-normalized images (temporary)
stain_normalized_cache = {}

def _detect_cells_from_unet(image_np: np.ndarray, min_nucleus_area: int = 500) -> List[Dict]:
    """Estimate cell instances from U-Net nucleus mask via connected components.

    This is more stable than raw threshold/watershed on some images and avoids
    counting tiny artifacts as "cells".
    """
    if not UNET_LOADED:
        return []

    aug = seg_transform(image=image_np)
    img_tensor_seg = aug['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask_logits = unet_model(img_tensor_seg)
        pred_mask = mask_logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    pred_mask_resized = cv2.resize(
        pred_mask,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    nucleus = (pred_mask_resized == 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_OPEN, kernel, iterations=1)
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(nucleus, connectivity=8)
    cells = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_nucleus_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cells.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'area': area,
            'source': 'unet_nucleus'
        })

    return cells

def preprocess_image_pipeline(image_np: np.ndarray, session_id: str = None):
    """
    Complete preprocessing pipeline following clinical workflow:
    1. Quality Assessment (pre-check)
    2. Stain Normalization (Macenko method)
    3. Multi-cell Detection (on normalized image)
    
    Returns: dict with all preprocessing results
    """
    results = {}
    
    # Step 1: Quality Assessment on original image
    quality_score, quality_flags, quality_metrics = assess_image_quality(image_np)
    results['quality'] = {
        'score': float(quality_score),
        'flags': quality_flags,
        'metrics': quality_metrics
    }
    
    # Step 2: Stain Normalization (always apply for consistency)
    stain_normalized = normalize_stain_macenko(image_np)
    results['stain_normalized'] = stain_normalized
    
    # Cache normalized image for download if session_id provided
    if session_id:
        stain_normalized_cache[session_id] = stain_normalized
    
    # Step 3: Multi-cell detection on normalized image
    # Prefer UNet-based nucleus instances when available; fall back to watershed detector.
    cells_method = 'unet' if UNET_LOADED else 'watershed'
    cells = _detect_cells_from_unet(stain_normalized) if UNET_LOADED else []
    if not cells:
        cells_method = 'watershed'
        cells = detect_multiple_cells(
            stain_normalized,
            min_cell_size=800,
            max_cell_size_ratio=0.20,
            circularity_threshold=0.50,
            overlap_threshold=0.50
        )
    results['cells'] = cells
    results['cells_method'] = cells_method
    
    return results

# ============================================================================
# SEGMENTATION METRICS (DYNAMIC CALCULATION)
# ============================================================================

def calculate_iou(pred_mask: np.ndarray, class_idx: int) -> float:
    """Calculate IoU for a specific class (would need ground truth for real metric)"""
    # For now, calculate based on mask distribution
    class_pixels = np.sum(pred_mask == class_idx)
    total_pixels = pred_mask.size
    
    if class_pixels == 0:
        return 0.0
    
    # Calculate a confidence-based metric
    coverage = class_pixels / total_pixels
    # Penalize if coverage is too low or too high
    if coverage < 0.05:
        iou = 0.0
    elif coverage > 0.95:
        iou = 0.3
    else:
        iou = min(coverage * 1.5, 0.95)
    
    return float(iou)

def calculate_metrics(pred_mask: np.ndarray, image: np.ndarray = None) -> dict:
    """Calculate essential segmentation metrics (removed IOU/DSC/F1)"""
    try:
        h, w = pred_mask.shape[:2]
        total_pixels = h * w
        
        # 1. Coverage Ratio (% of image segmented)
        segmented_pixels = (pred_mask > 0).sum()
        coverage = float(segmented_pixels / total_pixels) if total_pixels > 0 else 0.0
        
        # 2. Cell count from nucleus mask (prevents huge counts from tiny artifacts)
        nucleus = (pred_mask == 1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_OPEN, kernel, iterations=1)
        nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, kernel, iterations=2)

        min_nucleus_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(nucleus, connectivity=8)
        nucleus_areas = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area >= min_nucleus_area:
                nucleus_areas.append(area)

        num_cells = int(len(nucleus_areas))

        # 3. Average nucleus size (pixels)
        avg_cell_size = float(np.mean(nucleus_areas)) if nucleus_areas else 0.0
        
        # 4. Edge Density (complexity of segmentation)
        edges = cv2.Canny((pred_mask > 0).astype(np.uint8) * 255, 50, 150)
        edge_density = float(edges.sum() / total_pixels) if total_pixels > 0 else 0.0
        
        # 5. Solidity (how solid/filled are the segments)
        contours, _ = cv2.findContours((pred_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidities = []
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    solidities.append(solidity)
        
        avg_solidity = float(np.mean(solidities)) if solidities else 0.5
        
        return {
            'coverage_ratio': coverage,
            'num_cells': int(num_cells),
            'avg_cell_size': avg_cell_size,
            'edge_density': edge_density,
            'avg_solidity': avg_solidity,
            'accuracy': coverage  # Use coverage as accuracy proxy
        }
    except Exception as e:
        print(f"[BACKEND] Metrics calculation error: {e}")
        return {
            'coverage_ratio': 0.0,
            'num_cells': 0,
            'avg_cell_size': 0.0,
            'edge_density': 0.0,
            'avg_solidity': 0.5,
            'accuracy': 0.0
        }

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": True
    }

# ============================================================================
# HISTORICAL PREDICTIONS (MongoDB)
# ============================================================================

def _format_prediction_doc(doc: dict) -> dict:
    ts = doc.get("timestamp")
    if isinstance(ts, datetime):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = str(ts) if ts is not None else None

    return {
        "id": str(doc.get("_id")) if doc.get("_id") is not None else None,
        "patient_id": doc.get("patient_id"),
        "patient_name": doc.get("patient_name"),
        "kind": doc.get("kind"),
        "model": doc.get("model"),
        "xaiMethod": doc.get("xaiMethod"),
        "magnification": doc.get("magnification"),
        "classification": doc.get("classification"),
        "probabilities": doc.get("probabilities"),
        "quality": doc.get("quality"),
        "uncertainty": doc.get("uncertainty"),
        # Some earlier records may have stored uncertainty metrics under a
        # different key; merge if present so the frontend always sees them.
        "uncertainty_metrics": doc.get("uncertainty_metrics"),
        "clinical_decision": doc.get("clinical_decision"),
        "metrics": doc.get("metrics"),
        "images": doc.get("images"),
        "timestamp": ts_str,
    }


def _build_analysis_payload_from_record(record: dict) -> dict:
    """Normalize a MongoDB prediction document into the analysis payload
    expected by the PDF generator.

    This helper makes the historical PDF routes robust to older documents
    where some fields may be missing or stored under slightly different keys.
    """

    images = record.get("images", {}) or {}

    # Prefer explicit segmentation mask if present; fall back to any
    # older segmentation/overlay fields for backward compatibility.
    seg_mask = (
        images.get("segmentation_mask")
        or images.get("segmentation")
        or images.get("segmentation_overlay")
    )

    return {
        "predicted_class": record.get("classification"),
        "probabilities": record.get("probabilities", {}),
        "original_image_base64": images.get("original"),
        "xai_scorecam_base64": images.get("scorecam"),
        "xai_layercam_base64": images.get("layercam"),
        "segmentation_mask_base64": seg_mask,
        "uncertainty": record.get("uncertainty")
        or record.get("uncertainty_metrics", {}),
        "metrics": record.get("metrics", {}),
        "clinical_decision": record.get("clinical_decision", {}),
        "quality": record.get("quality", {}),
    }


async def _store_prediction(doc: dict):
    """Insert a prediction document into MongoDB, if available.

    This is best-effort only and should never break the main inference flow.
    """
    if db is None:
        return

    def _insert():
        db.predictions.insert_one(doc)

    try:
        await run_in_threadpool(_insert)
    except Exception as e:
        print(f"[BACKEND] Warning: failed to store prediction in MongoDB: {e}")

@app.get("/api/oldpreds")
async def list_old_predictions():
    """Fetch all previous predictions sorted by latest timestamp.

    If MongoDB is not available, this will gracefully return an empty list
    instead of a 500 error so the frontend can still render the page.
    """
    try:
        if db is None:
            print("[BACKEND] /api/oldpreds requested but MongoDB is not initialized; returning empty list.")
            return JSONResponse([])

        def _fetch():
            # Sort by the indexed _id field (which embeds creation time)
            # and cap the result size to avoid large in-memory sorts.
            return list(
                db.predictions
                .find()
                .sort("_id", -1)
                .limit(500)
            )

        preds = await run_in_threadpool(_fetch)
        result = [_format_prediction_doc(p) for p in preds]
        return JSONResponse(result)
    except Exception as e:
        print(f"[BACKEND] /api/oldpreds error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/oldpreds/{id}")
async def get_old_prediction(id: str):
    """Fetch a previous prediction by its ObjectId."""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database is not initialized")

        try:
            oid = ObjectId(id)
        except Exception:
            return JSONResponse({"error": "Invalid id format"}, status_code=400)

        def _fetch_one():
            return db.predictions.find_one({"_id": oid})

        record = await run_in_threadpool(_fetch_one)
        if not record:
            return JSONResponse({"error": "Record not found"}, status_code=404)

        return JSONResponse(_format_prediction_doc(record))
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/oldpreds/{id}/report")
async def download_old_prediction_report(id: str):
    """Generate and download a detailed PDF report for a stored prediction.

    This uses the same PDF generator as the live analysis endpoint, but
    reconstructs the analysis payload from the MongoDB document so the
    frontend can download a full report with a single click.
    """
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database is not initialized")

        try:
            oid = ObjectId(id)
        except Exception:
            return JSONResponse({"error": "Invalid id format"}, status_code=400)

        def _fetch_one():
            return db.predictions.find_one({"_id": oid})

        record = await run_in_threadpool(_fetch_one)
        if not record:
            return JSONResponse({"error": "Record not found"}, status_code=404)

        # Normalize record into the analysis payload expected by generate_pdf_report
        analysis_data = _build_analysis_payload_from_record(record)

        patient_metadata = {
            "patient_id": record.get("patient_id"),
            "patient_name": record.get("patient_name", "Anonymous"),
            "analysis_date": record.get("timestamp", datetime.utcnow()).strftime("%Y-%m-%d")
            if isinstance(record.get("timestamp"), datetime)
            else datetime.utcnow().strftime("%Y-%m-%d"),
        }

        # Dummy file object used only for filename metadata in the PDF
        class _DummyFile:
            filename = "stored_image.png"

        pdf_buffer = generate_pdf_report(analysis_data, patient_metadata, _DummyFile())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_part = (
            str(patient_metadata.get("patient_id", "unknown")).replace(" ", "_")
            if patient_metadata.get("patient_id")
            else "anonymous"
        )
        filename = f"herhealth_analysis_{patient_part}_{timestamp}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[BACKEND] Historical PDF generation error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================================================
# CLASSIFICATION ENDPOINTS
# ============================================================================

@app.get("/api/v1/classification/classes")
async def get_classes():
    """Get available classification classes"""
    return CLASSES

@app.post("/api/v1/classification/predict")
async def classify(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form(None),
):
    """Classify with complete preprocessing pipeline and robust XAI"""
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        # Read image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        
        print(f"[BACKEND] Starting preprocessing pipeline for classification...")
        
        # Complete preprocessing pipeline
        preprocessing = preprocess_image_pipeline(img_np, session_id)
        stain_normalized = preprocessing['stain_normalized']
        
        print(f"[BACKEND] Preprocessing complete. Quality: {preprocessing['quality']['score']:.2f}, Cells: {len(preprocessing['cells'])}")
        
        # Store original image
        original_b64 = image_to_base64(img_np)
        
        # Classification on stain-normalized image
        img_pil = Image.fromarray(stain_normalized)
        img_tensor = cls_transform(img_pil).to(DEVICE)
        
        with torch.no_grad():
            logits = convnext_model(img_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
        
        print(f"[BACKEND] Classification: {CLASSES[pred_class]} ({probs[pred_class]:.2%})")
        
        # Generate proper CAM heatmaps using robust implementations
        try:
            print(f"[BACKEND] Generating Score-CAM...")
            scorecam = score_cam_fast(convnext_model, img_tensor, pred_class, 
                                     layer_name='features.7', device=str(DEVICE), 
                                     top_k=96, batch_size=32)
            scorecam_b64 = cam_to_base64(scorecam)
        except Exception as cam_error:
            print(f"[BACKEND] Score-CAM failed: {cam_error}, using fallback")
            scorecam_b64 = None
        
        try:
            print(f"[BACKEND] Generating Layer-CAM...")
            layercam = layer_cam_convnext(convnext_model, img_tensor, pred_class,
                                         layer_name='features.7', device=str(DEVICE))
            layercam_b64 = cam_to_base64(layercam)
        except Exception as cam_error:
            print(f"[BACKEND] Layer-CAM failed: {cam_error}")
            layercam_b64 = scorecam_b64
        
        # Use Layer-CAM as fallback if Score-CAM failed
        if not scorecam_b64:
            scorecam_b64 = layercam_b64
        
        # Generate segmentation mask (no overlay) for comprehensive analysis
        segmentation_b64 = None
        seg_metrics = {}
        if UNET_LOADED:
            try:
                print(f"[BACKEND] Generating segmentation mask...")
                aug = seg_transform(image=stain_normalized)
                img_tensor_seg = aug['image'].unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    mask_logits = unet_model(img_tensor_seg)
                    pred_mask = mask_logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

                # Resize mask to original image size
                pred_mask_resized = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (stain_normalized.shape[1], stain_normalized.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Colorize mask (pure mask image, no blending with stain-normalized image)
                color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
                mask_colored = color_map[pred_mask_resized]
                segmentation_b64 = image_to_base64(mask_colored)

                # Calculate segmentation metrics
                seg_metrics = calculate_metrics(pred_mask_resized)
                print(f"[BACKEND] Segmentation generated: {seg_metrics['num_cells']} cells detected")

            except Exception as seg_error:
                print(f"[BACKEND] Segmentation generation failed: {seg_error}")
                segmentation_b64 = None
                seg_metrics = {}
        
        # Robust uncertainty estimation with MC Dropout
        try:
            print(f"[BACKEND] Estimating uncertainty...")
            uncertainty_metrics = estimate_uncertainty(convnext_model, img_tensor, pred_class, n_samples=20)
        except Exception as unc_error:
            print(f"[BACKEND] Uncertainty estimation failed: {unc_error}, using basic metrics")
            pred_prob = float(probs[pred_class])
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
            max_entropy = float(np.log(len(CLASSES)))
            confidence_score = 1.0 - (entropy / max_entropy)
            
            uncertainty_metrics = {
                'confidence': pred_prob,
                'entropy': entropy,
                'confidence_score': confidence_score,
                'uncertainty_lower': max(0.0, pred_prob - 0.1),
                'uncertainty_upper': min(1.0, pred_prob + 0.1),
                'prediction_stability': int(50 + confidence_score * 50),
                'is_reliable': confidence_score > 0.7
            }
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        print(f"[BACKEND] Classification complete in {processing_time_ms}ms")

        probabilities_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        # Generate clinical decision support
        pred_prob = float(probs[pred_class])
        risk_mapping = {
            'Dyskeratotic': 'high',
            'Koilocytotic': 'high',
            'Metaplastic': 'moderate',
            'Parabasal': 'low',
            'Superficial-Intermediate': 'low'
        }
        risk_level = risk_mapping.get(CLASSES[pred_class], 'moderate')
        
        # Risk score combines prediction confidence with segmentation quality if available
        seg_confidence = seg_metrics.get('accuracy', 1.0) if seg_metrics else 1.0
        risk_score = int(pred_prob * seg_confidence * 100)
        
        # Generate recommendations based on risk level and metrics
        recommendations = []
        if pred_prob < 0.75:
            recommendations.append("Consider additional testing or expert review due to moderate confidence.")
        if seg_metrics.get('coverage_ratio', 1.0) < 0.3:
            recommendations.append("Image quality or cell coverage may affect accuracy - consider retake.")
        if risk_level == 'high':
            recommendations.append("Schedule urgent clinical follow-up and additional cytological assessment.")
        elif risk_level == 'moderate':
            recommendations.append("Schedule routine follow-up within 6-12 months.")
        if not recommendations:
            recommendations.append("Continue routine screening as per clinical guidelines.")
        
        clinical_decision = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'primary_class': CLASSES[pred_class],
            'secondary_candidates': [
                {'class': CLASSES[i], 'probability': float(probs[i])}
                for i in range(len(CLASSES)) if i != pred_class
            ][:3],
            'recommendations': recommendations,
            'needs_review': bool(pred_prob < 0.75 or seg_metrics.get('accuracy', 1.0) < 0.75),
            'review_reason': 'Low confidence or poor segmentation quality' if pred_prob < 0.75 or seg_metrics.get('accuracy', 1.0) < 0.75 else None
        }

        # Best-effort: store prediction in MongoDB (including segmentation mask if available)
        doc = {
            'kind': 'classification',
            'model': 'ConvNeXt-Tiny',
            'xaiMethod': 'Score-CAM + Layer-CAM',
            'magnification': None,
            'classification': CLASSES[pred_class],
            'probabilities': probabilities_dict,
            'quality': preprocessing['quality'],
            'uncertainty': uncertainty_metrics,
            'images': {
                'original': original_b64,
                'segmentation': segmentation_b64,
                'scorecam': scorecam_b64,
                'layercam': layercam_b64,
            },
            'timestamp': datetime.utcnow(),
        }
        doc['patient_id'] = patient_id
        doc['patient_name'] = patient_name
        await _store_prediction(doc)

        return JSONResponse({
            'predicted_class': CLASSES[pred_class],
            'probabilities': probabilities_dict,
            'xai_scorecam_base64': scorecam_b64,
            'xai_layercam_base64': layercam_b64,
            'original_image_base64': original_b64,
            'segmentation_mask_base64': segmentation_b64,
            'metrics': seg_metrics,
            'uncertainty': uncertainty_metrics,
            'clinical_decision': clinical_decision,
            'preprocessing': {
                'quality_score': preprocessing['quality']['score'],
                'quality_flags': preprocessing['quality']['flags'],
                'cells_detected': len(preprocessing['cells']),
                'stain_normalized_available': True,
                'session_id': session_id
            },
            'processing_time_ms': processing_time_ms,
            'model_version': 'v2.0.0'
        })
    except Exception as e:
        print(f'[BACKEND] Classification error: {str(e)}')
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SEGMENTATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/segmentation/predict")
async def segment(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form(None),
):
    """Segment and classify a single image with preprocessing pipeline and proper metrics"""
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        # Read image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        
        print(f"[BACKEND] Starting preprocessing pipeline for segmentation...")
        
        # Complete preprocessing pipeline
        preprocessing = preprocess_image_pipeline(img_np, session_id)
        stain_normalized = preprocessing['stain_normalized']
        
        print(f"[BACKEND] Preprocessing complete. Quality: {preprocessing['quality']['score']:.2f}, Cells: {len(preprocessing['cells'])}")
        
        # Store original image
        original_b64 = image_to_base64(img_np)
        
        # Classification on stain-normalized image
        img_pil = Image.fromarray(stain_normalized)
        img_tensor_cls = cls_transform(img_pil).to(DEVICE)
        with torch.no_grad():
            logits = convnext_model(img_tensor_cls.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
        
        # Segmentation on stain-normalized image
        aug = seg_transform(image=stain_normalized)
        img_tensor_seg = aug['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_logits = unet_model(img_tensor_seg)
            pred_mask = mask_logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Resize mask to original image size
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (stain_normalized.shape[1], stain_normalized.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Colorize mask with proper visualization (pure mask, no overlay)
        color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        mask_colored = color_map[pred_mask_resized]
        mask_b64 = image_to_base64(mask_colored)
        
        # Calculate dynamic metrics
        metrics = calculate_metrics(pred_mask_resized)
        
        # Generate proper CAM heatmaps
        try:
            print(f"[BACKEND] Generating Score-CAM for segmentation...")
            scorecam = score_cam_fast(convnext_model, img_tensor_cls, pred_class,
                                     layer_name='features.7', device=str(DEVICE),
                                     top_k=96, batch_size=32)
            scorecam_b64 = cam_to_base64(scorecam)
        except Exception as cam_error:
            print(f"[BACKEND] Score-CAM failed in segmentation: {cam_error}")
            scorecam_b64 = None
        
        try:
            print(f"[BACKEND] Generating Layer-CAM for segmentation...")
            layercam = layer_cam_convnext(convnext_model, img_tensor_cls, pred_class,
                                         layer_name='features.7', device=str(DEVICE))
            layercam_b64 = cam_to_base64(layercam)
        except Exception as cam_error:
            print(f"[BACKEND] Layer-CAM failed in segmentation: {cam_error}")
            layercam_b64 = scorecam_b64
        
        # Use Layer-CAM as fallback if Score-CAM failed
        if not scorecam_b64:
            scorecam_b64 = layercam_b64
        
        # Calculate dynamic uncertainty bounds
        pred_prob = float(probs[pred_class])
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        max_entropy = float(np.log(len(CLASSES)))
        confidence_score = 1.0 - (entropy / max_entropy)
        
        # Combine classification and segmentation confidence for overall stability
        seg_confidence = metrics['accuracy']
        combined_confidence = (confidence_score + seg_confidence) / 2
        prediction_stability = int(50 + combined_confidence * 50)
        
        # Dynamic uncertainty margin
        uncertainty_margin = 0.15 * (1 - confidence_score)
        uncertainty_lower = max(0.0, pred_prob - uncertainty_margin)
        uncertainty_upper = min(1.0, pred_prob + uncertainty_margin)
        
        # Clinical decision based on actual metrics
        risk_mapping = {
            'Dyskeratotic': 'high',
            'Koilocytotic': 'high',
            'Metaplastic': 'moderate',
            'Parabasal': 'low',
            'Superficial-Intermediate': 'low'
        }
        risk_level = risk_mapping.get(CLASSES[pred_class], 'moderate')
        
        # Risk score combines prediction confidence with segmentation quality
        risk_score = int(pred_prob * metrics['accuracy'] * 100)
        
        # Generate recommendations based on metrics
        recommendations = []
        if pred_prob < 0.75:
            recommendations.append('Expert review recommended - Low confidence prediction')
        if metrics['coverage_ratio'] < 0.3:
            recommendations.append('Poor segmentation coverage - Consider re-scan')
        if metrics['avg_solidity'] < 0.4:
            recommendations.append('Segmentation quality unclear - Expert review recommended')
        if risk_level == 'high':
            recommendations.append('Follow-up cytology recommended')
        if not recommendations:
            recommendations.append('Routine screening sufficient')
        
        processing_time_ms = int((time.time() - start_time) * 1000)

        probabilities_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        # Best-effort: store combined segmentation + classification prediction
        seg_uncertainty = {
            'confidence': float(pred_prob * 100),
            'entropy': entropy,
            'confidence_score': confidence_score,
            'segmentation_confidence': seg_confidence,
            'combined_confidence': combined_confidence,
            'uncertainty_lower': uncertainty_lower,
            'uncertainty_upper': uncertainty_upper,
            'prediction_stability': prediction_stability
        }
        clinical_decision = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'primary_class': CLASSES[pred_class],
            'secondary_candidates': [
                {'class': CLASSES[i], 'probability': float(probs[i])}
                for i in range(len(CLASSES)) if i != pred_class
            ][:3],
            'recommendations': recommendations,
            'needs_review': bool(pred_prob < 0.75 or metrics['accuracy'] < 0.75),
            'review_reason': 'Low confidence or poor segmentation quality' if pred_prob < 0.75 or metrics['accuracy'] < 0.75 else None
        }

        doc = {
            'kind': 'segmentation',
            'model': 'ConvNeXt-Tiny + U-Net',
            'xaiMethod': 'Score-CAM + Layer-CAM',
            'magnification': None,
            'classification': CLASSES[pred_class],
            'probabilities': probabilities_dict,
            'quality': preprocessing['quality'],
            'metrics': metrics,
            'uncertainty': seg_uncertainty,
            'clinical_decision': clinical_decision,
            'images': {
                'original': original_b64,
                'segmentation': mask_b64,
                'scorecam': scorecam_b64,
                'layercam': layercam_b64,
            },
            'timestamp': datetime.utcnow(),
        }
        doc['patient_id'] = patient_id
        doc['patient_name'] = patient_name
        await _store_prediction(doc)

        return JSONResponse({
            'predicted_class': CLASSES[pred_class],
            'probabilities': probabilities_dict,
            'segmentation_mask_base64': mask_b64,
            'xai_scorecam_base64': scorecam_b64,
            'xai_layercam_base64': layercam_b64,
            'original_image_base64': original_b64,
            'preprocessing': {
                'quality_score': preprocessing['quality']['score'],
                'quality_flags': preprocessing['quality']['flags'],
                'cells_detected': len(preprocessing['cells']),
                'stain_normalized_available': True,
                'session_id': session_id
            },
            'metrics': metrics,
            'uncertainty': seg_uncertainty,
            'clinical_decision': clinical_decision,
            'processing_time_ms': processing_time_ms,
            'model_version': 'v2.0.0'
        })
    except Exception as e:
        print(f'[BACKEND] Segmentation error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PREPROCESSING ENDPOINTS
# ============================================================================

# ============================================================================
# PREPROCESSING ENDPOINTS (Updated to use robust pipeline)
# ============================================================================

@app.post("/api/v1/quality-assessment")
async def assess_quality(file: UploadFile = File(...)):
    """Robust image quality assessment"""
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        
        # Use robust quality assessment from compute_cam
        quality_score, quality_flags, quality_metrics = assess_image_quality(img_np)
        
        return JSONResponse({
            'quality': {
                'quality_score': float(quality_score * 100),  # Convert to percentage
                'quality_level': 'excellent' if quality_score >= 0.8 else 'good' if quality_score >= 0.6 else 'fair' if quality_score >= 0.4 else 'poor',
                'flags': quality_flags,
                'detailed_metrics': quality_metrics,
                'issues': [k for k, v in quality_flags.items() if v and k not in ['usable']],
                'recommendations': ['Image suitable for analysis'] if quality_flags.get('usable', False) else ['Consider re-imaging for better quality']
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stain-normalization")
async def normalize_stain(file: UploadFile = File(...)):
    """Apply Macenko stain normalization"""
    try:
        session_id = str(uuid.uuid4())
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img).astype(np.uint8)
        
        # Apply Macenko stain normalization
        normalized = normalize_stain_macenko(img_np)
        normalized_b64 = image_to_base64(normalized)
        
        # Cache for download
        stain_normalized_cache[session_id] = normalized
        
        return JSONResponse({
            'normalized_image_base64': normalized_b64,
            'session_id': session_id,
            'download_available': True
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/multi-cell-detection")
async def detect_cells(file: UploadFile = File(...)):
    """Robust multi-cell detection with watershed segmentation"""
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        
        # Apply stain normalization first for better detection
        img_normalized = normalize_stain_macenko(img_np)
        
        # Prefer UNet-based detection; fall back to watershed
        raw_cells = _detect_cells_from_unet(img_normalized) if UNET_LOADED else []
        if not raw_cells:
            raw_cells = detect_multiple_cells(
                img_normalized,
                min_cell_size=800,
                max_cell_size_ratio=0.20,
                circularity_threshold=0.50,
                overlap_threshold=0.50
            )

        # Only report "multi-cell" when > 1 cell instance is found
        if len(raw_cells) <= 1:
            return JSONResponse({
                'multi_cell': {
                    'total_cells': 0,
                    'cells': [],
                    'image_with_boxes_base64': None
                }
            })

        # Build frontend-compatible cell objects (id, bbox, confidence, crop)
        cells = []
        for idx, cell in enumerate(raw_cells):
            x, y, w_box, h_box = int(cell['x']), int(cell['y']), int(cell['width']), int(cell['height'])
            crop = img_normalized[max(0, y):max(0, y) + max(1, h_box), max(0, x):max(0, x) + max(1, w_box)]
            crop_b64 = image_to_base64(crop) if crop.size else None

            # Heuristic confidence based on area (bounded)
            area = float(cell.get('area', w_box * h_box))
            conf = 0.6 + 0.39 * min(1.0, area / 5000.0)
            conf = float(max(0.0, min(0.99, conf)))

            cells.append({
                'cell_id': f'cell_{idx + 1}',
                'bounding_box': {'x': x, 'y': y, 'width': w_box, 'height': h_box},
                'confidence': conf,
                'cell_image_base64': crop_b64
            })
        
        # Draw bounding boxes
        img_with_boxes = img_normalized.copy()
        for idx, cell in enumerate(cells):
            x, y, w_box, h_box = cell['bounding_box']['x'], cell['bounding_box']['y'], cell['bounding_box']['width'], cell['bounding_box']['height']
            cv2.rectangle(img_with_boxes, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"#{idx + 1}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        boxes_b64 = image_to_base64(img_with_boxes)
        
        return JSONResponse({
            'multi_cell': {
                'total_cells': len(cells),
                'cells': cells,
                'image_with_boxes_base64': boxes_b64
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alias route for frontend compatibility
@app.post("/api/v1/multi-cell-detect")
async def detect_cells_alias(file: UploadFile = File(...)):
    return await detect_cells(file)

# ============================================================================
# DOWNLOAD ENDPOINT FOR STAIN-NORMALIZED IMAGES
# ============================================================================

@app.get("/api/v1/download/stain-normalized/{session_id}")
async def download_stain_normalized(session_id: str):
    """Download stain-normalized image"""
    try:
        if session_id not in stain_normalized_cache:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        normalized_img = stain_normalized_cache[session_id]
        
        # Convert to PNG bytes
        img_pil = Image.fromarray(normalized_img)
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        buf.seek(0)
        
        # Clean up cache
        del stain_normalized_cache[session_id]
        
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=stain_normalized_{session_id[:8]}.png"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# BATCH PROCESSING
# ============================================================================

@app.post("/api/v1/batch-process")
async def batch_process(files: List[UploadFile] = File(...)):
    """Process multiple images in batch with dynamic metrics"""
    try:
        job_id = str(uuid.uuid4())
        results = []
        
        for file in files:
            image_data = await file.read()
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_np = np.array(img)
            
            # Apply stain normalization for consistency
            stain_normalized = normalize_stain_macenko(img_np)
            
            # Classification
            img_pil = Image.fromarray(stain_normalized)
            img_tensor = cls_transform(img_pil).to(DEVICE)
            
            with torch.no_grad():
                logits = convnext_model(img_tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0]
                pred_class = probs.argmax().item()
            
            # Calculate dynamic confidence metrics
            pred_prob = float(probs[pred_class])
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
            max_entropy = float(np.log(len(CLASSES)))
            confidence_score = 1.0 - (entropy / max_entropy)
            
            results.append({
                'filename': file.filename,
                'predicted_class': CLASSES[pred_class],
                'confidence': pred_prob * 100,
                'confidence_score': confidence_score,
                'probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
                'needs_review': pred_prob < 0.75
            })
        
        return JSONResponse({
            'job_id': job_id,
            'status': 'completed',
            'total_files': len(files),
            'processed_files': len(results),
            'results': results,
            'created_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# REPORT GENERATION
# ============================================================================

# ============================================================================
# REPORT GENERATION (PDF)
# ============================================================================

def create_pdf_header(canvas, doc, report_title="HerHealth.AI Analysis Report"):
    """Create professional PDF header with logo and aligned title block.

    The header occupies the margin band above the main content frame so that
    it never overlaps tables or images on any page.
    """
    canvas.saveState()

    page_width, _page_height = canvas._pagesize  # full page width

    # Position the header directly above the main content frame
    header_height = 80
    frame_top = doc.bottomMargin + doc.height
    bar_y = frame_top

    # Header background bar spans the full page width
    canvas.setFillColor(colors.HexColor('#2563eb'))
    canvas.rect(0, bar_y, page_width, header_height, stroke=0, fill=1)

    # Try to draw branded logo if available, otherwise fall back to text logo
    logo_drawn = False
    try:
        logo_path = os.environ.get(
            "HERHEALTH_LOGO_PATH",
            os.path.join(os.path.dirname(__file__), "her_health_logo.jpeg"),
        )

        if os.path.exists(logo_path):
            ext = os.path.splitext(logo_path)[1].lower()

            # If SVG and svglib is available, rasterize to PNG in-memory
            if ext == ".svg" and svg2rlg is not None:
                drawing = svg2rlg(logo_path)
                png_bytes = renderPM.drawToString(drawing, fmt="PNG")
                logo_img = ImageReader(io.BytesIO(png_bytes))
            else:
                # Fallback for PNG/JPEG or when svglib is not installed
                logo_img = ImageReader(logo_path)

            logo_width = 140
            logo_height = 36
            logo_y = bar_y + (header_height - logo_height) / 2
            canvas.drawImage(
                logo_img,
                30,
                logo_y,
                width=logo_width,
                height=logo_height,
                mask="auto",
            )
            logo_drawn = True
    except Exception as e:
        print(f"[BACKEND] PDF header logo load failed: {e}")

    canvas.setFillColor(colors.white)
    if not logo_drawn:
        # Text fallback logo styled to resemble the product branding
        primary_y = bar_y + header_height - 30
        secondary_y = bar_y + 16

        canvas.setFont("Helvetica-Bold", 22)
        canvas.drawString(30, primary_y, "HerHealth.AI")

        canvas.setFont("Helvetica", 11)
        canvas.drawString(30, secondary_y, "Cervical Cancer Screening with AI")

    # Right-aligned report metadata block (title + subtitle + timestamp)
    right_x = doc.width + doc.leftMargin - 30

    title_y = bar_y + header_height - 26
    subtitle_y = bar_y + header_height - 44
    date_y = bar_y + 16

    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawRightString(right_x, title_y, report_title)

    canvas.setFont("Helvetica", 11)
    canvas.drawRightString(right_x, subtitle_y, "AI-assisted cytology report")

    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(right_x, date_y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    canvas.restoreState()

def create_pdf_footer(canvas, doc):
    """Create PDF footer with page numbers and disclaimer"""
    canvas.saveState()
    
    # Footer line
    canvas.setStrokeColor(colors.HexColor('#e5e7eb'))
    canvas.line(30, 50, doc.width + doc.leftMargin - 30, 50)
    
    # Page number
    canvas.setFont("Helvetica", 9)
    canvas.drawString(30, 35, f"Page {doc.page}")
    
    # Disclaimer
    canvas.setFont("Helvetica", 8)
    disclaimer = "This is an AI-generated analysis for research purposes. Clinical decisions should involve professional medical review."
    # ReportLab uses drawCentredString (US spelling) on Canvas
    canvas.drawCentredString(doc.width / 2 + doc.leftMargin, 20, disclaimer)
    
    canvas.restoreState()

def base64_to_image(base64_string, max_width=400, max_height=300):
    """Convert base64 string to ReportLab Image object with size constraints"""
    try:
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        # Calculate scaling to fit within max dimensions
        width, height = img.size
        scale_w = max_width / width if width > max_width else 1
        scale_h = max_height / height if height > max_height else 1
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to BytesIO
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        # RLImage can take a file-like object directly
        return RLImage(img_buffer, width=new_width, height=new_height)
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def generate_pdf_report(analysis_data, patient_metadata, original_image_file):
    """Generate comprehensive PDF report with all analysis components"""
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=100,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.HexColor('#1f2937'),
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#2563eb'),
        borderWidth=1,
        borderColor=colors.HexColor('#e5e7eb'),
        borderPadding=8,
        backColor=colors.HexColor('#f8fafc')
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.spaceAfter = 6
    
    # Story elements
    story = []
    
    # Title
    story.append(Paragraph("Cervical Cell Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Section
    story.append(Paragraph("Patient Information", heading_style))
    
    patient_data = [
        ['Patient ID:', patient_metadata.get('patient_id', 'N/A')],
        ['Patient Name:', patient_metadata.get('patient_name', 'Anonymous')],
        ['Analysis Date:', patient_metadata.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))],
        ['Image Filename:', original_image_file.filename if hasattr(original_image_file, 'filename') else 'N/A'],
        ['Model Version:', 'v2.0.0']
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Classification Results
    story.append(Paragraph("Classification Result", heading_style))

    predicted_class = analysis_data.get('predicted_class', 'Unknown')
    probabilities = analysis_data.get('probabilities', {}) or {}

    story.append(Paragraph(f"<b>Primary Diagnosis:</b> {predicted_class}", normal_style))
    story.append(Spacer(1, 6))

    # Probability distribution table
    prob_data = [['Cell Type', 'Model Confidence']]
    for class_name, prob in probabilities.items():
        prob_data.append([class_name, f"{prob:.2%}"])
    
    prob_table = Table(prob_data, colWidths=[2.5*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
    ]))
    
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # Brief interpretation guide for classification
    classification_interp = """
    <b>Interpretation:</b><br/>
    • Highest confidence cell type = primary diagnosis (decision support only).
    """
    story.append(Paragraph(classification_interp, normal_style))
    story.append(Spacer(1, 20))
    
    # Images Section
    story.append(Paragraph("Image & Explainability Analysis", heading_style))

    images_block = analysis_data.get('images', {}) or {}

    # Base64 sources
    original_b64 = analysis_data.get('original_image_base64') or images_block.get('original')
    scorecam_b64 = analysis_data.get('xai_scorecam_base64') or images_block.get('scorecam')
    layercam_b64 = analysis_data.get('xai_layercam_base64') or images_block.get('layercam')

    # Build a 3-column grid: Original | ScoreCAM | LayerCAM
    if original_b64 or scorecam_b64 or layercam_b64:
        images_row = []
        labels_row = []

        # Use consistent target size so all three images align properly
        target_width = 1.8 * inch
        target_height = 1.8 * inch

        # Original image
        if original_b64:
            orig_img = base64_to_image(original_b64, max_width=170, max_height=170)
            if orig_img:
                orig_img.drawWidth = target_width
                orig_img.drawHeight = target_height
            images_row.append(orig_img or "")
            labels_row.append(Paragraph("<b>Original Image</b>", normal_style))
        else:
            images_row.append("")
            labels_row.append("")

        # Score-CAM
        if scorecam_b64:
            scorecam_img = base64_to_image(scorecam_b64, max_width=170, max_height=170)
            if scorecam_img:
                scorecam_img.drawWidth = target_width
                scorecam_img.drawHeight = target_height
            images_row.append(scorecam_img or "")
            labels_row.append(Paragraph("<b>ScoreCAM</b>", normal_style))
        else:
            images_row.append("")
            labels_row.append("")

        # Layer-CAM
        if layercam_b64:
            layercam_img = base64_to_image(layercam_b64, max_width=170, max_height=170)
            if layercam_img:
                layercam_img.drawWidth = target_width
                layercam_img.drawHeight = target_height
            images_row.append(layercam_img or "")
            labels_row.append(Paragraph("<b>LayerCAM</b>", normal_style))
        else:
            images_row.append("")
            labels_row.append("")

        img_table = Table([images_row, labels_row], colWidths=[target_width, target_width, target_width])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 1), (-1, 1), 4),
        ]))

        story.append(img_table)
        story.append(Spacer(1, 10))

        xai_interp = """
        <b>Interpretation:</b><br/>
        • ScoreCAM/LayerCAM highlight regions that most influenced the model.<br/>
        • Use highlighted areas to check if the model focuses on true cellular structures.
        """
        story.append(Paragraph(xai_interp, normal_style))
        story.append(Spacer(1, 15))

    story.append(PageBreak())
    
    # Uncertainty & Confidence Analysis
    uncertainty = analysis_data.get('uncertainty', {}) or {}
    if uncertainty:
        story.append(Paragraph("Confidence & Uncertainty Analysis", heading_style))

        raw_conf = uncertainty.get('confidence', 0)
        # Normalize confidence – accept either 0-1 or 0-100
        if raw_conf > 1:
            confidence = float(raw_conf)
        else:
            confidence = float(raw_conf) * 100.0

        entropy = float(uncertainty.get('entropy', uncertainty.get('entropy_normalized', 0.0)) or 0.0)
        stability = float(uncertainty.get('prediction_stability', uncertainty.get('overall_uncertainty', 0.0)) or 0.0)

        uncertainty_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Confidence Score', f"{confidence:.1f}%", 'High' if confidence > 80 else 'Moderate' if confidence > 60 else 'Low'],
            ['Prediction Entropy', f"{entropy:.3f}", 'Low' if entropy < 0.5 else 'Moderate' if entropy < 1.0 else 'High'],
            ['Model Stability', f"{stability}%", 'Stable' if stability > 70 else 'Moderate' if stability > 50 else 'Unstable']
        ]
        
        uncertainty_table = Table(uncertainty_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        uncertainty_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        
        story.append(uncertainty_table)
        story.append(Spacer(1, 8))

        uncertainty_interp = """
        <b>Interpretation:</b><br/>
        • &gt; 80% confidence → usable result.<br/>
        • 60–80% confidence → moderate risk, review with caution.
        """
        story.append(Paragraph(uncertainty_interp, normal_style))
        story.append(Spacer(1, 20))

    # Clinical Decision Support
    clinical_decision = analysis_data.get('clinical_decision', {}) or {}
    if clinical_decision:
        story.append(Paragraph("Clinical Decision Support", heading_style))
        
        risk_level = clinical_decision.get('risk_level', 'moderate')
        risk_score = clinical_decision.get('risk_score', 0)
        needs_review = clinical_decision.get('needs_review', False)
        
        # Risk assessment
        story.append(Paragraph(f"<b>Risk Level:</b> {risk_level.title()}", normal_style))
        story.append(Paragraph(f"<b>Risk Score:</b> {risk_score:.1f}/100", normal_style))
        story.append(Paragraph(f"<b>Requires Review:</b> {'Yes' if needs_review else 'No'}", normal_style))
        story.append(Spacer(1, 10))
        
        # Recommendations
        recommendations = clinical_decision.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("<b>Clinical Recommendations:</b>", normal_style))
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", normal_style))
            story.append(Spacer(1, 15))
        
        # Secondary candidates
        secondary = clinical_decision.get('secondary_candidates', [])
        if secondary:
            story.append(Paragraph("<b>Alternative Diagnoses (Differential):</b>", normal_style))
            
            diff_data = [['Cell Type', 'Probability']]
            for candidate in secondary[:3]:  # Top 3 alternatives
                class_name = candidate.get('class', 'Unknown')
                prob = candidate.get('probability', 0)
                diff_data.append([class_name, f"{prob:.2%}"])
            
            diff_table = Table(diff_data, colWidths=[2.5*inch, 1.5*inch])
            diff_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef2f2')]),
            ]))
            
            story.append(diff_table)
            story.append(Spacer(1, 8))

        clinical_interp = """
        <b>Interpretation:</b><br/>
        • Risk level/score summarize how concerning the pattern is.<br/>
        • Follow model recommendations as decision support, not final diagnosis.
        """
        story.append(Paragraph(clinical_interp, normal_style))
        story.append(Spacer(1, 20))
    
    # Footer disclaimer – keep a bit of spacing but avoid forcing
    # an unnecessary extra page when content is long.
    story.append(Spacer(1, 12))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#6b7280'),
        borderWidth=1,
        borderColor=colors.HexColor('#fbbf24'),
        borderPadding=10,
        backColor=colors.HexColor('#fff7ed'),
        alignment=TA_CENTER
    )
    
    disclaimer_text = """
    <b>IMPORTANT CLINICAL DISCLAIMER:</b><br/>
    This AI-generated report is for research and educational purposes only. All diagnostic decisions
    should be made by qualified healthcare professionals in consultation with clinical history,
    physical examination, and additional testing as appropriate. This system is not intended
    to replace professional medical judgment or clinical correlation.
    """
    
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF with custom header/footer
    def add_page_elements(canvas, doc):
        create_pdf_header(canvas, doc)
        create_pdf_footer(canvas, doc)
    
    doc.build(story, onFirstPage=add_page_elements, onLaterPages=add_page_elements)
    
    buffer.seek(0)
    return buffer

@app.post("/api/v1/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    analysis: str = Form(...),
):
    """Generate a comprehensive PDF analysis report with all model outputs and patient details"""
    try:
        # Parse analysis data
        try:
            analysis_data = json.loads(analysis)
        except Exception as parse_error:
            raise HTTPException(status_code=400, detail=f"Invalid analysis payload: {parse_error}")

        # Extract patient metadata
        patient_metadata = {
            'patient_id': analysis_data.get('patient_id'),
            'patient_name': analysis_data.get('patient_name', 'Anonymous'),
            'analysis_date': analysis_data.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))
        }

        # Generate PDF report
        pdf_buffer = generate_pdf_report(analysis_data, patient_metadata, file)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_part = patient_metadata.get('patient_id', 'unknown').replace(' ', '_') if patient_metadata.get('patient_id') else 'anonymous'
        filename = f"herhealth_analysis_{patient_part}_{timestamp}.pdf"
        
        # Return PDF as download
        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[BACKEND] PDF generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/api/v1/export-pdf")
async def export_pdf():
    """Legacy endpoint - redirects to generate-report"""
    return JSONResponse({
        'message': 'Use /api/v1/generate-report endpoint for PDF generation',
        'redirect': '/api/v1/generate-report'
    })


@app.get("/api/oldpreds/{id}/pdf")
async def download_old_prediction_pdf(id: str):
    """Generate and download a PDF report for a stored prediction.

    This reuses the existing PDF generation pipeline but fills the
    analysis payload from the MongoDB document instead of requiring
    the raw image upload again.
    """
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database is not initialized")

        try:
            oid = ObjectId(id)
        except Exception:
            return JSONResponse({"error": "Invalid id format"}, status_code=400)

        def _fetch_one():
            return db.predictions.find_one({"_id": oid})

        record = await run_in_threadpool(_fetch_one)
        if not record:
            return JSONResponse({"error": "Record not found"}, status_code=404)

        # Normalize record into the same analysis payload used for live reports
        analysis_data = _build_analysis_payload_from_record(record)

        # Use patient metadata directly from the stored MongoDB record
        patient_metadata = {
            'patient_id': record.get('patient_id'),
            'patient_name': record.get('patient_name', 'Anonymous'),
            'analysis_date': (
                record.get('timestamp').strftime('%Y-%m-%d')
                if isinstance(record.get('timestamp'), datetime)
                else datetime.utcnow().strftime('%Y-%m-%d')
            ),
        }

        # For historical reports we may not have the original file, so pass None
        # (the PDF generator only uses this for filename metadata)
        pdf_buffer = generate_pdf_report(analysis_data, patient_metadata, None)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_part = patient_metadata.get('patient_id', 'unknown').replace(' ', '_') if patient_metadata.get('patient_id') else 'anonymous'
        filename = f"herhealth_analysis_{patient_part}_{timestamp}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[BACKEND] Historical PDF generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Historical PDF generation failed: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'\n[BACKEND] Starting FastAPI server on http://0.0.0.0:{port}')
    print(f'[BACKEND] API documentation: http://localhost:{port}/docs\n')
    uvicorn.run(app, host='0.0.0.0', port=port)
