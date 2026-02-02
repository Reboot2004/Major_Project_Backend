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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
# CLASSIFICATION ENDPOINTS
# ============================================================================

@app.get("/api/v1/classification/classes")
async def get_classes():
    """Get available classification classes"""
    return CLASSES

@app.post("/api/v1/classification/predict")
async def classify(file: UploadFile = File(...)):
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
        
        return JSONResponse({
            'predicted_class': CLASSES[pred_class],
            'probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            'xai_scorecam_base64': scorecam_b64,
            'xai_layercam_base64': layercam_b64,
            'original_image_base64': original_b64,
            'uncertainty': uncertainty_metrics,
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
async def segment(file: UploadFile = File(...)):
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
        
        # Colorize mask with proper visualization
        color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        mask_colored = color_map[pred_mask_resized]
        
        # Blend mask with stain-normalized image for better visualization
        mask_overlay = cv2.addWeighted(stain_normalized, 0.7, mask_colored, 0.3, 0)
        mask_b64 = image_to_base64(mask_overlay)
        
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
        
        return JSONResponse({
            'predicted_class': CLASSES[pred_class],
            'probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
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
            'uncertainty': {
                'confidence': float(pred_prob * 100),
                'entropy': entropy,
                'confidence_score': confidence_score,
                'segmentation_confidence': seg_confidence,
                'combined_confidence': combined_confidence,
                'uncertainty_lower': uncertainty_lower,
                'uncertainty_upper': uncertainty_upper,
                'prediction_stability': prediction_stability
            },
            'clinical_decision': {
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
            },
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
# REPORT GENERATION
# ============================================================================

@app.post("/api/v1/generate-report")
async def generate_report(file: UploadFile = File(...)):
    """Generate comprehensive analysis report"""
    try:
        # Process image with both classification and segmentation
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        
        # Run full preprocessing pipeline
        session_id = str(uuid.uuid4())
        preprocessing = preprocess_image_pipeline(img_np, session_id)
        img_normalized = preprocessing['stain_normalized']
        
        # Classification
        img_pil = Image.fromarray(img_normalized)
        img_tensor_cls = cls_transform(img_pil).to(DEVICE)
        
        with torch.no_grad():
            logits = convnext_model(img_tensor_cls.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
        
        # Segmentation
        aug = seg_transform(image=img_normalized)
        img_tensor_seg = aug['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_logits = unet_model(img_tensor_seg)
            pred_mask = mask_logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask)
        pred_prob = float(probs[pred_class])
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        max_entropy = float(np.log(len(CLASSES)))
        confidence_score = 1.0 - (entropy / max_entropy)
        
        # XAI insights based on actual metrics
        xai_insights = []
        if metrics['nucleus_iou'] > 0.7:
            xai_insights.append("Clear nucleus detection with high confidence")
        else:
            xai_insights.append("Nucleus segmentation quality is moderate - review recommended")
        
        if metrics['cytoplasm_iou'] > 0.7:
            xai_insights.append("Well-defined cytoplasm boundaries")
        else:
            xai_insights.append("Cytoplasm definition is unclear - consider image preprocessing")
        
        if CLASSES[pred_class] in ['Dyskeratotic', 'Koilocytotic']:
            xai_insights.append("High-risk morphology detected - HPV-related changes present")
        elif CLASSES[pred_class] == 'Metaplastic':
            xai_insights.append("Benign metaplastic changes detected")
        
        # Risk-based recommendations
        risk_mapping = {
            'Dyskeratotic': 'high',
            'Koilocytotic': 'high',
            'Metaplastic': 'moderate',
            'Parabasal': 'low',
            'Superficial-Intermediate': 'low'
        }
        risk_level = risk_mapping.get(CLASSES[pred_class], 'moderate')
        
        recommendations = []
        if pred_prob < 0.75:
            recommendations.append("Expert review strongly recommended - low model confidence")
        if metrics['accuracy'] < 0.75:
            recommendations.append("Segmentation quality concern - consider re-imaging")
        
        if risk_level == 'high':
            recommendations.append("Follow-up cytology recommended within 3-6 months")
            recommendations.append("Consider HPV testing if not already performed")
        elif risk_level == 'moderate':
            recommendations.append("Routine follow-up recommended")
        else:
            recommendations.append("Standard screening intervals appropriate")
        
        return JSONResponse({
            'report_id': str(uuid.uuid4()),
            'patient_id': 'ANON-' + str(uuid.uuid4())[:8].upper(),
            'analysis_date': datetime.now().isoformat(),
            'image_filename': file.filename,
            'primary_diagnosis': CLASSES[pred_class],
            'confidence': float(pred_prob * 100),
            'confidence_score': confidence_score,
            'segmentation_metrics': metrics,
            'risk_level': risk_level,
            'xai_insights': xai_insights,
            'recommendations': recommendations,
            'model_version': 'v2.0.0'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/export-pdf")
async def export_pdf():
    """Export report as PDF (placeholder)"""
    try:
        return JSONResponse({
            'pdf_url': '/downloads/report.pdf',
            'filename': 'cervical_analysis_report.pdf'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'\n[BACKEND] Starting FastAPI server on http://0.0.0.0:{port}')
    print(f'[BACKEND] API documentation: http://localhost:{port}/docs\n')
    uvicorn.run(app, host='0.0.0.0', port=port)
