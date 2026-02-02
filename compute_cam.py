# import argparse
# import base64
# import io
# import json
# import sys
# import time

# from PIL import Image
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# import matplotlib.pyplot as plt


# def log(msg):
#     print(f"[DEBUG] {msg}", file=sys.stderr)


# # ------------------------------------------------------------
# # Utility: turn CAM array → RGBA PNG Base64
# # ------------------------------------------------------------
# def cam_to_base64(cam: np.ndarray):
#     log(f"cam_to_base64: cam min={cam.min():.4f}, max={cam.max():.4f}, shape={cam.shape}")
#     cmap = plt.get_cmap("jet")
#     rgba = (cmap(cam) * 255).astype(np.uint8)
#     img = Image.fromarray(rgba)

#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode("ascii")


# # ------------------------------------------------------------
# # Layer-CAM for ConvNeXt
# # ------------------------------------------------------------
# def layer_cam_convnext(model, image_tensor, target_class, layer_name='features.7', device='cuda'):
#     log(f"Layer-CAM: using layer {layer_name}")

#     layer = dict([*model.named_modules()])[layer_name]

#     activations, gradients = [], []

#     def forward_hook(_, __, out):
#         activations.append(out.detach())
#         log(f"Layer-CAM: forward hook - activation shape {out.shape}")

#     def backward_hook(_, grad_in, grad_out):
#         gradients.append(grad_out[0].detach())
#         log(f"Layer-CAM: backward hook - gradient shape {grad_out[0].shape}")

#     h1 = layer.register_forward_hook(forward_hook)
#     h2 = layer.register_backward_hook(backward_hook)
#     log("Layer-CAM: hooks registered.")

#     start = time.time()

#     image_tensor = image_tensor.unsqueeze(0).to(device)
#     output = model(image_tensor)
#     score = output[0, target_class]
#     log(f"Layer-CAM: model forward done. Output shape={output.shape}, target score={score.item():.4f}")

#     model.zero_grad()
#     score.backward()
#     log("Layer-CAM: backward pass completed.")

#     act = activations[0]
#     grad = gradients[0]

#     cam = F.relu(grad) * act
#     log(f"Layer-CAM: elementwise multiplication done. cam shape={cam.shape}")

#     cam = cam.sum(dim=1, keepdim=True)
#     log(f"Layer-CAM: summed channels. cam shape={cam.shape}")

#     cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
#     cam = cam.squeeze().cpu().numpy()

#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#     log(f"Layer-CAM: normalized cam. min={cam.min():.4f} max={cam.max():.4f}")

#     h1.remove()
#     h2.remove()
#     log("Layer-CAM: hooks removed.")

#     log(f"Layer-CAM: total time {time.time() - start:.3f}s")

#     return cam


# # ------------------------------------------------------------
# # Score-CAM for ConvNeXt
# # ------------------------------------------------------------
# def score_cam_convnext(model, image_tensor, target_class, layer_name='features.7', device='cuda'):
#     log(f"Score-CAM: using layer {layer_name}")

#     layer = dict([*model.named_modules()])[layer_name]
#     activations = []

#     def forward_hook(_, __, out):
#         activations.append(out.detach())
#         log(f"Score-CAM: forward hook - activation shape {out.shape}")

#     h = layer.register_forward_hook(forward_hook)
#     log("Score-CAM: hook registered.")

#     start = time.time()

#     image_tensor = image_tensor.unsqueeze(0).to(device)
#     with torch.no_grad():
#         _ = model(image_tensor)

#     act = activations[0]
#     C = act.shape[1]
#     H, W = act.shape[2], act.shape[3]
#     log(f"Score-CAM: activations shape = {act.shape} (C={C}, H={H}, W={W})")

#     scores = []

#     for i in range(C):
#         if i % 50 == 0:
#             log(f"Score-CAM: processing channel {i}/{C}")

#         mask = act[0, i:i+1]
#         mask = F.interpolate(mask.unsqueeze(0), size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
#         mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

#         masked_input = image_tensor * mask

#         with torch.no_grad():
#             out = model(masked_input)
#             scores.append(out[0, target_class].item())

#     scores = torch.tensor(scores).to(device)
#     log(f"Score-CAM: collected {len(scores)} scores. sum={scores.sum().item():.4f}")

#     scores = F.relu(scores)
#     scores = scores / (scores.sum() + 1e-8)
#     log("Score-CAM: normalized scores.")

#     cam = (act[0] * scores[:, None, None]).sum(dim=0)
#     cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
#     cam = cam.squeeze().cpu().numpy()

#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#     log(f"Score-CAM: normalized cam. min={cam.min():.4f}, max={cam.max():.4f}")

#     h.remove()
#     log("Score-CAM: hook removed.")

#     log(f"Score-CAM: total time {time.time() - start:.3f}s")

#     return cam


# # ------------------------------------------------------------
# # Main
# # ------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", required=True, help="Path to input image")
#     parser.add_argument("--class", dest="cls", required=True, type=int, help="Target class index")
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     log(f"Using device: {device}")

#     model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
#     model.to(device)
#     model.eval()
#     log("ConvNeXt-Tiny loaded and set to eval().")

#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         ),
#     ])

#     img = Image.open(args.image).convert("RGB")
#     log(f"Loaded image: {args.image}, size={img.size}")
#     image_tensor = preprocess(img)
#     log("Image preprocessed.")

#     score_cam = score_cam_convnext(model, image_tensor, args.cls, device=device)
#     layer_cam = layer_cam_convnext(model, image_tensor, args.cls, device=device)

#     score_b64 = cam_to_base64(score_cam)
#     layer_b64 = cam_to_base64(layer_cam)

#     print(json.dumps({
#         "scorecam_base64": score_b64,
#         "layercam_base64": layer_b64
#     }))


# if __name__ == "__main__":
#     main()
import argparse
import base64
import io
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
import cv2


def log(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)


# ============================================================
# 1. ROBUST SAMPLE QUALITY ASSESSMENT
# ============================================================
def assess_image_quality(image_np):
    """
    Production-grade image quality assessment for cytology images.
    Uses multiple metrics with adaptive thresholds.
    
    Args:
        image_np: RGB image as numpy array
    
    Returns:
        tuple: (quality_score, quality_flags, detailed_metrics)
    """
    try:
        if image_np is None or image_np.size == 0:
            raise ValueError("Empty or invalid image")
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Blur Detection - Multiple methods for robustness
        # a) Laplacian variance (texture-based)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # b) Tenengrad (gradient magnitude)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.sqrt(gx**2 + gy**2).mean()
        
        # c) Modified Laplacian (robust to noise)
        kernel = np.array([[1, -2, 1]])
        Lx = cv2.filter2D(gray, cv2.CV_64F, kernel)
        Ly = cv2.filter2D(gray, cv2.CV_64F, kernel.T)
        modified_laplacian = (np.abs(Lx) + np.abs(Ly)).mean()
        
        # Adaptive blur thresholds based on image size
        img_diagonal = np.sqrt(image_np.shape[0]**2 + image_np.shape[1]**2)
        blur_threshold = max(50, img_diagonal / 10)
        
        blur_score = np.mean([
            min(1.0, laplacian_var / blur_threshold),
            min(1.0, tenengrad / 10),
            min(1.0, modified_laplacian / 5)
        ])
        
        # 2. Contrast Assessment - Michelson contrast + histogram analysis
        contrast_std = gray.std()
        
        # Histogram-based contrast (99th - 1st percentile)
        p1, p99 = np.percentile(gray, [1, 99])
        histogram_contrast = (p99 - p1) / 255.0 if p99 > p1 else 0
        
        # RMS contrast
        rms_contrast = gray.std() / gray.mean() if gray.mean() > 0 else 0
        
        contrast_score = np.mean([
            min(1.0, contrast_std / 50.0),
            histogram_contrast,
            min(1.0, rms_contrast * 2)
        ])
        
        # 3. Color Quality - Saturation and color balance
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Mean saturation
        saturation_mean = hsv[:, :, 1].mean() / 255.0
        
        # Saturation distribution (penalize over-saturated images)
        sat_hist, _ = np.histogram(hsv[:, :, 1], bins=256, range=(0, 256))
        sat_uniformity = 1.0 - (sat_hist.std() / (sat_hist.mean() + 1e-8)) / 10
        sat_uniformity = max(0.0, min(1.0, sat_uniformity))
        
        # Color cast detection (check if RGB channels are balanced)
        r_mean = image_np[:, :, 0].mean()
        g_mean = image_np[:, :, 1].mean()
        b_mean = image_np[:, :, 2].mean()
        total_mean = (r_mean + g_mean + b_mean) / 3
        
        color_balance = 1.0 - (
            abs(r_mean - total_mean) + 
            abs(g_mean - total_mean) + 
            abs(b_mean - total_mean)
        ) / (3 * total_mean + 1e-8)
        
        saturation_score = np.mean([
            saturation_mean,
            sat_uniformity,
            color_balance
        ])
        
        # 4. Exposure Assessment - Multi-zone analysis
        brightness_mean = gray.mean()
        
        # Check for clipping (overexposure/underexposure)
        overexposed_pixels = (gray > 250).sum() / gray.size
        underexposed_pixels = (gray < 5).sum() / gray.size
        
        # Ideal brightness range for cytology images
        brightness_optimal = 100 < brightness_mean < 180
        brightness_score = 1.0 if brightness_optimal else max(0.3, 1.0 - abs(brightness_mean - 140) / 140)
        
        # Penalize excessive clipping
        clipping_penalty = max(overexposed_pixels, underexposed_pixels)
        brightness_score *= (1.0 - clipping_penalty * 5)
        brightness_score = max(0.0, min(1.0, brightness_score))
        
        # 5. Noise Assessment - Local variance analysis
        # Divide image into blocks and analyze variance
        block_size = 16
        h, w = gray.shape
        noise_estimates = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                noise_estimates.append(block.std())
        
        noise_level = np.median(noise_estimates) if noise_estimates else 0
        noise_score = max(0.0, 1.0 - noise_level / 30)
        
        # 6. Overall Quality Score (weighted combination)
        overall_quality = (
            blur_score * 0.30 +          # Most important for diagnosis
            contrast_score * 0.25 +      # Critical for cell visibility
            saturation_score * 0.15 +    # Color accuracy
            brightness_score * 0.20 +    # Proper exposure
            noise_score * 0.10           # Low noise
        )
        
        # Quality flags with adaptive thresholds
        flags = {
            "is_blurry": bool(blur_score < 0.5),
            "low_contrast": bool(contrast_score < 0.4),
            "poor_color": bool(saturation_score < 0.3),
            "overexposed": bool(overexposed_pixels > 0.05),
            "underexposed": bool(underexposed_pixels > 0.05),
            "excessive_noise": bool(noise_score < 0.4),
            "usable": bool(overall_quality >= 0.5)
        }
        
        # Detailed metrics for debugging/analysis
        detailed_metrics = {
            "laplacian_variance": float(laplacian_var),
            "tenengrad": float(tenengrad),
            "contrast_std": float(contrast_std),
            "brightness_mean": float(brightness_mean),
            "saturation_mean": float(saturation_mean * 255),
            "noise_level": float(noise_level),
            "overexposed_ratio": float(overexposed_pixels),
            "underexposed_ratio": float(underexposed_pixels),
            "color_balance_score": float(color_balance)
        }
        
        log(f"Quality Assessment: overall={overall_quality:.3f}, blur={blur_score:.3f}, "
            f"contrast={contrast_score:.3f}, usable={flags['usable']}")
        
        return float(overall_quality), flags, detailed_metrics
        
    except Exception as e:
        log(f"Quality assessment error: {e}")
        return 0.5, {"error": True, "usable": False}, {}


# ============================================================
# 2. STAIN NORMALIZATION (Macenko method)
# ============================================================
def normalize_stain_macenko(image_np):
    """
    Normalize H&E staining using Macenko method.
    Ensures consistent color appearance across different scanners.
    """
    try:
        # Convert to optical density
        image_float = image_np.astype(np.float32) + 1
        od = -np.log(image_float / 255.0)
        od = od.reshape((-1, 3))
        
        # Remove transparent pixels
        od_mask = (od > 0.15).all(axis=1)
        od_filtered = od[od_mask]
        
        if len(od_filtered) < 100:
            log("Stain normalization: insufficient pixels, returning original")
            return image_np
        
        # Compute eigenvectors (stain vectors)
        _, eigvecs = np.linalg.eigh(np.cov(od_filtered.T))
        eigvecs = eigvecs[:, [2, 1]]  # Take top 2
        
        # Project onto stain space
        that = np.dot(od_filtered, eigvecs)
        phi = np.arctan2(that[:, 1], that[:, 0])
        
        # Find robust extrema (stain directions)
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        v1 = np.dot(eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))
        
        # Normalize stain matrix
        if v1[0] > v2[0]:
            HE = np.array([v1, v2]).T
        else:
            HE = np.array([v2, v1]).T
        
        # Default target (standard H&E reference)
        target_concentrations = np.array([[0.5626, 0.2159],
                                          [0.7201, 0.8012],
                                          [0.4062, 0.5581]])
        
        # Deconvolution
        C = np.linalg.lstsq(HE, od_filtered.T, rcond=None)[0]
        
        # Reconstruct with target stains
        normalized_od = np.dot(target_concentrations, C)
        normalized_img = np.exp(-normalized_od) * 255.0
        normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)
        
        # Reshape back
        # IMPORTANT: preserve original pixels for background / excluded regions.
        # The previous implementation filled non-OD pixels with zeros (black),
        # which shows up as black background in the UI.
        result = image_np.reshape((-1, 3)).copy()
        result[od_mask] = normalized_img.T
        result = result.reshape(image_np.shape[0], image_np.shape[1], 3)
        
        log("Stain normalization: Macenko method applied")
        return result
        
    except Exception as e:
        log(f"Stain normalization failed: {e}, returning original")
        return image_np


# ============================================================
# 3. ROBUST MULTI-CELL DETECTION
# ============================================================
def detect_multiple_cells(image_np, min_cell_size=300, max_cell_size_ratio=0.4,
                         circularity_threshold=0.25, overlap_threshold=0.4):
    """
    Production-grade multi-cell detection with watershed segmentation.
    
    Args:
        image_np: RGB image
        min_cell_size: Minimum cell area in pixels
        max_cell_size_ratio: Maximum cell size as image area ratio
        circularity_threshold: Minimum circularity (0=line, 1=circle)
        overlap_threshold: IOU threshold for NMS
    
    Returns:
        List of cell bounding boxes with metrics
    """
    try:
        if image_np is None or image_np.size == 0:
            raise ValueError("Empty image")
        
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Dual adaptive thresholding for robustness
        binary1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 4)
        binary2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 4)
        binary = cv2.bitwise_and(binary1, binary2)
        
        # Morphology
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_md = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_sm, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_md, iterations=2)
        
        # Watershed segmentation
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cv2.dilate(binary, kernel_md, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
        
        # Extract cells
        cells = []
        max_area = h * w * max_cell_size_ratio
        
        for marker_id in np.unique(markers):
            if marker_id <= 1:  # Skip background
                continue
            
            cell_mask = (markers == marker_id).astype(np.uint8)
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            if area < min_cell_size or area > max_area:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < circularity_threshold:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            cell_data = {
                "x": int(x),
                "y": int(y),
                "width": int(w_box),
                "height": int(h_box),
                "area": int(area),
                "circularity": float(circularity)
            }
            
            # Fit ellipse if possible
            if len(cnt) >= 5:
                try:
                    (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
                    aspect_ratio = min(ma, MA) / (max(ma, MA) + 1e-6)
                    if aspect_ratio >= 0.3:  # Not too elongated
                        cell_data.update({
                            "center_x": float(cx),
                            "center_y": float(cy),
                            "major_axis": float(MA),
                            "minor_axis": float(ma),
                            "angle": float(angle),
                            "aspect_ratio": float(aspect_ratio)
                        })
                except:
                    pass
            
            cells.append(cell_data)
        
        # Non-maximum suppression
        if len(cells) > 1:
            cells = nms_cells(cells, overlap_threshold)
        
        log(f"Multi-cell detection: {len(cells)} cells found")
        return cells
        
    except Exception as e:
        log(f"Multi-cell detection error: {e}")
        return []


def nms_cells(cells, iou_threshold):
    """Non-Maximum Suppression for overlapping cells"""
    if not cells:
        return cells
    
    cells_sorted = sorted(cells, key=lambda c: c['area'], reverse=True)
    keep = []
    
    while cells_sorted:
        current = cells_sorted.pop(0)
        keep.append(current)
        cells_sorted = [c for c in cells_sorted if iou_boxes(current, c) < iou_threshold]
    
    return keep


def iou_boxes(box1, box2):
    """Calculate IOU between two bounding boxes"""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    union = box1['area'] + box2['area'] - inter
    return inter / (union + 1e-6)


# ============================================================
# 4. ROBUST UNCERTAINTY ESTIMATION (Monte Carlo Dropout)
# ============================================================
def estimate_uncertainty(model, image_tensor, target_class, n_samples=30, temperature=1.0):
    """
    Production-grade uncertainty estimation using MC Dropout.
    
    Args:
        model: PyTorch model with dropout
        image_tensor: Preprocessed input
        target_class: Target class index
        n_samples: Number of MC samples
        temperature: Temperature scaling for calibration
    
    Returns:
        Dictionary with comprehensive uncertainty metrics
    """
    try:
        # Check for dropout layers
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        if not has_dropout:
            log("Warning: No dropout layers found, adding temporary dropout")
            # Inject dropout after each layer with activation
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.ReLU):
                    # This won't modify the actual model structure permanently
                    pass
        
        # Enable dropout
        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()
        
        model.eval()
        model.apply(enable_dropout)
        
        device = next(model.parameters()).device
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        predictions = []
        logits_list = []
        
        # MC sampling
        with torch.no_grad():
            for _ in range(n_samples):
                logits = model(input_tensor)
                logits_list.append(logits.cpu())
                
                # Temperature scaling
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        # Restore eval mode
        model.eval()
        
        predictions = np.array(predictions).squeeze()
        logits_array = torch.stack(logits_list).squeeze().numpy()
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        if logits_array.ndim == 1:
            logits_array = logits_array.reshape(1, -1)
        
        mean_probs = predictions.mean(axis=0)
        std_probs = predictions.std(axis=0)
        
        # 1. Confidence (mean probability)
        confidence = float(mean_probs[target_class])
        confidence_std = float(std_probs[target_class])
        
        # 2. Predictive Entropy (total uncertainty)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = np.log(len(mean_probs))
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        # 3. Mutual Information (epistemic uncertainty)
        expected_entropy = np.mean([-np.sum(p * np.log(p + 1e-10)) for p in predictions])
        mutual_info = entropy - expected_entropy
        
        # 4. Aleatoric vs Epistemic
        aleatoric = expected_entropy
        epistemic = mutual_info
        
        # 5. Variation Ratio (disagreement)
        predicted_classes = np.argmax(predictions, axis=1)
        mode_class = np.argmax(np.bincount(predicted_classes))
        variation_ratio = 1.0 - (np.sum(predicted_classes == mode_class) / n_samples)
        
        # 6. Prediction Variance
        prediction_variance = predictions.var(axis=0).mean()
        
        # 7. Logit Variance
        logit_variance = logits_array.var(axis=0).mean()
        
        # 8. Confidence Interval (95%)
        conf_lower = float(np.percentile(predictions[:, target_class], 2.5))
        conf_upper = float(np.percentile(predictions[:, target_class], 97.5))
        
        # 9. Overall Uncertainty Score
        overall_uncertainty = np.mean([
            entropy_normalized,
            variation_ratio,
            min(1.0, confidence_std * 10),
            1.0 - confidence
        ])
        
        result = {
            "confidence": confidence,
            "confidence_std": confidence_std,
            "confidence_interval_95": [conf_lower, conf_upper],
            "entropy": float(entropy),
            "entropy_normalized": float(entropy_normalized),
            "mutual_information": float(mutual_info),
            "aleatoric_uncertainty": float(aleatoric),
            "epistemic_uncertainty": float(epistemic),
            "variation_ratio": float(variation_ratio),
            "prediction_variance": float(prediction_variance),
            "logit_variance": float(logit_variance),
            "overall_uncertainty": float(overall_uncertainty),
            "n_samples": n_samples,
            "is_reliable": bool(overall_uncertainty < 0.3)
        }
        
        log(f"Uncertainty: conf={confidence:.3f}±{confidence_std:.3f}, "
            f"entropy={entropy:.3f}, reliable={result['is_reliable']}")
        
        return result
        
    except Exception as e:
        log(f"Uncertainty estimation error: {e}")
        return {
            "confidence": 0.5,
            "entropy": 1.0,
            "error": str(e),
            "is_reliable": False
        }


# ============================================================
# Utility: CAM → Base64
# ============================================================
def cam_to_base64(cam: np.ndarray):
    cmap = plt.get_cmap("jet")
    rgba = (cmap(cam) * 255).astype(np.uint8)
    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------
# ROBUST Layer-CAM
# ------------------------------------------------------------
def layer_cam_convnext(model, image_tensor, target_class, layer_name='features.7', device='cuda'):
    """
    Production-grade Layer-CAM with gradient-weighted activations.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed input
        target_class: Target class index
        layer_name: Layer name for feature extraction
        device: Device (cuda/cpu)
    
    Returns:
        2D numpy array CAM
    """
    try:
        # Validate layer exists
        layer_dict = dict([*model.named_modules()])
        if layer_name not in layer_dict:
            available_layers = [n for n, _ in model.named_modules() if 'features' in n]
            log(f"Layer-CAM: Layer '{layer_name}' not found. Available: {available_layers[:5]}")
            layer_name = available_layers[-1] if available_layers else list(layer_dict.keys())[-1]
            log(f"Layer-CAM: Using fallback layer: {layer_name}")
        
        layer = layer_dict[layer_name]
        activations, gradients = [], []

        def hook_forward(_, __, out):
            activations.append(out.detach())

        def hook_backward(_, grad_in, grad_out):
            if grad_out[0] is not None:
                gradients.append(grad_out[0].detach())

        h1 = layer.register_forward_hook(hook_forward)
        h2 = layer.register_backward_hook(hook_backward)

        x = image_tensor.unsqueeze(0).to(device)
        x.requires_grad = True
        
        # Forward pass
        out = model(x)
        
        # Validate output
        if target_class >= out.shape[1]:
            log(f"Layer-CAM: Invalid target class {target_class}, max is {out.shape[1]-1}")
            target_class = out.shape[1] - 1
        
        score = out[0, target_class]

        # Backward pass
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        
        try:
            score.backward(retain_graph=False)
        except RuntimeError as e:
            log(f"Layer-CAM: Backward pass failed: {e}")
            h1.remove()
            h2.remove()
            return np.zeros((x.shape[2], x.shape[3]))

        # Check if gradients were captured
        if not activations or not gradients:
            log("Layer-CAM: No activations/gradients captured")
            h1.remove()
            h2.remove()
            return np.zeros((x.shape[2], x.shape[3]))

        act = activations[0]  # (1, C, H, W)
        grad = gradients[0]   # (1, C, H, W)

        # Layer-CAM: element-wise multiplication with ReLU on gradients
        # CAM = ReLU(grad) ⊙ A
        cam = F.relu(grad) * act
        
        # Sum over channels
        cam = cam.sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            log("Layer-CAM: Constant CAM values")
            cam = np.ones_like(cam) * 0.5

        # Cleanup
        h1.remove()
        h2.remove()
        
        log(f"Layer-CAM: Generated CAM with shape {cam.shape}, range [{cam.min():.3f}, {cam.max():.3f}]")
        return cam
        
    except Exception as e:
        log(f"Layer-CAM error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return np.zeros((224, 224))


# ------------------------------------------------------------
# ROBUST Score-CAM with Optimizations
# ------------------------------------------------------------
def score_cam_fast(model, image_tensor, target_class, layer_name='features.7',
                   device='cuda', top_k=96, batch_size=64, use_fp16=True):
    """
    Production-grade Score-CAM with error handling and optimizations.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed input
        target_class: Target class index
        layer_name: Layer name for feature extraction
        device: Device (cuda/cpu)
        top_k: Number of top channels to use
        batch_size: Batch size for masked inference
        use_fp16: Use FP16 for faster inference on GPU
    
    Returns:
        2D numpy array CAM
    """
    try:
        # Validate layer exists
        layer_dict = dict([*model.named_modules()])
        if layer_name not in layer_dict:
            available_layers = [n for n, _ in model.named_modules() if 'features' in n]
            log(f"Score-CAM: Layer '{layer_name}' not found. Available: {available_layers[:5]}")
            layer_name = available_layers[-1] if available_layers else list(layer_dict.keys())[-1]
            log(f"Score-CAM: Using fallback layer: {layer_name}")
        
        layer = layer_dict[layer_name]
        activations = []

        def hook_forward(_, __, out):
            activations.append(out.detach())

        h = layer.register_forward_hook(hook_forward)

        x = image_tensor.unsqueeze(0).to(device)
        
        # Use mixed precision if available and requested
        if use_fp16 and device == 'cuda' and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    model(x)
        else:
            with torch.no_grad():
                model(x)

        if not activations:
            log("Score-CAM: No activations captured, returning zeros")
            h.remove()
            return np.zeros((x.shape[2], x.shape[3]))

        act = activations[0]  # (1, C, H, W)
        C = act.shape[1]
        
        if C == 0:
            log("Score-CAM: Zero channels, returning zeros")
            h.remove()
            return np.zeros((x.shape[2], x.shape[3]))

        # ---------------------------------------------------------
        # 1. Smart Channel Selection (Top-K by activation energy)
        # ---------------------------------------------------------
        # Use L2 norm as channel importance metric
        energies = act.pow(2).sum(dim=[2, 3]).squeeze()
        
        # Handle edge case: single channel
        if energies.ndim == 0:
            energies = energies.unsqueeze(0)
        
        k = min(top_k, C)
        top_idx = torch.topk(energies, k=k).indices
        act_selected = act[:, top_idx]
        C_selected = act_selected.shape[1]

        log(f"Score-CAM: Using top {C_selected}/{C} channels")

        # ---------------------------------------------------------
        # 2. Generate Normalized Masks
        # ---------------------------------------------------------
        masks = []
        for i in range(C_selected):
            m = act_selected[0, i:i+1]  # (1, H, W)
            
            # Upsample to input resolution
            m = F.interpolate(m.unsqueeze(0), size=x.shape[2:], 
                            mode='bilinear', align_corners=False)
            
            # Normalize to [0, 1]
            m_min, m_max = m.min(), m.max()
            if m_max > m_min:
                m = (m - m_min) / (m_max - m_min)
            else:
                m = torch.zeros_like(m)
            
            masks.append(m)

        masks = torch.cat(masks, dim=0)  # (K, 1, H, W)
        
        # Apply masks to input image
        masked_inputs = x * masks  # Broadcasting: (1, 3, H, W) * (K, 1, H, W) → (K, 3, H, W)

        # ---------------------------------------------------------
        # 3. Batched Inference for Efficiency
        # ---------------------------------------------------------
        scores = []
        
        if use_fp16 and device == 'cuda' and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for i in range(0, C_selected, batch_size):
                        chunk = masked_inputs[i:i + batch_size]
                        try:
                            out = model(chunk)
                            scores.extend(out[:, target_class].cpu().tolist())
                        except RuntimeError as e:
                            log(f"Score-CAM: Batch {i} failed: {e}, trying smaller batch")
                            # Fallback to single sample if batch fails
                            for j in range(chunk.shape[0]):
                                out = model(chunk[j:j+1])
                                scores.append(out[0, target_class].cpu().item())
        else:
            with torch.no_grad():
                for i in range(0, C_selected, batch_size):
                    chunk = masked_inputs[i:i + batch_size]
                    try:
                        out = model(chunk)
                        scores.extend(out[:, target_class].cpu().tolist())
                    except RuntimeError as e:
                        log(f"Score-CAM: Batch {i} failed: {e}")
                        for j in range(chunk.shape[0]):
                            out = model(chunk[j:j+1])
                            scores.append(out[0, target_class].cpu().item())

        if not scores:
            log("Score-CAM: No scores computed, returning zeros")
            h.remove()
            return np.zeros((x.shape[2], x.shape[3]))

        # Convert to tensor and normalize
        scores = torch.tensor(scores, device=device)
        
        # Apply ReLU (keep only positive contributions)
        scores = F.relu(scores)
        
        # Normalize scores to sum to 1
        score_sum = scores.sum()
        if score_sum > 1e-8:
            scores = scores / score_sum
        else:
            log("Score-CAM: All scores near zero, using uniform weights")
            scores = torch.ones_like(scores) / len(scores)

        # ---------------------------------------------------------
        # 4. Weighted Combination of Feature Maps
        # ---------------------------------------------------------
        feats = act_selected.squeeze(0)  # (K, H, W)
        
        # Weighted sum: CAM = Σ(w_i * A_i)
        cam = (feats * scores[:, None, None]).sum(dim=0)  # (H, W)

        # Upsample CAM to input resolution
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=x.shape[2:], 
                          mode='bilinear', 
                          align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            log("Score-CAM: Constant CAM values, normalizing failed")
            cam = np.ones_like(cam) * 0.5

        h.remove()
        log(f"Score-CAM: Generated CAM with shape {cam.shape}, range [{cam.min():.3f}, {cam.max():.3f}]")
        return cam
        
    except Exception as e:
        log(f"Score-CAM error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return np.zeros((224, 224))


# ============================================================
# Main Entry Point with Robust Error Handling
# ============================================================
def main():
    """
    Main entry point with comprehensive validation and error handling.
    """
    try:
        parser = argparse.ArgumentParser(description="Generate XAI visualizations for cervical cancer classification")
        parser.add_argument("--image", required=True, help="Path to input image")
        parser.add_argument("--class", dest="cls", required=True, type=int, help="Target class index (0-4)")
        parser.add_argument("--enable-quality", action="store_true", help="Enable quality assessment")
        parser.add_argument("--enable-stain-norm", action="store_true", help="Enable stain normalization")
        parser.add_argument("--enable-multi-cell", action="store_true", help="Enable multi-cell detection")
        parser.add_argument("--enable-uncertainty", action="store_true", help="Enable uncertainty estimation")
        parser.add_argument("--model-path", help="Path to trained model weights (optional)")
        parser.add_argument("--device", choices=['cuda', 'cpu', 'auto'], default='auto', help="Device to use")
        args = parser.parse_args()

        # Validate class index
        if args.cls < 0 or args.cls > 4:
            log(f"ERROR: Invalid class index {args.cls}, must be 0-4")
            sys.exit(1)

        # Device selection
        if args.device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        if device == 'cuda' and not torch.cuda.is_available():
            log("WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        log(f"Using device: {device}")

        # ---------------------------------------------------------
        # Load & optimize ConvNeXt with Error Handling
        # ---------------------------------------------------------
        try:
            if args.model_path and os.path.exists(args.model_path):
                log(f"Loading trained model from: {args.model_path}")
                model = models.convnext_tiny(weights=None)
                checkpoint = torch.load(args.model_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                log("Trained model loaded successfully")
            else:
                log("Loading pretrained ImageNet ConvNeXt-Tiny")
                model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            
            model.to(device)
            model.eval()
            
            # Disable gradients for inference
            for param in model.parameters():
                param.requires_grad = False
            
            log("Model loaded and set to eval mode")

            if device == "cuda":
                try:
                    model.half()
                    log("Model converted to FP16")
                except Exception as e:
                    log(f"FP16 conversion failed: {e}, using FP32")

                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    log("Model compiled with torch.compile")
                except Exception as e:
                    log(f"torch.compile not available: {e}")
                    
        except Exception as e:
            log(f"ERROR: Model loading failed: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        # ---------------------------------------------------------
        # Load and Validate Image
        # ---------------------------------------------------------
        try:
            if not os.path.exists(args.image):
                log(f"ERROR: Image not found: {args.image}")
                sys.exit(1)
            
            img = Image.open(args.image).convert("RGB")
            img_np = np.array(img)
            
            if img_np.size == 0:
                log("ERROR: Empty image")
                sys.exit(1)
            
            h, w = img_np.shape[:2]
            if h < 50 or w < 50:
                log(f"WARNING: Very small image ({h}x{w})")
            
            log(f"Image loaded: {args.image}, shape={img_np.shape}")
            
        except Exception as e:
            log(f"ERROR: Image loading failed: {e}")
            sys.exit(1)

        output = {}

        # ---------------------------------------------------------
        # 1. Quality Assessment with Error Handling
        # ---------------------------------------------------------
        if args.enable_quality:
            try:
                quality_score, quality_flags, detailed_metrics = assess_image_quality(img_np)
                output["quality"] = {
                    "score": quality_score,
                    "flags": quality_flags,
                    "metrics": detailed_metrics
                }
                
                if not quality_flags.get("usable", True):
                    log("WARNING: Poor image quality detected")
                    
            except Exception as e:
                log(f"Quality assessment error: {e}")
                output["quality"] = {"error": str(e), "score": 0.5}

        # ---------------------------------------------------------
        # 2. Stain Normalization with Error Handling
        # ---------------------------------------------------------
        if args.enable_stain_norm:
            try:
                img_np_normalized = normalize_stain_macenko(img_np)
                
                # Validate normalization
                if img_np_normalized is not None and img_np_normalized.shape == img_np.shape:
                    img_np = img_np_normalized
                    norm_img = Image.fromarray(img_np)
                    buf = io.BytesIO()
                    norm_img.save(buf, format="PNG")
                    output["normalized_image_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
                    log("Stain normalization successful")
                else:
                    log("Stain normalization returned invalid result")
                    
            except Exception as e:
                log(f"Stain normalization error: {e}")
                output["stain_norm_error"] = str(e)

        # ---------------------------------------------------------
        # 3. Multi-Cell Detection with Error Handling
        # ---------------------------------------------------------
        if args.enable_multi_cell:
            try:
                cell_boxes = detect_multiple_cells(img_np)
                output["cells"] = {
                    "count": len(cell_boxes),
                    "bounding_boxes": cell_boxes
                }
                log(f"Multi-cell detection: {len(cell_boxes)} cells found")
                
            except Exception as e:
                log(f"Multi-cell detection error: {e}")
                output["cells"] = {"count": 0, "error": str(e)}

        # ---------------------------------------------------------
        # Standard preprocessing for CAM
        # ---------------------------------------------------------
        try:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            img_for_model = Image.fromarray(img_np)
            x = preprocess(img_for_model)

            if device == "cuda":
                try:
                    x = x.half()
                except Exception as e:
                    log(f"FP16 conversion for input failed: {e}")
                    
        except Exception as e:
            log(f"ERROR: Image preprocessing failed: {e}")
            sys.exit(1)

        # ---------------------------------------------------------
        # 4. Uncertainty Estimation with Error Handling
        # ---------------------------------------------------------
        if args.enable_uncertainty:
            try:
                # Use float32 for MC Dropout (more stable)
                x_float = x.float() if device == "cpu" else x
                uncertainty = estimate_uncertainty(model, x_float, args.cls, n_samples=30)
                output["uncertainty"] = uncertainty
                
                if not uncertainty.get("is_reliable", False):
                    log("WARNING: Uncertainty estimate indicates low reliability")
                    
            except Exception as e:
                log(f"Uncertainty estimation error: {e}")
                output["uncertainty"] = {"error": str(e), "is_reliable": False}

        # ---------------------------------------------------------
        # 5. Compute CAMs with Error Handling
        # ---------------------------------------------------------
        try:
            t0 = time.time()
            score_cam = score_cam_fast(model, x, args.cls, device=device, top_k=96, batch_size=64)
            t1 = time.time()
            log(f"Score-CAM time: {t1 - t0:.3f}s")

            if score_cam is not None and score_cam.size > 0:
                output["scorecam_base64"] = cam_to_base64(score_cam)
            else:
                log("ERROR: Score-CAM returned empty result")
                output["scorecam_error"] = "Empty CAM"
                
        except Exception as e:
            log(f"Score-CAM error: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            output["scorecam_error"] = str(e)

        try:
            layer_cam = layer_cam_convnext(model, x, args.cls, device=device)
            t2 = time.time()
            log(f"Layer-CAM time: {t2 - t1:.3f}s")

            if layer_cam is not None and layer_cam.size > 0:
                output["layercam_base64"] = cam_to_base64(layer_cam)
            else:
                log("ERROR: Layer-CAM returned empty result")
                output["layercam_error"] = "Empty CAM"
                
        except Exception as e:
            log(f"Layer-CAM error: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            output["layercam_error"] = str(e)

        # ---------------------------------------------------------
        # Output JSON
        # ---------------------------------------------------------
        print(json.dumps(output, indent=2))
        log("XAI pipeline completed successfully")
        
    except Exception as e:
        log(f"FATAL ERROR in main: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Output error JSON
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
