from PIL import Image
import numpy as np
import cv2
import os

IMAGE_PATH = "your_leaf.jpg"  # ← change this

os.makedirs("debug_output", exist_ok=True)

def validate(img_array):
    gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    if blur < 5:
        return {"valid": False, "reason": "Too blurry"}
    if brightness < 10:
        return {"valid": False, "reason": "Too dark"}
    if brightness > 250:
        return {"valid": False, "reason": "Overexposed"}
    hsv     = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total   = img_array.shape[0] * img_array.shape[1]
    g_ratio = float(np.sum((g > r) & (g > b)) / total)
    masks   = [
        cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255])),
        cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230])),
        cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230])),
        cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230])),
    ]
    combined = masks[0]
    for m in masks[1:]: combined = cv2.bitwise_or(combined, m)
    coverage = float(np.sum(combined > 0) / total)
    if g_ratio < 0.03 and coverage < 0.03:
        return {"valid": False, "reason": "No leaf detected"}
    return {"valid": True, "blur_score": round(blur,2),
            "brightness": round(brightness,2), "leaf_coverage": round(coverage,2)}


def remove_background(img_array):
    hsv   = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    masks = [
        cv2.inRange(hsv, np.array([10, 15, 15]), np.array([110, 255, 255])),
        cv2.inRange(hsv, np.array([3,  15, 15]), np.array([30,  255, 220])),
        cv2.inRange(hsv, np.array([0,  15, 15]), np.array([3,   255, 220])),
        cv2.inRange(hsv, np.array([170,15, 15]), np.array([180, 255, 220])),
    ]
    mask = masks[0]
    for m in masks[1:]: mask = cv2.bitwise_or(mask, m)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask         = cv2.dilate(mask, kernel_small, iterations=2)
    mask         = cv2.erode(mask,  kernel_small, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean_mask = np.zeros_like(mask)
        largest    = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean_mask

    result           = np.full_like(img_array, 255)
    result[mask > 0] = img_array[mask > 0]
    return result


def remove_noise(img_array):
    return cv2.bilateralFilter(img_array, d=5, sigmaColor=30, sigmaSpace=30)


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 50)
print("  PREPROCESSING DEBUG — exact main.py pipeline")
print("=" * 50)

# Stage 1 — Original
img = Image.open(IMAGE_PATH).convert("RGB")
arr = np.array(img)
Image.fromarray(arr).save("debug_output/stage1_original.jpg")
print(f"\n✅ Stage 1 — Original: {img.size}")

# Stage 2 — Validation
quality = validate(arr)
print(f"\n{'✅' if quality['valid'] else '❌'} Stage 2 — Validation")
print(f"   blur={quality.get('blur_score')}  brightness={quality.get('brightness')}  leaf_coverage={quality.get('leaf_coverage')}")
if not quality["valid"]:
    print(f"   ❌ REJECTED: {quality['reason']}")
    exit()

# Stage 3 — Background removal (save mask overlay too)
arr_bg = remove_background(arr)
Image.fromarray(arr_bg).save("debug_output/stage3_bg_removed.jpg")

# Also save mask overlay for debugging
hsv   = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
masks = [
    cv2.inRange(hsv, np.array([10, 15, 15]), np.array([110, 255, 255])),
    cv2.inRange(hsv, np.array([3,  15, 15]), np.array([30,  255, 220])),
    cv2.inRange(hsv, np.array([0,  15, 15]), np.array([3,   255, 220])),
    cv2.inRange(hsv, np.array([170,15, 15]), np.array([180, 255, 220])),
]
mask = masks[0]
for m in masks[1:]: mask = cv2.bitwise_or(mask, m)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_rgb             = np.zeros_like(arr)
mask_rgb[mask > 0]   = [0, 200, 0]
overlay = Image.blend(img, Image.fromarray(mask_rgb.astype(np.uint8)), alpha=0.4)
overlay.save("debug_output/stage3_mask_overlay.jpg")
print(f"\n✅ Stage 3 — Background removed + mask overlay saved")

# Stage 4 — Noise removal
arr_denoised = remove_noise(arr_bg)
Image.fromarray(arr_denoised).save("debug_output/stage4_denoised.jpg")
print(f"✅ Stage 4 — Noise removal done")

# Stage 5 — Resize + normalize
final_img = Image.fromarray(arr_denoised).resize((256, 256), Image.LANCZOS)
arr_norm  = np.array(final_img, dtype=np.float32) / 255.0
Image.fromarray((arr_norm * 255).astype(np.uint8)).save("debug_output/stage5_final.jpg")
print(f"\n✅ Stage 5 — Final model input")
print(f"   shape={arr_norm.shape}  min={arr_norm.min():.3f}  max={arr_norm.max():.3f}  mean={arr_norm.mean():.3f}")

# Open all
print("\nOpening all images...")
for f in ["stage1_original", "stage3_mask_overlay", "stage3_bg_removed",
          "stage4_denoised", "stage5_final"]:
    Image.open(f"debug_output/{f}.jpg").show()

print("\nAll saved in debug_output/")