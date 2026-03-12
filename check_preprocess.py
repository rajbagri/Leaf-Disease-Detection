from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2
import os

# ── change this to your image path ──
IMAGE_PATH = "Black-Spot-Diplocarpon-rosae-GettyImages-1097545284.jpg"

os.makedirs("debug_output", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  EXACT COPY OF main.py FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_image(img_array):
    gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    if blur_score < 5:
        return {"valid": False, "reason": "Image is too blurry."}
    if brightness < 10:
        return {"valid": False, "reason": "Image is too dark."}
    if brightness > 250:
        return {"valid": False, "reason": "Image is overexposed."}
    return {
        "valid":      True,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
    }


def validate_is_leaf(img_array):
    r, g, b      = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio  = np.sum((g > r) & (g > b)) / total_pixels
    hsv        = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230]))
    red_mask1  = cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230]))
    red_mask2  = cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230]))
    combined      = cv2.bitwise_or(cv2.bitwise_or(green_mask, brown_mask),
                                   cv2.bitwise_or(red_mask1,  red_mask2))
    leaf_coverage = np.sum(combined > 0) / total_pixels
    if green_ratio < 0.03 and leaf_coverage < 0.03:
        return {"valid": False, "reason": "No leaf detected."}
    return {
        "valid":         True,
        "leaf_coverage": round(float(leaf_coverage), 2),
        "green_ratio":   round(float(green_ratio), 2),
    }


def remove_background(img, rembg_session):
    try:
        rgba       = remove(img, session=rembg_session)
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    except Exception as e:
        print(f"  rembg failed: {e} — using original")
        return img.convert("RGB")


def remove_noise(img):
    arr      = np.array(img)
    denoised = cv2.bilateralFilter(arr, d=5, sigmaColor=30, sigmaSpace=30)
    return Image.fromarray(denoised)


def normalize(img):
    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (256,256,3) without batch dim for visualization


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE STEP BY STEP
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("  PREPROCESSING DEBUG — exact main.py pipeline")
print("=" * 55)

# ── Load image ────────────────────────────────────────────────────────────────
img = Image.open(IMAGE_PATH).convert("RGB")
img.save("debug_output/stage1_original.jpg")
print(f"\n✅ Stage 1 — Original")
print(f"   Size: {img.size}")
print(f"   Saved: debug_output/stage1_original.jpg")

# ── Step 1: Quality validation ────────────────────────────────────────────────
img_array = np.array(img)
quality   = validate_image(img_array)
print(f"\n{'✅' if quality['valid'] else '❌'} Stage 2 — Quality Check")
print(f"   Blur score:  {quality.get('blur_score')}  (needs > 5)")
print(f"   Brightness:  {quality.get('brightness')}  (needs 10-250)")
if not quality["valid"]:
    print(f"   ❌ REJECTED: {quality['reason']}")
    exit()
print(f"   Result: PASSED")

# ── Step 2: Leaf detection ────────────────────────────────────────────────────
leaf_check = validate_is_leaf(img_array)

# Save leaf detection overlay
hsv        = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
green_mask = cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255]))
brown_mask = cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230]))
red_mask1  = cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230]))
red_mask2  = cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230]))
combined   = cv2.bitwise_or(cv2.bitwise_or(green_mask, brown_mask),
                             cv2.bitwise_or(red_mask1,  red_mask2))
mask_rgb             = np.zeros_like(img_array)
mask_rgb[combined>0] = [0, 200, 0]
overlay = Image.blend(img, Image.fromarray(mask_rgb.astype(np.uint8)), alpha=0.4)
overlay.save("debug_output/stage3_leaf_detection_overlay.jpg")

print(f"\n{'✅' if leaf_check['valid'] else '❌'} Stage 3 — Leaf Detection")
print(f"   Green ratio:   {leaf_check.get('green_ratio')}  (needs > 0.03)")
print(f"   Leaf coverage: {leaf_check.get('leaf_coverage')}  (needs > 0.03)")
print(f"   Saved overlay: debug_output/stage3_leaf_detection_overlay.jpg")
if not leaf_check["valid"]:
    print(f"   ❌ REJECTED: {leaf_check['reason']}")
    exit()
print(f"   Result: PASSED")

# ── Step 3: Background removal ────────────────────────────────────────────────
print(f"\n⏳ Stage 4 — Background Removal (rembg u2netp) — loading model...")
rembg_session = new_session("u2netp")
img_no_bg     = remove_background(img, rembg_session)
img_no_bg.save("debug_output/stage4_background_removed.jpg")
print(f"✅ Stage 4 — Background Removed")
print(f"   Saved: debug_output/stage4_background_removed.jpg")

# ── Step 4: Noise removal ─────────────────────────────────────────────────────
img_denoised = remove_noise(img_no_bg)
img_denoised.save("debug_output/stage5_denoised.jpg")
print(f"\n✅ Stage 5 — Noise Removal (bilateral filter)")
print(f"   Saved: debug_output/stage5_denoised.jpg")

# ── Step 5: Normalize ─────────────────────────────────────────────────────────
arr = normalize(img_denoised)
final = Image.fromarray((arr * 255).astype(np.uint8))
final.save("debug_output/stage6_final_model_input.jpg")
print(f"\n✅ Stage 6 — Normalized (this is exactly what the model sees)")
print(f"   Size:       {final.size}")
print(f"   Pixel min:  {arr.min():.4f}")
print(f"   Pixel max:  {arr.max():.4f}")
print(f"   Pixel mean: {arr.mean():.4f}")
print(f"   Saved: debug_output/stage6_final_model_input.jpg")

# ── Open all images ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Opening all stage images...")
print("=" * 55)

Image.open("debug_output/stage1_original.jpg").show()
Image.open("debug_output/stage3_leaf_detection_overlay.jpg").show()
Image.open("debug_output/stage4_background_removed.jpg").show()
Image.open("debug_output/stage5_denoised.jpg").show()
Image.open("debug_output/stage6_final_model_input.jpg").show()

print("""
Saved images:
  stage1_original.jpg               — your original image
  stage3_leaf_detection_overlay.jpg — green = detected as leaf
  stage4_background_removed.jpg     — after rembg (background = white)
  stage5_denoised.jpg               — after bilateral filter
  stage6_final_model_input.jpg      — exactly what model receives
""")