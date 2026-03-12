import os
os.environ["U2NET_HOME"] = os.path.expanduser("~/.u2net")  # local path

from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2

IMAGE_PATH = "Black-Spot-Diplocarpon-rosae-GettyImages-1097545284.jpg"  # ← change this

os.makedirs("debug_output", exist_ok=True)

# ── exact copy of main.py validate() ─────────────────────────────────────────
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
    r,g,b   = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total   = img_array.shape[0] * img_array.shape[1]
    g_ratio = float(np.sum((g>r)&(g>b)) / total)
    masks   = [
        cv2.inRange(hsv, np.array([10,10,10]), np.array([110,255,255])),
        cv2.inRange(hsv, np.array([3, 10,10]), np.array([30, 255,230])),
        cv2.inRange(hsv, np.array([0, 10,10]), np.array([3,  255,230])),
        cv2.inRange(hsv, np.array([170,10,10]),np.array([180,255,230])),
    ]
    combined = masks[0]
    for m in masks[1:]: combined = cv2.bitwise_or(combined, m)
    coverage = float(np.sum(combined>0)/total)
    if g_ratio < 0.03 and coverage < 0.03:
        return {"valid": False, "reason": "No leaf detected"}
    return {"valid": True, "blur_score": round(blur,2),
            "brightness": round(brightness,2), "leaf_coverage": round(coverage,2)}

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 50)
print("  PREPROCESSING DEBUG")
print("=" * 50)

# Stage 1 — Original
img = Image.open(IMAGE_PATH).convert("RGB")
img.save("debug_output/stage1_original.jpg")
print(f"\n✅ Stage 1 — Original: {img.size}")

# Stage 2 — Validation
quality = validate(np.array(img))
print(f"\n{'✅' if quality['valid'] else '❌'} Stage 2 — Validation")
print(f"   blur={quality.get('blur_score')}  brightness={quality.get('brightness')}  leaf_coverage={quality.get('leaf_coverage')}")
if not quality["valid"]:
    print(f"   ❌ REJECTED: {quality['reason']}")
    exit()

# Stage 3 — Leaf detection overlay (visual only)
img_array  = np.array(img)
hsv        = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
masks      = [
    cv2.inRange(hsv, np.array([10,10,10]), np.array([110,255,255])),
    cv2.inRange(hsv, np.array([3, 10,10]), np.array([30, 255,230])),
    cv2.inRange(hsv, np.array([0, 10,10]), np.array([3,  255,230])),
    cv2.inRange(hsv, np.array([170,10,10]),np.array([180,255,230])),
]
combined = masks[0]
for m in masks[1:]: combined = cv2.bitwise_or(combined, m)
mask_rgb             = np.zeros_like(img_array)
mask_rgb[combined>0] = [0, 200, 0]
overlay = Image.blend(img, Image.fromarray(mask_rgb.astype(np.uint8)), alpha=0.4)
overlay.save("debug_output/stage3_leaf_overlay.jpg")
print(f"\n✅ Stage 3 — Leaf overlay saved: debug_output/stage3_leaf_overlay.jpg")

# Stage 4 — Background removal
print(f"\n⏳ Stage 4 — Background removal (u2netp)...")
rembg_session = new_session("u2netp")
try:
    rgba = remove(img, session=rembg_session)
    bg   = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    img  = bg
    print(f"✅ Stage 4 — Background removed")
except Exception as e:
    print(f"⚠️  Stage 4 — rembg failed ({e}), using original")
img.save("debug_output/stage4_bg_removed.jpg")

# Stage 5 — Noise removal
img = Image.fromarray(cv2.bilateralFilter(np.array(img), d=5, sigmaColor=30, sigmaSpace=30))
img.save("debug_output/stage5_denoised.jpg")
print(f"✅ Stage 5 — Noise removal done")

# Stage 6 — Resize + normalize
img_resized = img.resize((256, 256), Image.LANCZOS)
arr         = np.array(img_resized, dtype=np.float32) / 255.0
final       = Image.fromarray((arr * 255).astype(np.uint8))
final.save("debug_output/stage6_final.jpg")
print(f"\n✅ Stage 6 — Final model input")
print(f"   shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")

# Open all
print("\nOpening all images...")
for f in ["stage1_original","stage3_leaf_overlay","stage4_bg_removed","stage5_denoised","stage6_final"]:
    Image.open(f"debug_output/{f}.jpg").show()

print("\nAll saved in debug_output/")