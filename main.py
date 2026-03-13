"""
Leaf Disease Image Preprocessing Pipeline
------------------------------------------
This script preprocesses a leaf image through a series of stages before
passing it to a disease classification model. The pipeline includes:
    1. Loading the original image
    2. Validating image quality (blur, brightness, leaf coverage)
    3. Sharpening mildly blurry images (if needed)
    4. Removing the background (isolating the leaf)
    5. Reducing noise
    6. Resizing and normalizing for model input

All intermediate outputs are saved to the 'debug_output/' directory for inspection.
"""

from PIL import Image
import numpy as np
import cv2
import os

# Path to the input leaf image
IMAGE_PATH = "Algal-Leaf-Spot-Symptoms.jpg"

# Directory where all intermediate debug images will be saved
os.makedirs("debug_output", exist_ok=True)

# ---------------------------------------------------------------------------
# Blur score thresholds (computed using Laplacian variance).
# Higher score = sharper image; lower score = blurrier image.
# ---------------------------------------------------------------------------
BLUR_REJECT  = 5    # Images scoring below this are too blurry to recover; reject them outright
BLUR_SHARPEN = 50   # Images scoring between BLUR_REJECT and this are mildly blurry; apply sharpening
                    # Images scoring above BLUR_SHARPEN are already sharp enough; skip sharpening


def validate(img_array):
    """
    Validates the quality of an input image before processing.

    Checks performed:
        - Blur: Uses the variance of the Laplacian to detect blurriness.
                A low variance means the image lacks sharp edges (blurry).
        - Brightness: Measures average pixel intensity on the grayscale image.
                      Very dark or overexposed images are rejected.
        - Leaf presence: Uses HSV color masking to check whether the image
                         contains enough green/plant-colored content to be
                         considered a valid leaf photo.

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        dict: A result dictionary containing:
              - 'valid' (bool): Whether the image passed all checks.
              - 'reason' (str): Rejection reason if valid=False.
              - 'blur_score' (float): Laplacian variance score.
              - 'needs_sharpen' (bool): True if blur score is between thresholds.
              - 'brightness' (float): Mean grayscale pixel intensity (0-255).
              - 'leaf_coverage' (float): Fraction of pixels identified as leaf/plant.
    """
    # Convert to grayscale for blur and brightness checks
    gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Laplacian variance: high = sharp edges present, low = blurry
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Mean intensity: 0 = black, 255 = white
    brightness = float(gray.mean())

    # Reject images that are too blurry to be useful
    if blur < BLUR_REJECT:
        return {"valid": False, "reason": "Image is too blurry. Please upload a clearer photo."}

    # Reject images that are too dark (likely unlit or underexposed)
    if brightness < 10:
        return {"valid": False, "reason": "Image is too dark. Please upload a well-lit photo."}

    # Reject images that are overexposed (washed out, no usable detail)
    if brightness > 250:
        return {"valid": False, "reason": "Image is overexposed. Please upload a better photo."}

    # Convert to HSV for color-based leaf detection
    hsv     = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Separate the R, G, B channels for green-dominance check
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total   = img_array.shape[0] * img_array.shape[1]

    # Ratio of pixels where green channel dominates both red and blue
    g_ratio = float(np.sum((g > r) & (g > b)) / total)

    # Define HSV masks for various plant-related colors:
    #   - Green hues (leaf body)
    #   - Yellow-green / olive tones (aged or diseased areas)
    #   - Dark red-orange hues (disease spots or edges)
    #   - Wraparound reds in HSV (hue near 0 and 180 both represent red)
    masks = [
        cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255])),  # Green range
        cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230])),  # Yellow-green
        cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230])),  # Low-hue red
        cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230])),  # High-hue red wrap
    ]

    # Merge all masks into a single combined mask
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # Fraction of pixels matching any plant-like color
    coverage = float(np.sum(combined > 0) / total)

    # Reject if neither green-dominant pixels nor plant-colored pixels are found
    if g_ratio < 0.03 and coverage < 0.03:
        return {"valid": False, "reason": "No leaf detected."}

    return {
        "valid":         True,
        "blur_score":    round(blur, 2),
        "needs_sharpen": blur < BLUR_SHARPEN,  # Flag for the sharpening stage
        "brightness":    round(brightness, 2),
        "leaf_coverage": round(coverage, 2),
    }


def sharpen_image(img_array: np.ndarray) -> np.ndarray:
    """
    Sharpens a mildly blurry image using the Unsharp Masking technique.

    How it works:
        1. A slightly blurred version of the image is created using a Gaussian blur.
        2. The difference between the original and the blurred version is the
           high-frequency detail (edges and textures).
        3. That detail is added back to the original at a weighted strength,
           effectively amplifying the edges and making the image appear sharper.

    This method preserves color fidelity because it sharpens based on
    luminance contrast rather than shifting hue or saturation values.

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        np.ndarray: Sharpened image as an RGB NumPy array.
    """
    # Create a blurred version to extract the low-frequency (soft) component
    blurred = cv2.GaussianBlur(img_array, (0, 0), sigmaX=2)

    # Blend: 1.5x original minus 0.5x blurred = original + 0.5x (original - blurred)
    # This adds high-frequency edge detail back on top of the image
    sharpened = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)

    return sharpened


def remove_background(img_array: np.ndarray) -> np.ndarray:
    """
    Isolates the leaf from the background by creating a color-based mask.

    Process:
        1. Converts the image to HSV color space for robust color segmentation.
        2. Builds masks for green and plant-adjacent colors to identify the leaf.
        3. Applies morphological operations to fill gaps and smooth the mask edges.
        4. Keeps only the largest detected contour (the main leaf body).
        5. Fills non-leaf pixels with white (255, 255, 255) to produce a clean,
           white-background image suitable for model input.

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        np.ndarray: Image with background replaced by white pixels.
    """
    # Convert to HSV for color-based leaf segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Build color masks for leaf/plant pixels (same color ranges as validation)
    masks = [
        cv2.inRange(hsv, np.array([10, 15, 15]), np.array([110, 255, 255])),  # Green
        cv2.inRange(hsv, np.array([3,  15, 15]), np.array([30,  255, 220])),  # Yellow-green
        cv2.inRange(hsv, np.array([0,  15, 15]), np.array([3,   255, 220])),  # Low-hue red
        cv2.inRange(hsv, np.array([170,15, 15]), np.array([180, 255, 220])),  # High-hue red wrap
    ]

    # Combine all masks into one unified leaf mask
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    # Morphological closing: fills small gaps and holes within the leaf region
    # A large elliptical kernel (20x20) ensures broad gap-filling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilation followed by erosion (a mild open/close cycle) to smooth mask edges
    # and recover any leaf pixels missed by the initial color mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask         = cv2.dilate(mask, kernel_small, iterations=2)
    mask         = cv2.erode(mask,  kernel_small, iterations=1)

    # Find external contours in the mask; keep only the largest one (the main leaf)
    # This discards small isolated blobs from the background or noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean_mask = np.zeros_like(mask)
        largest    = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean_mask

    # Start with an all-white canvas the same size as the input image
    result = np.full_like(img_array, 255)

    # Copy leaf pixels from the original; background pixels remain white
    result[mask > 0] = img_array[mask > 0]

    return result


def remove_noise(img_array: np.ndarray) -> np.ndarray:
    """
    Reduces noise in the image using a bilateral filter.

    A bilateral filter smooths the image while preserving edges, unlike a
    simple Gaussian blur which blurs edges as well. It considers both the
    spatial distance between pixels and the difference in their color values,
    so pixels that are spatially close AND similarly colored get blended,
    while pixels across sharp edges (different colors) are left distinct.

    Parameters used:
        d=5         : Diameter of each pixel's neighborhood for filtering
        sigmaColor=30 : Color tolerance — how different colors can be and still
                        be blended together (lower = more edge-preserving)
        sigmaSpace=30 : Spatial tolerance — how far apart pixels can be and
                        still influence each other

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        np.ndarray: Noise-reduced image as an RGB NumPy array.
    """
    return cv2.bilateralFilter(img_array, d=5, sigmaColor=30, sigmaSpace=30)


# =============================================================================
# MAIN PIPELINE — runs each preprocessing stage in order and saves debug images
# =============================================================================

print("=" * 50)
print("  PREPROCESSING DEBUG -- exact main.py pipeline")
print("=" * 50)

# ---------------------------------------------------------------------------
# Stage 1: Load the original image
# ---------------------------------------------------------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
arr = np.array(img)

# Save the unmodified original for reference
Image.fromarray(arr).save("debug_output/stage1_original.jpg")
print(f"\nStage 1 -- Original loaded: {img.size}")

# ---------------------------------------------------------------------------
# Stage 2: Validate image quality
# ---------------------------------------------------------------------------
quality = validate(arr)

status_label = "PASS" if quality["valid"] else "FAIL"
print(f"\nStage 2 -- Validation: {status_label}")
print(f"   blur={quality.get('blur_score')}  brightness={quality.get('brightness')}  leaf_coverage={quality.get('leaf_coverage')}")
print(f"   needs_sharpen={quality.get('needs_sharpen')}  (threshold: {BLUR_REJECT} reject / {BLUR_SHARPEN} sharpen)")

# Stop the pipeline entirely if the image does not meet quality requirements
if not quality["valid"]:
    print(f"   REJECTED: {quality['reason']}")
    exit()

# ---------------------------------------------------------------------------
# Stage 3: Sharpening (only if the image is mildly blurry)
# ---------------------------------------------------------------------------
if quality.get("needs_sharpen"):
    arr = sharpen_image(arr)
    Image.fromarray(arr).save("debug_output/stage3_sharpened.jpg")

    # Re-measure blur after sharpening to confirm the improvement
    gray_after = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur_after = cv2.Laplacian(gray_after, cv2.CV_64F).var()

    print(f"\nStage 3 -- Sharpening applied")
    print(f"   blur before: {quality['blur_score']}  ->  blur after: {round(blur_after, 2)}")
    print(f"   Saved: debug_output/stage3_sharpened.jpg")
else:
    print(f"\nStage 3 -- Skipped (image already sharp, blur={quality['blur_score']})")

# ---------------------------------------------------------------------------
# Stage 4: Background removal
# ---------------------------------------------------------------------------
arr_bg = remove_background(arr)
Image.fromarray(arr_bg).save("debug_output/stage4_bg_removed.jpg")

# Generate a semi-transparent mask overlay for visual debugging.
# This shows which pixels were identified as leaf (highlighted in green).
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
masks = [
    cv2.inRange(hsv, np.array([10, 15, 15]), np.array([110, 255, 255])),
    cv2.inRange(hsv, np.array([3,  15, 15]), np.array([30,  255, 220])),
    cv2.inRange(hsv, np.array([0,  15, 15]), np.array([3,   255, 220])),
    cv2.inRange(hsv, np.array([170,15, 15]), np.array([180, 255, 220])),
]
mask = masks[0]
for m in masks[1:]:
    mask = cv2.bitwise_or(mask, m)

# Close gaps in the debug mask (same operation as in remove_background)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Create a green-tinted layer to represent the detected leaf area
mask_rgb           = np.zeros_like(arr)
mask_rgb[mask > 0] = [0, 200, 0]

# Blend the original image with the green mask layer at 40% transparency
overlay = Image.blend(
    Image.fromarray(arr),
    Image.fromarray(mask_rgb.astype(np.uint8)),
    alpha=0.4
)
overlay.save("debug_output/stage4_mask_overlay.jpg")

print(f"\nStage 4 -- Background removed")
print(f"   Saved: debug_output/stage4_bg_removed.jpg")
print(f"   Saved: debug_output/stage4_mask_overlay.jpg")

# ---------------------------------------------------------------------------
# Stage 5: Noise reduction
# ---------------------------------------------------------------------------
arr_denoised = remove_noise(arr_bg)
Image.fromarray(arr_denoised).save("debug_output/stage5_denoised.jpg")
print(f"\nStage 5 -- Noise reduction done")
print(f"   Saved: debug_output/stage5_denoised.jpg")

# ---------------------------------------------------------------------------
# Stage 6: Resize and normalize for model input
# ---------------------------------------------------------------------------
# Resize to the model's expected input resolution (256x256)
final_img = Image.fromarray(arr_denoised).resize((256, 256), Image.LANCZOS)

# Normalize pixel values from [0, 255] to [0.0, 1.0] (float32)
arr_norm = np.array(final_img, dtype=np.float32) / 255.0

# Save a uint8 version (scaled back to 0-255) for visual inspection
Image.fromarray((arr_norm * 255).astype(np.uint8)).save("debug_output/stage6_final.jpg")

print(f"\nStage 6 -- Final model input ready")
print(f"   shape={arr_norm.shape}  min={arr_norm.min():.3f}  max={arr_norm.max():.3f}  mean={arr_norm.mean():.3f}")

# ---------------------------------------------------------------------------
# Open all saved debug images for visual inspection
# ---------------------------------------------------------------------------
print("\nOpening all debug images...")

# Always show original and post-background-removal stages
images_to_open = ["stage1_original"]

# Only show the sharpened image if sharpening was actually applied
if quality.get("needs_sharpen"):
    images_to_open.append("stage3_sharpened")

images_to_open += ["stage4_mask_overlay", "stage4_bg_removed", "stage5_denoised", "stage6_final"]

for f in images_to_open:
    Image.open(f"debug_output/{f}.jpg").show()

print("\nAll debug images saved to debug_output/")