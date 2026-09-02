import os
import rasterio
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def create_side_by_side(t1_path: str, t2_path: str, output_path: str) -> str:
    """
    Creates a side-by-side composite image from T1 and T2 raster files.
    Preserves aspect ratio and annotates BEFORE / AFTER.
    """
    if not HAS_PIL:
        # Fallback for environments without Pillow (e.g. testing with MockVLMClient)
        with open(output_path, "w") as f:
            f.write("DUMMY IMAGE")
        return output_path

    def load_grayscale(path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            # Simple normalization based on percentiles to handle outliers
            p2, p98 = np.percentile(arr[arr > 0], (2, 98)) if np.any(arr > 0) else (0, 1)
            if p98 == p2:
                p98 = p2 + 1
            arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            return arr

    t1_arr = load_grayscale(t1_path)
    t2_arr = load_grayscale(t2_path)
    
    # Resize T2 to match T1 height if they differ slightly
    h1, w1 = t1_arr.shape
    h2, w2 = t2_arr.shape
    
    img1 = Image.fromarray(t1_arr).convert("RGB")
    img2 = Image.fromarray(t2_arr).convert("RGB")
    
    if h1 != h2:
        img2 = img2.resize((int(w2 * (h1 / h2)), h1))
        
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    composite = Image.new("RGB", (w1 + w2, h1))
    composite.paste(img1, (0, 0))
    composite.paste(img2, (w1, 0))
    
    # Annotate
    draw = ImageDraw.Draw(composite)
    # Use default font if custom font not available
    # Text in top-left of each side
    draw.text((10, 10), "T1: BEFORE", fill=(255, 0, 0))
    draw.text((w1 + 10, 10), "T2: AFTER", fill=(255, 0, 0))
    
    composite.save(output_path)
    return output_path
