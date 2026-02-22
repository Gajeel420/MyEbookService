# Building Custom Datasets for 16-Bit Game Character Model Renderings

A step-by-step guide to extracting, processing, and structuring image datasets from classic 16-bit game character sprites for use in machine learning, upscaling, style transfer, and generative AI pipelines.

---

## Table of Contents

1. [Overview and Use Cases](#1-overview-and-use-cases)
2. [Prerequisites and Tools](#2-prerequisites-and-tools)
3. [Step 1 — Source Material and Legal Considerations](#step-1--source-material-and-legal-considerations)
4. [Step 2 — Extracting Sprites from ROM Files](#step-2--extracting-sprites-from-rom-files)
5. [Step 3 — Rendering Sprites to Clean PNGs](#step-3--rendering-sprites-to-clean-pngs)
6. [Step 4 — Removing Backgrounds and Isolating Characters](#step-4--removing-backgrounds-and-isolating-characters)
7. [Step 5 — Palette Normalization and Color Correction](#step-5--palette-normalization-and-color-correction)
8. [Step 6 — Generating Multi-Angle and Multi-Pose Renderings](#step-6--generating-multi-angle-and-multi-pose-renderings)
9. [Step 7 — Upscaling Sprites to Usable Resolution](#step-7--upscaling-sprites-to-usable-resolution)
10. [Step 8 — Augmenting the Dataset](#step-8--augmenting-the-dataset)
11. [Step 9 — Labeling and Annotation](#step-9--labeling-and-annotation)
12. [Step 10 — Structuring and Exporting the Final Dataset](#step-10--structuring-and-exporting-the-final-dataset)
13. [Appendix: Recommended Tools Reference](#appendix-recommended-tools-reference)

---

## 1. Overview and Use Cases

Classic 16-bit games (SNES, Sega Genesis/Mega Drive, Neo Geo, etc.) rendered characters as **sprite sheets** — grids of individual animation frames packed into a single image file. These sprite sheets are a rich source of stylized pixel art character data.

**Common reasons to build this kind of dataset:**

- Training AI upscalers (e.g., ESRGAN, Real-ESRGAN) on pixel art specifically
- Style transfer — mapping pixel art characters to 3D or painted art styles
- Generative models (Stable Diffusion fine-tuning, GANs) conditioned on retro character aesthetics
- Game asset reconstruction and remastering pipelines
- Academic research in low-resolution image processing
- Building reference libraries for artists and game developers

---

## 2. Prerequisites and Tools

### Software You Will Need

| Tool | Purpose | Platform |
|------|---------|----------|
| **Tile Molester** or **YY-CHR** | Sprite/tile extraction from ROMs | Windows/Linux (via Wine) |
| **Tile Layer Pro** | Alternative tile viewer | Windows |
| **ZSNES** or **bsnes** (with screenshot tools) | In-emulator rendering | Linux/Windows/macOS |
| **RetroArch** | Multi-system emulation, frame capture | All platforms |
| **Python 3.10+** | Scripting, automation, dataset pipelines | All platforms |
| **Pillow (PIL)** | Image processing in Python | All platforms |
| **OpenCV** | Background removal, contour detection | All platforms |
| **rembg** | AI-based background removal | All platforms |
| **GIMP** or **Aseprite** | Manual sprite editing | All platforms |
| **Aseprite** | Sprite sheet slicing and export | All platforms |

### Python Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pillow opencv-python numpy rembg torch torchvision
```

### Recommended Directory Layout

```
16bit-dataset/
├── roms/               # Source ROM files (keep private)
├── raw_sheets/         # Extracted sprite sheets (PNG)
├── raw_frames/         # Individual sliced frames (PNG)
├── clean_frames/       # Background-removed, isolated sprites
├── normalized/         # Palette-normalized sprites
├── upscaled/           # High-resolution versions
├── augmented/          # Augmented copies for training
├── annotations/        # JSON/CSV label files
└── final_dataset/      # Packaged and split dataset
    ├── train/
    ├── val/
    └── test/
```

---

## Step 1 — Source Material and Legal Considerations

### 1.1 Understanding the Legal Landscape

Before extracting any assets, understand the legal context for your use case:

- **Personal use and research** generally falls under fair use in many jurisdictions, but this varies by country.
- **Publishing a dataset** derived from commercial game assets can expose you to DMCA takedown or legal action.
- **Open-licensed games**: Several indie and open-source projects release pixel art characters under Creative Commons or MIT licenses — these are ideal for public datasets.
  - [OpenGameArt.org](https://opengameart.org) — Large library of freely licensed pixel art sprites
  - [Itch.io free assets](https://itch.io/game-assets/free) — Many free pixel art packs
  - [Liberated Pixel Cup assets](https://lpc.opengameart.org) — CC-BY-SA licensed

**Recommendation:** For any dataset you intend to publish or use commercially, source assets exclusively from open-licensed projects. For private research pipelines, consult legal counsel regarding your jurisdiction's fair use provisions.

### 1.2 Identifying Good Source Games

Ideal 16-bit games for character datasets share these traits:

- Rich animation sets (many frames per character)
- Distinct character designs (minimizes ambiguity for labeling)
- Consistent sprite sizing (easier to automate slicing)
- Diverse color palettes across characters

**Well-known examples** (characters with large sprite sets):
- Street Fighter II (SNES) — large fighters with many move frames
- Mortal Kombat (Sega Genesis) — dark palette, high contrast
- Chrono Trigger (SNES) — RPG overworld and battle sprites
- Sonic the Hedgehog series — simple silhouettes, high motion variety
- Mega Man X (SNES) — clean outlines, good for segmentation

---

## Step 2 — Extracting Sprites from ROM Files

### 2.1 Using YY-CHR (Tile/Pattern Editor)

YY-CHR reads the raw tile data encoded in ROM files. SNES and Genesis store character graphics as indexed tile patterns.

1. Open YY-CHR.
2. Go to **File → Open** and load your ROM file (`.sfc`, `.smc`, `.md`, `.bin`).
3. In the format selector, choose the appropriate color depth:
   - SNES: `4BPP SNES` (4 bits per pixel, 16 colors per tile)
   - Genesis: `4BPP Genesis`
4. Scroll through the tile viewer until you see character sprite data. It will look like scattered pixel art tiles.
5. Export visible sections using **File → Save BMP**.

> **Note:** YY-CHR shows raw tile data without palette applied. Colors will look wrong — palette application happens in a later step.

### 2.2 Using Tile Molester for Structured Extraction

Tile Molester (Java-based) is more powerful for structured sheet extraction:

```bash
java -jar tilemolester.jar
```

1. Open the ROM file.
2. Use the codec list to select the matching graphics format.
3. Navigate to sprite data offsets (look these up in ROM hacking wikis for the specific game on sites like [RHDN](https://www.romhacking.net)).
4. Select the tile region and export as PNG.

### 2.3 In-Emulator Frame Capture (RetroArch Method)

For games where sprites are difficult to extract directly (due to compression or encryption), emulator frame capture is an alternative:

1. Install RetroArch and load the appropriate core for your system.
2. Load the ROM.
3. Enable **Frame Delay** and navigate through all character animations manually.
4. Use RetroArch's built-in **Screenshot** (F8 by default) or set up a script to capture frames automatically.

**Automated capture script using RetroArch + xdotool (Linux):**

```bash
#!/bin/bash
# Capture a screenshot every 100ms for 10 seconds
# Adjust timing based on animation speed

OUTPUT_DIR="./raw_captures"
mkdir -p "$OUTPUT_DIR"

for i in $(seq 1 100); do
    # Send screenshot key to RetroArch window
    xdotool key --window $(xdotool search --name "RetroArch") F8
    sleep 0.1
done
```

---

## Step 3 — Rendering Sprites to Clean PNGs

### 3.1 Slicing Sprite Sheets with Python

Once you have raw sprite sheets, slice them into individual frames. Most 16-bit character sprites follow a regular grid layout.

```python
# slice_sheet.py
from PIL import Image
import os

def slice_sprite_sheet(
    sheet_path: str,
    output_dir: str,
    frame_width: int,
    frame_height: int,
    rows: int,
    cols: int,
    prefix: str = "frame"
):
    """
    Slice a uniform sprite sheet grid into individual frame images.

    Args:
        sheet_path:   Path to the sprite sheet PNG.
        output_dir:   Directory to save sliced frames.
        frame_width:  Pixel width of each frame.
        frame_height: Pixel height of each frame.
        rows:         Number of rows in the grid.
        cols:         Number of columns in the grid.
        prefix:       Filename prefix for output frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    sheet = Image.open(sheet_path).convert("RGBA")

    frame_index = 0
    for row in range(rows):
        for col in range(cols):
            left   = col * frame_width
            upper  = row * frame_height
            right  = left + frame_width
            lower  = upper + frame_height

            frame = sheet.crop((left, upper, right, lower))
            frame.save(os.path.join(output_dir, f"{prefix}_{frame_index:04d}.png"))
            frame_index += 1

    print(f"Saved {frame_index} frames to {output_dir}")


# Example: Street Fighter II Ryu sprite sheet (56x64 per frame, 8x6 grid)
slice_sprite_sheet(
    sheet_path="raw_sheets/ryu_sheet.png",
    output_dir="raw_frames/ryu",
    frame_width=56,
    frame_height=64,
    rows=6,
    cols=8,
    prefix="ryu"
)
```

### 3.2 Detecting Frame Dimensions Automatically

If the sprite sheet dimensions are unknown, use OpenCV to detect bounding boxes:

```python
# detect_frames.py
import cv2
import numpy as np
from PIL import Image

def detect_sprite_bounds(image_path: str, bg_color=(0, 0, 0, 0)):
    """
    Detect non-empty frame regions in a sprite sheet using contour detection.
    Returns a list of (x, y, w, h) bounding boxes.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Use alpha channel as mask if available, otherwise threshold on magenta/black
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
    else:
        # Common transparency color in 16-bit games: pure black or magenta
        lower = np.array([0, 0, 200])   # magenta lower bound (BGR)
        upper = np.array([10, 0, 255])  # magenta upper bound
        mask = cv2.inRange(img[:, :, :3], lower, upper)
        mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    return sorted(bounds, key=lambda b: (b[1], b[0]))  # sort top-left to bottom-right
```

---

## Step 4 — Removing Backgrounds and Isolating Characters

### 4.1 Transparency-Key Removal (Magenta/Black Background)

16-bit games use a fixed "transparent color" (commonly magenta `#FF00FF` or pure black `#000000`) as the background fill on sprite sheets. Remove it:

```python
# remove_bg_color.py
from PIL import Image
import numpy as np

def remove_transparency_key(image_path: str, output_path: str, key_color=(255, 0, 255)):
    """
    Replace a solid transparency-key color with full transparency (alpha=0).
    """
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

    # Build mask where pixel matches key color
    mask = (r == key_color[0]) & (g == key_color[1]) & (b == key_color[2])
    data[:,:,3][mask] = 0  # set alpha to 0 for matching pixels

    result = Image.fromarray(data, "RGBA")
    result.save(output_path, "PNG")
```

### 4.2 AI-Based Background Removal with rembg

For frames captured from emulator screenshots (where background is complex), use `rembg`:

```python
# remove_bg_ai.py
from rembg import remove
from PIL import Image
import io

def remove_background_ai(input_path: str, output_path: str):
    with open(input_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data)
    img = Image.open(io.BytesIO(output_data)).convert("RGBA")
    img.save(output_path)
```

> **Note:** `rembg` works well on photographic backgrounds but may over-remove pixel art edges. Use transparency-key removal as the primary method when a consistent key color is present.

### 4.3 Batch Processing All Frames

```python
# batch_remove_bg.py
import os
from remove_bg_color import remove_transparency_key

INPUT_DIR  = "raw_frames"
OUTPUT_DIR = "clean_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if not filename.endswith(".png"):
            continue
        input_path  = os.path.join(root, filename)
        rel_path    = os.path.relpath(root, INPUT_DIR)
        out_dir     = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filename)
        remove_transparency_key(input_path, output_path)

print("Batch background removal complete.")
```

---

## Step 5 — Palette Normalization and Color Correction

16-bit systems used hardware-constrained color palettes (256 or 512 colors depending on the system). Emulators and extraction tools can introduce inconsistencies. Normalize palettes for a consistent dataset.

### 5.1 Extracting the Palette from a Frame

```python
# palette_utils.py
from PIL import Image

def extract_palette(image_path: str, max_colors: int = 16):
    """
    Extract the dominant colors from a sprite (excluding fully transparent pixels).
    Returns a sorted list of (R, G, B) tuples.
    """
    img = Image.open(image_path).convert("RGBA")
    pixels = list(img.getdata())

    # Exclude transparent pixels
    opaque = [(r, g, b) for r, g, b, a in pixels if a > 10]

    # Quantize to find dominant colors
    quantized = Image.new("RGB", (len(opaque), 1))
    quantized.putdata(opaque)
    quantized = quantized.quantize(colors=max_colors)

    palette_data = quantized.getpalette()
    colors = [
        (palette_data[i], palette_data[i+1], palette_data[i+2])
        for i in range(0, max_colors * 3, 3)
    ]
    return colors
```

### 5.2 Remapping Palette to a Target Reference

```python
# remap_palette.py
import numpy as np
from PIL import Image

def remap_palette(image_path: str, output_path: str, target_palette: list):
    """
    Remap each pixel to the nearest color in the target palette.
    Useful for standardizing palette variations across multiple ROM dumps.
    """
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    target = np.array(target_palette, dtype=np.float32)  # shape (N, 3)
    rgb = data[:, :, :3]
    alpha = data[:, :, 3]

    # Compute distance from each pixel to each palette color
    h, w, _ = rgb.shape
    flat_rgb = rgb.reshape(-1, 3)                                    # (H*W, 3)
    dists = np.linalg.norm(flat_rgb[:, None, :] - target[None, :, :], axis=2)  # (H*W, N)
    nearest_idx = np.argmin(dists, axis=1)                           # (H*W,)
    remapped = target[nearest_idx].reshape(h, w, 3).astype(np.uint8)

    result_data = np.dstack([remapped, alpha.astype(np.uint8)])
    Image.fromarray(result_data, "RGBA").save(output_path)
```

---

## Step 6 — Generating Multi-Angle and Multi-Pose Renderings

Most 16-bit character sprites only face left and right. To build a richer dataset, generate additional variations.

### 6.1 Horizontal Flip (Mirror)

```python
# generate_flips.py
from PIL import Image
import os

def generate_flipped(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue
        img = Image.open(os.path.join(input_dir, filename))
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        base, ext = os.path.splitext(filename)
        flipped.save(os.path.join(output_dir, f"{base}_flipped{ext}"))
```

### 6.2 Isometric Projection Simulation

For training 3D-aware models, simulate depth by skewing the sprite:

```python
# isometric_transform.py
from PIL import Image
import numpy as np

def apply_isometric_skew(image_path: str, output_path: str, angle_deg: float = 26.565):
    """
    Apply a shear transform to simulate an isometric viewing angle.
    angle_deg: Standard isometric angle is ~26.565 degrees.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # Shear matrix coefficients
    shear = np.tan(np.radians(angle_deg))

    # PIL affine transform: (a, b, c, d, e, f) maps (x,y) → (ax+by+c, dx+ey+f)
    # Horizontal shear: x' = x + shear*y
    transform = (1, shear, -shear * h / 2, 0, 1, 0)

    new_w = int(w + abs(shear) * h)
    skewed = img.transform(
        (new_w, h),
        Image.AFFINE,
        transform,
        resample=Image.NEAREST  # use NEAREST to preserve pixel art crispness
    )
    skewed.save(output_path)
```

> **Important:** Always use `Image.NEAREST` resampling when transforming pixel art to avoid interpolation blur.

### 6.3 Grouping Frames by Animation State

Organize frames by semantic category for more useful training labels:

```python
# organize_animations.py
# Example animation state mapping — customize per character/game
ANIMATION_MAP = {
    "idle":        range(0, 4),
    "walk":        range(4, 12),
    "run":         range(12, 18),
    "jump":        range(18, 22),
    "attack_light": range(22, 28),
    "attack_heavy": range(28, 36),
    "hurt":        range(36, 40),
    "death":       range(40, 46),
}

import shutil, os

def organize_by_animation(frames_dir: str, output_base: str, animation_map: dict):
    for anim_name, frame_range in animation_map.items():
        anim_dir = os.path.join(output_base, anim_name)
        os.makedirs(anim_dir, exist_ok=True)
        for i in frame_range:
            src = os.path.join(frames_dir, f"frame_{i:04d}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(anim_dir, os.path.basename(src)))
```

---

## Step 7 — Upscaling Sprites to Usable Resolution

Raw 16-bit sprites are tiny (often 16×16 to 64×64 pixels). Upscale them for use in modern pipelines.

### 7.1 Nearest-Neighbor Upscaling (Pixel Art Preserving)

```python
# upscale_nearest.py
from PIL import Image
import os

def upscale_nearest(input_dir: str, output_dir: str, scale: int = 4):
    """
    Upscale images using nearest-neighbor interpolation.
    Preserves pixel art hard edges. Common scales: 2x, 4x, 8x.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue
        img = Image.open(os.path.join(input_dir, filename))
        w, h = img.size
        upscaled = img.resize((w * scale, h * scale), Image.NEAREST)
        upscaled.save(os.path.join(output_dir, filename))
```

### 7.2 xBRZ Algorithm Upscaling (Smooth Edges)

For smoother upscaling that attempts to infer edge curves in pixel art, use the `xbrz` Python binding:

```bash
pip install xbrz
```

```python
# upscale_xbrz.py
import xbrz
from PIL import Image
import numpy as np
import os

def upscale_xbrz(input_path: str, output_path: str, scale: int = 4):
    """
    Upscale a sprite using the xBRZ algorithm.
    scale must be 2, 3, 4, 5, or 6.
    """
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    upscaled_data = xbrz.scale(data, scale)
    result = Image.fromarray(upscaled_data, "RGBA")
    result.save(output_path)
```

### 7.3 AI Upscaling with Real-ESRGAN (Pixel Art Model)

For the highest-quality upscaling trained specifically on pixel art:

```bash
pip install realesrgan basicsr facexlib gfpgan
```

```python
# upscale_realesrgan.py
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

def upscale_realesrgan(input_path: str, output_path: str, scale: int = 4):
    """
    Upscale using Real-ESRGAN with the pixel art model (RealESRGAN_x4plus_anime_6B).
    Download model weights from the Real-ESRGAN releases page.
    """
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=6, num_grow_ch=32, scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path="weights/RealESRGAN_x4plus_anime_6B.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=scale)
    cv2.imwrite(output_path, output)
```

---

## Step 8 — Augmenting the Dataset

Data augmentation multiplies your dataset size and improves model generalization.

```python
# augment_dataset.py
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import random

def augment_sprite(image_path: str, output_dir: str, n_augments: int = 5):
    """
    Generate N augmented variants of a sprite frame.
    Pixel-art safe: uses only transforms that don't introduce blur.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGBA")
    base = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(n_augments):
        augmented = img.copy()

        # 1. Random horizontal flip
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Random brightness shift (subtle, preserves palette feel)
        brightness_factor = random.uniform(0.85, 1.15)
        rgb_layer = augmented.convert("RGB")
        rgb_layer = ImageEnhance.Brightness(rgb_layer).enhance(brightness_factor)
        r, g, b = rgb_layer.split()
        _, _, _, a = augmented.split()
        augmented = Image.merge("RGBA", (r, g, b, a))

        # 3. Random hue rotation (simulate palette swaps / alt costumes)
        if random.random() > 0.6:
            data = np.array(augmented, dtype=np.float32)
            shift = random.randint(-20, 20)
            # Shift hue via HSV channel manipulation
            from PIL import ImageOps
            rgb_pil = Image.fromarray(data[:,:,:3].astype(np.uint8))
            hsv = np.array(rgb_pil.convert("HSV"), dtype=np.int32)
            hsv[:,:,0] = (hsv[:,:,0] + shift) % 256
            shifted = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
            rs, gs, bs = shifted.split()
            augmented = Image.merge("RGBA", (rs, gs, bs, augmented.split()[3]))

        # 4. Small pixel offsets (simulate sub-pixel jitter)
        dx, dy = random.randint(-2, 2), random.randint(-2, 2)
        augmented = augmented.transform(
            augmented.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
            resample=Image.NEAREST
        )

        out_name = f"{base}_aug_{i:03d}.png"
        augmented.save(os.path.join(output_dir, out_name))

    print(f"Generated {n_augments} augments for {base}")
```

**Augmentations to avoid for pixel art:**
- Gaussian blur (destroys the pixel structure)
- Bilinear/bicubic resizing
- Heavy distortion or perspective transforms
- JPEG compression (introduces artifacts)

---

## Step 9 — Labeling and Annotation

### 9.1 Flat Classification Labels (Character Name / Game)

```python
# generate_labels.py
import os
import json

def generate_flat_labels(dataset_root: str, output_path: str):
    """
    Walk a directory structured as:
      dataset_root/
        character_name/
          animation_state/
            frame_XXXX.png
    And produce a flat JSON label file.
    """
    records = []
    for char_name in os.listdir(dataset_root):
        char_dir = os.path.join(dataset_root, char_name)
        if not os.path.isdir(char_dir):
            continue
        for anim_state in os.listdir(char_dir):
            anim_dir = os.path.join(char_dir, anim_state)
            if not os.path.isdir(anim_dir):
                continue
            for filename in os.listdir(anim_dir):
                if not filename.endswith(".png"):
                    continue
                rel_path = os.path.relpath(
                    os.path.join(anim_dir, filename), dataset_root
                )
                records.append({
                    "file":       rel_path,
                    "character":  char_name,
                    "animation":  anim_state,
                    "game":       "street_fighter_2",  # customize per source
                    "system":     "snes",
                    "frame_idx":  int(filename.split("_")[-1].split(".")[0])
                })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {len(records)} label records → {output_path}")
```

### 9.2 Bounding Box Annotations (for Object Detection)

```python
# generate_bboxes.py
from PIL import Image
import json
import os
import numpy as np

def compute_tight_bbox(image_path: str):
    """
    Compute the tight bounding box around the non-transparent pixels.
    Returns (x_min, y_min, x_max, y_max) or None if fully transparent.
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max), int(y_max)

def generate_bbox_annotations(frames_dir: str, output_path: str):
    annotations = []
    for filename in os.listdir(frames_dir):
        if not filename.endswith(".png"):
            continue
        path = os.path.join(frames_dir, filename)
        bbox = compute_tight_bbox(path)
        if bbox:
            annotations.append({
                "file": filename,
                "bbox": {"x_min": bbox[0], "y_min": bbox[1],
                         "x_max": bbox[2], "y_max": bbox[3]}
            })
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)
```

### 9.3 COCO Format Export

```python
# export_coco.py
import json
import os
from PIL import Image

def export_to_coco(records: list, dataset_root: str, output_path: str):
    """
    Export dataset annotations to COCO JSON format.
    records: list of dicts with keys: file, character, animation, bbox
    """
    categories = {}
    images     = []
    annotations = []
    ann_id = 1

    for idx, rec in enumerate(records):
        img_path = os.path.join(dataset_root, rec["file"])
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id":        idx,
            "file_name": rec["file"],
            "width":     w,
            "height":    h,
        })

        char = rec["character"]
        if char not in categories:
            categories[char] = len(categories) + 1

        if "bbox" in rec:
            b = rec["bbox"]
            annotations.append({
                "id":          ann_id,
                "image_id":    idx,
                "category_id": categories[char],
                "bbox": [b["x_min"], b["y_min"],
                         b["x_max"] - b["x_min"],
                         b["y_max"] - b["y_min"]],
                "area": (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]),
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "images":      images,
        "annotations": annotations,
        "categories":  [{"id": v, "name": k} for k, v in categories.items()]
    }
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO dataset exported → {output_path}")
```

---

## Step 10 — Structuring and Exporting the Final Dataset

### 10.1 Train / Validation / Test Split

```python
# split_dataset.py
import os
import shutil
import random
import json

def split_dataset(
    labels_path: str,
    source_root: str,
    output_root: str,
    train_ratio: float = 0.75,
    val_ratio:   float = 0.15,
    # test gets the remainder (0.10)
    seed: int = 42
):
    with open(labels_path) as f:
        records = json.load(f)

    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": records[:n_train],
        "val":   records[n_train:n_train + n_val],
        "test":  records[n_train + n_val:]
    }

    for split_name, split_records in splits.items():
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_labels = []

        for rec in split_records:
            src = os.path.join(source_root, rec["file"])
            char_dir = os.path.join(split_dir, rec["character"])
            os.makedirs(char_dir, exist_ok=True)
            dst = os.path.join(char_dir, os.path.basename(rec["file"]))
            shutil.copy(src, dst)

            new_rec = dict(rec)
            new_rec["file"] = os.path.relpath(dst, output_root)
            split_labels.append(new_rec)

        label_out = os.path.join(split_dir, "labels.json")
        with open(label_out, "w") as f:
            json.dump(split_labels, f, indent=2)
        print(f"{split_name}: {len(split_records)} records → {split_dir}")
```

### 10.2 Dataset Statistics Report

```python
# dataset_stats.py
import json
import os
from collections import Counter
from PIL import Image

def print_dataset_stats(labels_path: str, source_root: str):
    with open(labels_path) as f:
        records = json.load(f)

    chars   = Counter(r["character"]  for r in records)
    anims   = Counter(r["animation"]  for r in records)
    systems = Counter(r.get("system", "unknown") for r in records)

    sizes = []
    for r in records:
        try:
            img = Image.open(os.path.join(source_root, r["file"]))
            sizes.append(img.size)
        except Exception:
            pass

    print(f"\n{'='*50}")
    print(f" Dataset Statistics")
    print(f"{'='*50}")
    print(f" Total frames      : {len(records)}")
    print(f" Characters        : {len(chars)}")
    print(f" Animation states  : {len(anims)}")
    print(f" Systems           : {dict(systems)}")
    if sizes:
        ws, hs = zip(*sizes)
        print(f" Frame size range  : {min(ws)}x{min(hs)} → {max(ws)}x{max(hs)}")
    print(f"\n Top characters:")
    for char, count in chars.most_common(10):
        print(f"   {char:20s} : {count:>5} frames")
    print(f"\n Animation distribution:")
    for anim, count in anims.most_common():
        print(f"   {anim:20s} : {count:>5} frames")
    print(f"{'='*50}\n")
```

### 10.3 Packaging as a HuggingFace Dataset

```python
# export_hf_dataset.py
from datasets import Dataset, DatasetDict, Image as HFImage
import json, os
from PIL import Image

def export_to_huggingface(splits_root: str, output_path: str):
    """
    Package the final dataset as a HuggingFace DatasetDict for easy sharing.
    """
    split_datasets = {}

    for split_name in ["train", "val", "test"]:
        labels_path = os.path.join(splits_root, split_name, "labels.json")
        if not os.path.exists(labels_path):
            continue
        with open(labels_path) as f:
            records = json.load(f)

        image_paths = [os.path.join(splits_root, r["file"]) for r in records]
        characters  = [r["character"]  for r in records]
        animations  = [r["animation"]  for r in records]

        split_datasets[split_name] = Dataset.from_dict({
            "image":     image_paths,
            "character": characters,
            "animation": animations,
        }).cast_column("image", HFImage())

    dataset_dict = DatasetDict(split_datasets)
    dataset_dict.save_to_disk(output_path)
    print(f"HuggingFace dataset saved → {output_path}")
```

---

## Appendix: Recommended Tools Reference

| Tool | URL | Notes |
|------|-----|-------|
| YY-CHR | [romhacking.net](https://www.romhacking.net/utilities/119/) | Tile pattern editor |
| Tile Molester | [romhacking.net](https://www.romhacking.net/utilities/108/) | Java-based, more flexible |
| Aseprite | [aseprite.org](https://www.aseprite.org) | Best sprite sheet slicer/editor |
| RetroArch | [retroarch.com](https://www.retroarch.com) | Emulator for frame capture |
| Real-ESRGAN | [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | AI upscaler, anime model |
| rembg | [github.com/danielgatis/rembg](https://github.com/danielgatis/rembg) | AI background removal |
| OpenGameArt | [opengameart.org](https://opengameart.org) | Open-licensed sprite assets |
| RHDN | [romhacking.net](https://www.romhacking.net) | ROM offset databases |
| HuggingFace Datasets | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets) | Dataset packaging and sharing |

---

## Quick Reference: Full Pipeline Summary

```
ROM File
   │
   ├─[YY-CHR / Tile Molester]─► Raw Sprite Sheets (PNG)
   │
   ├─[slice_sheet.py]──────────► Raw Frames (PNG grid slices)
   │
   ├─[remove_bg_color.py]──────► Clean Frames (alpha-transparent)
   │
   ├─[remap_palette.py]────────► Normalized Frames (consistent colors)
   │
   ├─[generate_flips.py]───────► Mirror Variants
   │
   ├─[upscale_xbrz.py / ESRGAN]► Upscaled Frames (2x–8x)
   │
   ├─[augment_dataset.py]──────► Augmented Variants
   │
   ├─[generate_labels.py]──────► JSON Labels
   │
   ├─[split_dataset.py]────────► train / val / test splits
   │
   └─[export_hf_dataset.py]────► HuggingFace DatasetDict
```

---

*Tutorial version 1.0 — February 2026*
