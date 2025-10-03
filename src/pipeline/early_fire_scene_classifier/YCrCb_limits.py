#!/usr/bin/env python3
"""
Visualize YCbCr colors for fixed Y with Cb on x-axis and Cr on y-axis,
and overlay a rectangular boundary defined by (cb_min, cb_max, cr_min, cr_max).
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ====================== CONFIG ======================
Y_FIXED = 100          # Fixed Y value (0-255)
CB_MIN = 0         # Minimum Cb value for boundary
CB_MAX = 140        # Maximum Cb value for boundary
CR_MIN = 160         # Minimum Cr value for boundary
CR_MAX = 255         # Maximum Cr value for boundary
SAVE_PATH = "/lab/projects/fire_smoke_awr/src/pipeline/early_fire_scene_classifier/plane.png"     # e.g., "ycbcr_plane.png" or None to skip saving
# =====================================================

# Clamp values
Y_FIXED = np.uint8(np.clip(Y_FIXED, 0, 255))
cb_min, cb_max = [int(np.clip(v, 0, 255)) for v in (CB_MIN, CB_MAX)]
cr_min, cr_max = [int(np.clip(v, 0, 255)) for v in (CR_MIN, CR_MAX)]

# Create 256x256 Cb–Cr grid
cb_vals = np.arange(256, dtype=np.uint8)
cr_vals = np.arange(256, dtype=np.uint8)
CB, CR = np.meshgrid(cb_vals, cr_vals)  # shapes: (256, 256)

# Build YCrCb image (OpenCV uses channel order: Y, Cr, Cb)
Y = np.full_like(CB, Y_FIXED, dtype=np.uint8)
ycrcb = np.dstack([Y, CR, CB])  # (H, W, 3)

# Convert to RGB for display
rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(rgb, origin="lower", extent=[0, 255, 0, 255], interpolation="nearest")
ax.set_xlabel("Cb")
ax.set_ylabel("Cr")
ax.set_title(f"YCbCr color field at Y={int(Y_FIXED)}")

# Overlay rectangular boundary
width = max(0, cb_max - cb_min)
height = max(0, cr_max - cr_min)
rect = Rectangle((cb_min, cr_min), width, height, fill=False, linewidth=2)
ax.add_patch(rect)

# Ticks and limits
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_xticks(range(0, 256, 32))
ax.set_yticks(range(0, 256, 32))
ax.grid(False)

# Save or show
if SAVE_PATH:
    plt.savefig(SAVE_PATH, bbox_inches="tight", dpi=150)
    print(f"Saved to {SAVE_PATH}")

plt.show()
