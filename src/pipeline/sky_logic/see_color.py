import cv2
import numpy as np

# ======== CONFIG =========
Y_value  = 100   # Luminance
Cb_value = 157   # Blue-difference chroma
Cr_value = 111   # Red-difference chroma
output_path = "solid_ycrcb.png"
# =========================

# Create a 100x100 array filled with (Y, Cr, Cb)
# OpenCV uses YCrCb ordering
img_ycrcb = np.full((100, 100, 3), (Y_value, Cr_value, Cb_value), dtype=np.uint8)

# Save in YCrCb format directly won't look correct in most viewers,
# so convert to BGR for saving
img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(output_path, img_bgr)

print(f"✅ Saved solid YCrCb image to {output_path}")
