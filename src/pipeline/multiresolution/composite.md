# Multi-resolution Composite Image Strategy

## Overview

This document outlines the multi-resolution strategy used to prepare images for object detection. The primary goal is to create a standardized input size (e.g., 640x640) for the detector while preserving both high-resolution details and global context, especially for wide-aspect-ratio images.

The pipeline intelligently handles images based on their aspect ratio, applying one of two distinct processing methods.

---

## General Pipeline Steps

### 1. Image Aspect Ratio Classification

The first step is to classify the input image:
- **Wide Image:** An image where the width is greater than the height (and both dimensions are larger than the target size).
- **Tall or Small Image:** An image where the height is greater than or equal to the width, or where either dimension is smaller than the target size.

The subsequent processing depends on this classification.

### 2. Processing for Wide-Aspect-Ratio Images

For wide images, a two-band **composite image** is generated. This combines a high-resolution cropped region with a resized view of the full image.

-   **Top Band (High-Resolution Crop):**
    1.  A region of interest is identified in the original image. This is often guided by a **skyline detection** algorithm (`detect_skyline_y`) to intelligently center the crop.
    2.  A high-resolution version of this region is cropped from an upscaled "intermediate" version of the original image.
    3.  This crop is resized and padded to form the top band of the composite.

-   **Bottom Band (Global Context):**
    1.  The entire original image is resized to fit the target width (e.g., 640px), preserving its aspect ratio.
    2.  This resized image is padded to form the bottom band.

-   **Final Composite:**
    The top and bottom bands are stacked vertically to create the final square composite image. The height of the top band is precisely calculated to ensure the final stacked image has the correct dimensions (e.g., 640x640).

### 3. Processing for Tall or Small Images

For images that are not wide, a simpler approach is used:

1.  **Resize and Pad:** The image is resized to fit within the target dimensions (e.g., 640x640) while maintaining its aspect ratio.
2.  **Canvas Placement:** The resized image is placed onto a square canvas of the target size, with padding added to the empty areas. This is a standard letterboxing or pillarboxing process.

---

## Metadata Generation

A critical output of this pipeline is a `meta` dictionary. This dictionary stores all the scaling, cropping, and padding parameters used during the transformation.

This metadata is essential for the post-detection step of **mapping bounding box coordinates** from the processed image back to the original image's coordinate system. Because the pipeline knows whether a detection occurred in the top or bottom band (or in a padded image), it can apply the correct inverse transformation.

---

## Intended Use

-   **Improved Detection Accuracy:** By providing a high-resolution view of key regions, the model can better detect small or distant objects that might be lost in a simple downscaling.
-   **Consistent Input:** Ensures the object detector receives a consistently sized input, regardless of the original image's dimensions.
-   **Data Augmentation:** Acts as a sophisticated form of data augmentation that provides both focused and contextual views simultaneously.

---

## Dependencies

-   `numpy`
-   `opencv-python`
