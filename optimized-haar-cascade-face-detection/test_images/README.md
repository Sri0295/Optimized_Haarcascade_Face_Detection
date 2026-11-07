# Test Images

This directory contains grayscale images used to evaluate the Haar Cascade face detection system.

The images are processed by the detector using both default and fine-tuned parameter configurations.

---

## Purpose

The test images are used to determine how effectively the Haar Cascade classifier detects faces under different conditions.

The research evaluates the system using grayscale images and examines its behavior across different:

- Lighting conditions
- Face sizes
- Image scales
- Backgrounds
- Viewing conditions
- Occlusion conditions

The research specifically investigates real-world grayscale images to evaluate the effect of parameter tuning. :contentReference[oaicite:14]{index=14}

---

## Image Preprocessing

Before detection, input images are converted from RGB to grayscale.

Example:

```python
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)