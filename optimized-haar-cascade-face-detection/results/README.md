# Detection Results

This directory contains the experimental results produced by the Haar Cascade face detection system.

The results are used to compare the behavior of the default Haar Cascade configuration with the fine-tuned configuration.

---

## Purpose

The primary objective of the project is to investigate whether tuning the Haar Cascade detection parameters can reduce false positives and improve face detection accuracy.

The research evaluates the detector using:

- Accuracy
- Precision
- Recall
- F1-Score
- False-positive behavior
- Processing time

The paper states that the final optimized model was also tested under real-world conditions to evaluate robustness. :contentReference[oaicite:7]{index=7}

---

## Default Haar Cascade Results

The default Haar Cascade configuration produced:

| Metric | Result |
|---|---:|
| Accuracy | 22% |
| Precision | 0.21 |
| Recall | 0.20 |
| F1-Score | 0.21 |

The default configuration produced a higher number of false-positive detections and missed some genuine objects because the detection parameters were not optimized for the evaluated images. :contentReference[oaicite:8]{index=8}

---

## Fine-Tuned Haar Cascade Results

After parameter optimization, the results were:

| Metric | Result |
|---|---:|
| Accuracy | 90% |
| Precision | 0.89 |
| Recall | 0.92 |
| F1-Score | 0.90 |

The fine-tuned detector produced fewer false positives and more consistent detections than the default configuration. :contentReference[oaicite:9]{index=9}

---

## Performance Comparison

```text
                    DEFAULT       FINE-TUNED

Accuracy              22%             90%
Precision             0.21            0.89
Recall                0.20            0.92
F1-Score              0.21            0.90