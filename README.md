# Vesuvius Challenge: ink detection

## Basic Information
* **Author:** Romain Frossard
* **Supervisor:** Prof. Frédéric Kaplan (Digital Humanities Laboratory - DHLAB)
* **Academic Year:** 2025-2026 (Spring Semester)

## About
This project was developed at the École Polytechnique Fédérale de Lausanne (EPFL) as part of a semester project. It provides a highly robust computer vision pipeline and a resulting dataset of over 800 binary ink masks derived from historical papyrus fragments (sourced from the Duke University and Oslo Archives). Initially designed to overcome the 2D masking bottleneck for 3D synthetic data generation in the **Vesuvius Challenge**, this tool isolates true carbon ink from degraded plant fibers to provide pixel-perfect ground truth.

## Research Summary
The extraction of clean 2D inputs from heavily degraded historical manuscripts is a formidable computer vision challenge. Deterministic mathematical filters (like Sauvola thresholding) often fail due to severe illumination variances and structural noise. 

To solve this, our pipeline leverages a deep learning framework:
1. **Feature Extraction:** We utilize a frozen Vision Foundation Model (**NVlabs/RADIO v2.5-l**). By upscaling input patches, each spatial token maps to a highly localized 4x4 pixel area, preserving microscopic ink features.
2. **Segmentation:** A custom Multi-Layer Perceptron (MLP) head processes the dense embeddings, trained with a specialized hybrid Dice-BCE loss function to heavily penalize the erasure of faint strokes.
3. **Post-Processing:** A hybrid bounded hysteresis binarization algorithm safely translates continuous sub-pixel probabilities into crisp, structural masks, successfully rejecting physical edge artifacts and hallucinated background fibers.

Evaluated on a Leave-One-Out Cross-Validation (LOOCV) protocol, the model achieves a Mean Recall of 97.72% and a Mean IoU of 81.44%.

![Qualitative Results](images/figure_1.png)

## Installation and Usage

### Dependencies
This project requires Python 3.10+ and the packages listed in `requirements.txt`.
To install the dependencies, create a virtual environment and run:

```bash
pip install -r requirements.txt
```

### Usage
*(Adapt this section depending on how you structured your scripts in `/lib`)*

To run the inference pipeline on a new high-resolution papyrus scan:

```bash
python lib/inference.py --input data/raw_scan.tif --output results/mask.png
```

For exploratory data analysis and qualitative visualizations, refer to the Jupyter notebooks in the `notebooks/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
dhlab-vesuvius-ink-masking - Romain Frossard  
Copyright (c) 2026 EPFL  
This program is licensed under the terms of the MIT License.