# Alzheimer's Research: Multimodal Classification

This repository contains the research codebase for a high-performance system designed for Alzheimer's disease classification. The project achieves state-of-the-art results by leveraging both clinical data and imaging features through late-fusion architectures.

## Project Overview

- **Dataset**: Multimodal data including fMRI imaging and ADNI clinical variables.
- **Performance**: Achieves ~97% accuracy using a late-fusion ensemble approach.
- **Key Features**:
  - Image-only CNN ensembles.
  - Clinical data expert models.
  - Unified fusion logic for final prediction.

## Repository Structure

- `ensemble.py`, `ensemble1.py`, `ensemble2.py`: Imaging ensemble models.
- `train_clinical_only.py`: Training clinical data experts.
- `finalize_unified_97_fusion.py`: Final fusion logic for multimodal predictions.
- `generate_publication_plots.py`: Visualization tools for research findings.
- `cv_project_report.pdf`: Detailed research report.

---
*Note: Large imaging datasets (`.pkl`) and raw clinical data (`.csv`) are excluded from this repository due to size constraints.*
