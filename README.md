# Soybean Pest Detection Benchmark

This repository contains the code, experiment configurations, statistical analysis scripts, Weighted Boxes Fusion (WBF) implementation, and exact dataset partition assignments used in the paper:

**Real-Time Soybean Pest Monitoring: Benchmarking Object Detection Models and Evaluation Protocols**

## Study Overview

This study benchmarks state-of-the-art object detection models for soybean insect pest detection using the InsectBase dataset.

The evaluated models are:

- YOLOv8s
- YOLO26s
- RT-DETR-L

Three evaluation protocols are included:

- E1: Image-level random splitting
- E2: Specimen-level random splitting
- E3: Specimen-level 10-fold cross-validation

Weighted Boxes Fusion (WBF) was evaluated by optimizing fusion weights on the validation set using Optuna. The selected weights were then fixed and applied once to the independent test set to obtain the final reported performance.

---

## Repository Structure

```
configs/
```

Experiment configuration files.

```
scripts/
```

Training, evaluation, statistical analysis, partition generation, WBF, and efficiency benchmarking scripts.

```
partitions/
```

Exact image-level, specimen-level, and cross-validation partition assignments used in the paper.

```
requirements.txt
```

Python package requirements.

---

## Partition Files

### Image-level random splits

- seed0.csv
- seed42.csv
- seed123.csv

### Specimen-level random splits

- seed0_specimen.csv
- seed42_specimen.csv
- seed123_specimen.csv

### Specimen-level 10-fold cross-validation

- kfold_ms42.csv

The random-split CSV files contain:

```
filename,specimen_id,split
```

The cross-validation CSV contains:

```
filename,specimen_id,fold,split
```

These files provide the exact partition assignments used in the experiments and allow reproduction of the evaluation protocols reported in the paper.

---

## Environment Setup

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset

The InsectBase soybean pest dataset is **not redistributed** in this repository.

Please obtain the dataset from the original source referenced in the paper and organize it according to the expected directory structure before running the training scripts.

---

## Reproducibility

This repository includes:

- Training scripts
- Evaluation scripts
- Model configuration files
- Weighted Boxes Fusion implementation
- Statistical analysis scripts
- Efficiency benchmarking scripts
- Exact dataset partition assignments

Together, these files enable reproduction of the experiments and evaluation protocols presented in the paper.

---

## Citation

If you use this repository in your research, please cite the associated paper:

**Real-Time Soybean Pest Monitoring: Benchmarking Object Detection Models and Evaluation Protocols**

Citation details will be updated after publication.

---

## License

This repository is released under the MIT License.
