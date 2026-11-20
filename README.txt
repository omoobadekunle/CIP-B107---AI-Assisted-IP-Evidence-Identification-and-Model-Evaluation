# CIP-B107: AI-Assisted IP Evidence Identification and Model Evaluation

> A digital forensics and machine learning investigation focused on identifying, validating, and classifying IP address evidence using Python, Pandas, and Scikit-Learn.

## Table of Contents

- [Project Overview](#project-overview)
- [Assignment Objectives](#assignment-objectives)
- [Case Information](#case-information)
- [Technology Stack](#technology-stack)
- [Methodology](#methodology)
- [Dataset Profile](#dataset-profile)
- [Evidence Preservation](#evidence-preservation)
- [IP Extraction and Validation](#ip-extraction-and-validation)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Evaluation](#model-evaluation)
- [Historical Fine-Tuning Analysis](#historical-fine-tuning-analysis)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Recommendations](#recommendations)
- [Learning Outcomes](#learning-outcomes)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Project Overview

This project was completed as part of the **ICDFA CIP-B107 – AI-Assisted IP Evidence Identification and Model Evaluation** module. The assignment involved conducting a forensic examination of an IP-related dataset, extracting potential IP evidence from raw text, validating findings, building machine learning models, evaluating classifier performance, and critically assessing historical AI fine-tuning results.

The investigation combined traditional digital forensic methodology with machine learning techniques to determine how effectively artificial intelligence can assist analysts in identifying potentially relevant IP address evidence.

---

## Assignment Objectives

- Preserve digital evidence using forensic principles.
- Verify evidence integrity using SHA-256 hashing.
- Extract IP address candidates from raw text.
- Distinguish valid IPv4 addresses from false positives.
- Develop a rule-based classification baseline.
- Build and evaluate a Logistic Regression classifier.
- Compare machine learning performance against simpler approaches.
- Investigate historical model fine-tuning logs.
- Identify model limitations and dataset leakage.
- Produce defensible forensic findings and recommendations.

---

## Case Information

| Item | Details |
|--------|---------|
| Module | ICDFA CIP-B107 |
| Case Reference | CIP-B107-LAB01 |
| Dataset Records | 229 |
| Source Classes | 5 |
| Environment | Kali Linux |
| Student | Adekunle Ogunyemi |
| Analysis Type | AI-Assisted Digital Forensics |
| Evidence Source | Supplied ICDFA Practical Package |

---

## Technology Stack

### Operating Environment
- Kali Linux
- Python 3

### Libraries & Frameworks
- Pandas
- Scikit-Learn
- ipaddress
- NumPy
- Matplotlib
- Seaborn

### Forensic Utilities
- sha256sum
- zip

---

## Methodology

The investigation followed a structured forensic workflow:

1. Evidence acquisition and preservation.
2. Integrity verification using SHA-256 hashing.
3. Dataset profiling and validation.
4. IP candidate extraction using regular expressions.
5. IPv4 validation using Python's `ipaddress` module.
6. Machine learning model development.
7. Historical fine-tuning log analysis.
8. Classification performance evaluation.
9. Analyst review of misclassified records.
10. Documentation and reporting.

---

## Dataset Profile

### Dataset Summary

| Metric | Value |
|----------|--------|
| Total Records | 229 |
| Training Records | 183 |
| Test Records | 46 |
| Columns | IP, Class, Text |
| Missing Values | 0 |

### Class Distribution

| Class | Count |
|---------|---------|
| A | 177 |
| N | 32 |
| B | 12 |
| C | 4 |
| E | 4 |

The dataset demonstrates significant class imbalance, making overall accuracy alone insufficient for evaluating model performance.

---

## Evidence Preservation

Forensic integrity procedures included:

- Creation of separate originals and working-copy directories.
- SHA-256 hashing of all supplied evidence files.
- Evidence manifest generation.
- Post-analysis integrity verification.
- Documentation of timestamps and handling procedures.

### Evidence Files

- dataset_small.csv
- dataset_train.jsonl
- dataset_test.jsonl
- result.csv
- fine_tuning_steps.txt

All evidence files successfully passed integrity verification checks.

---

## IP Extraction and Validation

### Extraction Process

A regular expression was used to locate strings matching IPv4 formatting patterns.

### Results

| Measure | Count |
|-----------|---------|
| Regex Matches | 229 |
| Valid IPv4 Addresses | 198 |
| Invalid Candidates | 31 |
| Source Positive Labels | 197 |
| Source Negative Labels | 32 |

### Common False Positives

- Version numbers
- Software build identifiers
- Invalid octets above 255
- Addresses containing leading zeros
- URL-embedded numeric patterns

### Validation Strategy

Each candidate was validated using Python's built-in `ipaddress.IPv4Address()` functionality to ensure technical accuracy.

---

## Machine Learning Pipeline

### Model

**Logistic Regression Classifier**

### Feature Engineering

- TF-IDF Vectorization
- Word n-grams (1–3)
- Balanced class weighting

### Workflow

Raw Prompt Data
→ TF-IDF Vectorization
→ Logistic Regression
→ Classification Output
→ Evaluation Metrics

### Leakage Assessment

The investigation discovered:

- 11 overlapping prompts between training and testing datasets.
- 12 candidate-level overlaps.
- Measurable inflation of reported accuracy due to data leakage.

---

## Model Evaluation

### Performance Comparison

| Model | Accuracy |
|---------|----------|
| Logistic Regression (Full Holdout) | 97.83% |
| Logistic Regression (Clean Subset) | 93.75% |
| Rule-Based Baseline | 95.65% |
| Majority Baseline | 73.91% |

### ML Model Metrics

| Metric | Value |
|----------|---------|
| Accuracy | 97.83% |
| Precision | 97.14% |
| Recall | 100% |
| F1 Score | 98.55% |

### Key Observation

Although the machine learning model achieved the highest accuracy, part of its reported performance was influenced by train/test overlap. When leakage was removed, performance reduced to 93.75%, highlighting the importance of rigorous evaluation methods.

---

## Historical Fine-Tuning Analysis

Historical logs were reviewed to understand prior model behavior.

### Findings

- Training accuracy reached 100%.
- Validation checks were performed intermittently.
- Historical classification accuracy reached 97.83%.
- Performance was evaluated using the same test dataset.
- Evidence of dataset leakage existed in both historical and current evaluations.

### Important Conclusion

The historical results demonstrate memorization of the supplied dataset rather than proving real-world generalization capability.

---

## Key Findings

- Five evidence files were successfully preserved and verified.
- 229 IP candidates were extracted from source data.
- 198 valid IPv4 addresses were confirmed.
- 31 candidates were rejected as invalid.
- Logistic Regression achieved the highest recorded accuracy.
- Rule-based classification remained highly competitive.
- Train/test leakage significantly affected reported performance.
- Historical fine-tuning results reflected the same leakage issue.
- ML and rule-based approaches exhibited different failure modes.

---

## Limitations

- Single train/test split.
- Small dataset size.
- Significant class imbalance.
- Presence of train/test overlap.
- Model lacked technical IP validation awareness.
- Historical training configuration could not be fully verified.

---

## Recommendations

### Technical Improvements

- Remove overlapping prompts from evaluation datasets.
- Expand negative-class examples.
- Introduce mandatory IPv4 validation before classification.
- Rebuild test datasets with stricter separation controls.

### Operational Improvements

- Deploy as analyst-assistance only.
- Escalate ML/rule disagreements for manual review.
- Avoid fully automated decision-making.
- Require corroborating evidence before drawing investigative conclusions.

---

## Learning Outcomes

This assignment demonstrated practical skills in:

- Digital evidence preservation.
- Chain-of-custody management.
- Dataset profiling.
- Python-based forensic analysis.
- Regular expression extraction.
- IPv4 validation.
- Machine learning classification.
- Confusion matrix analysis.
- Model evaluation and benchmarking.
- Data leakage identification.
- Forensic reporting and documentation.

---

## Project Structure

```text
project/
├── originals/
├── working/
├── outputs/
├── reports/
├── screenshots/
├── scripts/
├── dataset_small.csv
├── dataset_train.jsonl
├── dataset_test.jsonl
├── result.csv
└── fine_tuning_steps.txt
```

---

## Author

**Adekunle Ogunyemi**

Digital Forensics & Cybersecurity Student

Module: ICDFA CIP-B107

---

## License

This repository is provided for academic, educational, and portfolio demonstration purposes only.
