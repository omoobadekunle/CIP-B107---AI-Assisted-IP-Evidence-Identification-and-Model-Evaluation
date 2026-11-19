# CIP-B107: AI-Assisted IP Evidence Identification and Model Evaluation

**A forensic examination of a supplied IP dataset using Python, Pandas and Scikit-Learn, carried out as part of the ICDFA CIP-B107 module.**

| Field | Detail |
|---|---|
| **Module** | ICDFA CIP-B107 AI-Assisted IP Evidence Identification and Model Evaluation |
| **Case reference** | CIP-B107-LAB01 |
| **Case** | IP address dataset, 229 records, 5 source classes |
| **Examiner** | Adekunle Ogunyemi |
| **Environment** | Kali Linux |
| **Evidence source** | Supplied ICDFA practical package |
| **Examination date** | 18 November 2026, 13:50:38 WAT |
| **Time standard** | WAT for examination times, UTC for model timestamps where stated |
| **Status** | Completed |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Executive Summary](#2-executive-summary)
3. [Objectives](#3-objectives)
4. [Tools and Environment](#4-tools-and-environment)
5. [Evidence Handling and Methodology](#5-evidence-handling-and-methodology)
6. [Dataset Profiling](#6-dataset-profiling)
7. [IP Candidate Extraction and Validation](#7-ip-candidate-extraction-and-validation)
8. [Offline Machine Learning Classifier](#8-offline-machine-learning-classifier)
9. [Historical Fine-Tuning Investigation](#9-historical-fine-tuning-investigation)
10. [Model Evaluation and Analyst Review](#10-model-evaluation-and-analyst-review)
11. [Key Findings and Correlations](#11-key-findings-and-correlations)
12. [Scope and Limitations](#12-scope-and-limitations)
13. [Intelligence Memorandum](#13-intelligence-memorandum)
14. [Executive Briefing](#14-executive-briefing)
15. [Key Learning Outcomes](#15-key-learning-outcomes)
16. [Repository Contents](#16-repository-contents)
17. [Author](#17-author)

---

## 1. Overview

This repository documents my AI-assisted IP evidence identification practical for the ICDFA CIP-B107 module. The exercise required me to preserve a supplied dataset, extract IP addresses from raw text, train an offline machine learning classifier, investigate historical fine-tuning logs, evaluate the model, and produce a forensic report.

I preserved the source evidence, created separate working copies, calculated SHA-256 hashes, extracted IP candidates with a regular expression, validated them with Python's `ipaddress` module, and trained a Logistic Regression classifier with Scikit-Learn on the supplied JSONL files. I paid particular attention to what the historical training logs actually prove, and to the difference between training accuracy and real-world model reliability.

## 2. Executive Summary

I conducted a forensic examination of a supplied dataset of 229 IP-related records using Kali Linux and Python. My main objective was to understand how well an AI model can identify and classify IP addresses from raw text, and where the limits of that model lie.

I verified the integrity of the five supplied evidence files with SHA-256 hashing, recorded file sizes and the examination time in WAT, and built an evidence manifest. All analysis was performed on working copies so that the originals were never altered.

### Key Findings

- Five evidence files were preserved and verified with SHA-256. All five passed a post-examination integrity recheck.
- The JSONL files match the expected split: 183 training records and 46 testing records.
- The training set is 163 positive / 20 negative; the test set is 34 positive / 12 negative.
- 229 IP-like strings were extracted. Python's `ipaddress` module rejected 31 as invalid, leaving 198 valid IPv4 addresses.
- The offline Logistic Regression classifier achieved **97.83%** accuracy on the full holdout.
- The rule-based baseline achieved **95.65%** on the same holdout.
- The majority-class baseline (always predicting "positive") achieved **73.91%**.
- 11 exact prompt overlaps exist between the training and test sets. With these removed, ML accuracy falls to **93.75%** on the clean subset (n=16).
- The historical `result.csv` reports a final accuracy of 97.83%, precision of 97.14%, recall of 1.0, F1 of 0.9855, AUROC of 0.9975 and AUPRC of 0.9991.
- The historical result and my fresh model report the same accuracy because both are evaluated on the same test set, which contains 11 leaked training examples.

My conclusions are limited to the records examined and the test split used. Further work with more balanced data would be needed before making stronger claims about model reliability.

## 3. Objectives

- Preserve the supplied evidence and record its integrity.
- Profile the dataset and the JSONL files.
- Extract IP candidates from raw text and distinguish valid IPv4 addresses from misleading strings.
- Build a transparent rule-based baseline and an offline machine learning classifier.
- Investigate the historical fine-tuning logs and interpret what they actually prove.
- Evaluate the model using a confusion matrix, per-class metrics and an 8-record analyst review.
- Produce an executive briefing and final submission package.
- Identify limitations and recommend improvements for future work.

## 4. Tools and Environment

| Tool | Purpose |
|---|---|
| Kali Linux | Examination environment |
| `sha256sum` | SHA-256 hashing of evidence files |
| Python 3 | Scripting and analysis |
| Pandas | Loading and profiling the dataset and JSONL files |
| `ipaddress` | Validating IPv4 candidates |
| Scikit-Learn | TF-IDF vectorisation and Logistic Regression classifier |
| Matplotlib and Seaborn | Confusion matrix visualisation |
| `zip` | Packaging the final submission |

All work was done offline. No external API or paid service was used.

## 5. Evidence Handling and Methodology

I created a folder structure with separate areas for originals, working copies, scripts, reports, screenshots and outputs. The five core evidence files were copied into originals and left untouched.

| Property | Result |
|---|---|
| Filenames | `dataset_small.csv`, `dataset_train.jsonl`, `dataset_test.jsonl`, `result.csv`, `fine_tuning_steps.txt` |
| Hash algorithm | SHA-256 |
| Examination time | 18 November 2026, 13:50:38 WAT |
| Integrity recheck | All five files returned OK |
| Additional reference files | `compute_classification_metrics.ipynb`, `preprocess_dataset.py` (also hashed and verified) |

The evidence manifest (`evidence_manifest.csv`) records the filename, SHA-256 hash, file size in bytes, examination time in WAT, examiner name and transfer source for each evidence file.

## 6. Dataset Profiling

`dataset_small.csv` has 229 rows and three columns (`IP`, `class`, `text`), with no missing values.

### 6.1 Class Distribution

| Class | Count | Approximate share |
|---|---|---|
| A | 177 | 77.3% |
| N | 32 | 14.0% |
| B | 12 | 5.2% |
| C | 4 | 1.7% |
| E | 4 | 1.7% |
| **Total** | **229** | **100%** |

The distribution is heavily uneven. A model could reach about 77% accuracy by always predicting Class A, so I relied on per-class metrics and a confusion matrix rather than overall accuracy alone.

### 6.2 JSONL Profile

| File | Records | Positive | Negative |
|---|---|---|---|
| `dataset_train.jsonl` | 183 | 163 | 20 |
| `dataset_test.jsonl` | 46 | 34 | 12 |

The prompt format is `context;candidate ->`, and the completion is `positive` or `negative`. The semicolon separates the raw text context from the candidate IP string. The record counts match the expected 183/46 split exactly.

### 6.3 CSV vs JSONL Comparison

The `ip` column in the CSV matches the candidate in the JSONL prompt, the `class` column maps to the source annotation, and the `text` column contains the XML context. The JSONL bundles the same content into a single prompt string and labels it binary (positive or negative) instead of multi-class (A, B, C, E, N).

## 7. IP Candidate Extraction and Validation

### 7.1 First Extraction Attempt

A regular expression matching four groups of one to three digits separated by dots produced 229 candidate strings. It also caught strings that are not valid IPv4 addresses, for example:

- `1.000.000.000`: leading zeros and large octets.
- `608.80.24.1`: first octet over 255.
- `0.0.0.392`: last octet over 255.
- `0.9.9.3`: a version number extracted from a SourceForge URL.
- `257.0.1.38`: first octet over 255, part of a five-part version string.
- `6.8.4.2`: an SDK version string.

A regular expression alone cannot determine whether a string is a valid IP address.

### 7.2 Adding Validation

I passed every candidate through `ipaddress.IPv4Address`, which rejects octets over 255 and leading zeros. A `ValueError` marks the candidate as invalid.

### 7.3 Three-Way Comparison

| Concept | Count |
|---|---|
| Regex matched a candidate | 229 |
| Syntactically valid IPv4 | 198 |
| Source-labelled positive | 197 |
| Source-labelled negative | 32 |

### 7.4 Disagreements

| Type | Records | Handling |
|---|---|---|
| Invalid IPv4 but source-labelled positive | 7 | Retained and explained in analyst notes, as positive labels are not proof of technical validity (e.g. `608.80.24.1`, `1.000.000.000`) |
| Invalid IPv4 and source-labelled negative | 24 | Consistent: validator and source agree the string is not an IP |
| Valid IPv4 but source-labelled negative | 8 | `0.0.0.0` (x4) unspecified; `0.5.1.3` and `0.6.7.8` in the reserved 0.0.0.0/8 block; `0.9.9.3` (x2) a software version number in a URL |

### 7.5 Candidate Review Export

I exported `outputs/candidate_review.csv` with the columns `source_row`, `context`, `candidate`, `regex_match`, `ipv4_valid`, `source_label` and `analyst_note`. All 229 rows carry analyst notes.

## 8. Offline Machine Learning Classifier

### 8.1 Approach

I used Scikit-Learn, running entirely on my machine. The input feature was the full prompt string, converted with TF-IDF using word n-grams from 1 to 3 and a token pattern that keeps alphanumeric groups and dots together. I trained a Logistic Regression classifier with `class_weight='balanced'`. The vectoriser was fitted only on the training records, so no test vocabulary leaked into the training features.

### 8.2 Leakage Check

| Check | Result |
|---|---|
| Unique training prompts | 138 |
| Unique test prompts | 25 |
| Exact prompt overlaps | 11 |
| Unique training candidates | 96 |
| Unique test candidates | 23 |
| Candidate-level overlaps | 12 |

About 24% of test prompts (11 of 46) are identical to a training prompt, and about 26% of test candidates share an IP string with a training record. Any model will trivially classify these correctly, which inflates the measured accuracy.

### 8.3 ML Classifier Results

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| negative | 1.00 | 0.92 | 0.96 | 12 |
| positive | 0.97 | 1.00 | 0.99 | 34 |
| **Overall accuracy** | | | **0.9783** | 46 |

The model made one error: it predicted "positive" for one negative record (see Section 10).

### 8.4 Rule-Based Baseline

The rule: extract the candidate; if it does not match the IPv4 regex, predict negative; if it matches and passes `ipaddress.IPv4Address`, predict positive; otherwise predict negative.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| negative | 1.00 | 0.83 | 0.91 | 12 |
| positive | 0.94 | 1.00 | 0.97 | 34 |
| **Overall accuracy** | | | **0.9565** | 46 |

The rule baseline is 2.17 percentage points behind the ML model. It misses 2 negatives because `0.0.0.0` passes the `ipaddress` check.

### 8.5 Majority Baseline

The majority class in the training set is "positive" (163 of 183). A model that always predicts positive achieves 73.91% on the test set.

### 8.6 Leakage-Aware Supplementary Evaluation

| Evaluation | Accuracy | n |
|---|---|---|
| Full test set | 0.9783 | 46 |
| Clean subset (no overlapping prompts) | 0.9375 | 16 |
| Rule baseline (full test set) | 0.9565 | 46 |
| Majority baseline | 0.7391 | 46 |

Removing the leaked prompts reduces accuracy by about 4 percentage points, showing that the headline figure is optimistic. Outputs were saved to `outputs/predictions.csv`, `outputs/phase_c_classification_report.txt` and `outputs/phase_c_confusion_matrix_v2.png`.

## 9. Historical Fine-Tuning Investigation

### 9.1 Source Files

I examined `fine_tuning_steps.txt` (49 rows, a truncated view of the start of the run) and `result.csv` (733 rows).

### 9.2 Full Run Properties

| Property | Value |
|---|---|
| Total steps | 733 |
| Total elapsed tokens | 34,661 |
| Total elapsed examples | 733 |

### 9.3 Training vs Validation

Training accuracy reaches 1.0 by step 5 and remains at 1.0 for the remaining 728 steps, which is consistent with overfitting: the model memorised the training set. Validation loss was recorded only 92 times in 733 steps. Of those checks, 82 recorded validation sequence accuracy of 1.0 and 10 recorded below 1.0. The first check recorded 0.0 (untrained model) and the last recorded 1.0.

### 9.4 Final Classification Metrics (Step 733)

| Metric | Value |
|---|---|
| classification/accuracy | 0.978261 |
| classification/precision | 0.971429 |
| classification/recall | 1.000000 |
| classification/auroc | 0.997549 |
| classification/auprc | 0.999147 |
| classification/f1.0 | 0.985507 |

### 9.5 What These Numbers Prove and Do Not Prove

The historical model's final accuracy of 97.83% equals my fresh Logistic Regression accuracy because both use the same `dataset_test.jsonl` holdout, which contains 11 prompts also present in the training set. The historical accuracy is inflated by the same leakage. This is a historical result supplied by the module, not a model I trained, and it is reported as such.

### 9.6 Legacy Workflow Note

The original workflow (upload, fine-tuning job, status poll, prediction) used OpenAI's legacy "ada" model and obsolete fine-tuning endpoints. The supplied scripts also contain embedded credentials. I did not run these scripts and did not copy any embedded token into this report.

### 9.7 Limitations of the Historical Logs

| I can conclude | I cannot conclude |
|---|---|
| The model memorised the training set | The historical model's real-world reliability |
| Final classification metrics exist in `result.csv` | The training/validation split composition |
| Those metrics were measured on a test set with training overlaps | Whether the model would reproduce at a different step count |

The analysis was saved to `reports/phase_d_historical_analysis.txt` and the loss plot to `outputs/phase_d_loss_plot.png`.

## 10. Model Evaluation and Analyst Review

### 10.1 Confusion Matrix (ML Model)

| | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | 11 (TN) | 1 (FP) |
| **Actual positive** | 0 (FN) | 34 (TP) |

| Metric | Value |
|---|---|
| Accuracy | 0.9783 |
| Precision | 0.9714 |
| Recall | 1.0000 |
| F1 | 0.9855 |
| Majority baseline | 0.7391 |

### 10.2 Eight-Record Analyst Review

| # | Candidate | Actual | ML | Rule | Type | Analyst note |
|---|---|---|---|---|---|---|
| 1 | `2.20.31.4` | positive | positive | positive | TP | Valid IPv4 in positive context; both agree |
| 2 | `13.3.1.17` | positive | positive | positive | TP | Valid IPv4 in positive context; both agree |
| 3 | `608.80.24.1` | negative | negative | negative | TN | Invalid IPv4 (octet over 255); both correctly rejected |
| 4 | `257.0.1.38` | negative | negative | negative | TN | Invalid IPv4 (octet over 255); both correctly rejected |
| 5 | `0.0.0.0` | negative | negative | positive | TN (ML beat rule) | Passes the IPv4 check but is the unspecified address; ML used context, rule alone was wrong |
| 6 | `0.0.0.0` | negative | negative | positive | TN (ML beat rule) | As record 5 |
| 7 | `256.0.1.26` | negative | positive | negative | FP | Invalid IPv4 (octet 256); rule correct, ML matched a four-group digit pattern inside a `<string>` tag |
| 8 | (additional negative) | negative | negative | negative | TN | Added for balance; both correct |

**Review conclusions**

- The ML model beat the rule on the two `0.0.0.0` cases, where the rule sees a valid IPv4 and the ML learns from context that the source labels it negative.
- The ML model failed on `256.0.1.26`, where the rule was correct. The ML model has no notion of octet range.
- The two approaches have complementary failure modes; neither is universally better.

The review was saved to `outputs/analyst_review.csv`.

### 10.3 Counterexample: High Accuracy on Imbalanced Data

The majority-class baseline scores 73.91% on this test set. The ML model's 97.83% is well above that, so it is genuinely useful on this split, but the headline figure needs that context: roughly 74 points come from predicting the common class and about 24 from actual classification skill.

### 10.4 False-Negative Consequences for Forensic Triage

A false negative is a positive candidate labelled negative, meaning a real IP address is dropped before analyst review. This is more serious than a false positive, which only costs analyst time. The ML model produced 0 false negatives on this test set; the rule baseline produced 2 (the two `0.0.0.0` records).

### 10.5 Analyst Escalation

Where the ML model and the rule baseline disagree (records 5, 6 and 7), the record should be escalated to an analyst rather than resolved automatically. Automated resolution would have produced the wrong answer in all three cases.

## 11. Key Findings and Correlations

### Findings

- Five evidence files were preserved and verified with SHA-256 hashing.
- The JSONL split matches the brief: 183 training and 46 testing records.
- 229 IP-like strings were extracted; 31 were rejected as invalid, leaving 198 valid IPv4 addresses.
- The classifier achieved 97.83% on the full holdout and 93.75% on the clean subset.
- The rule baseline achieved 95.65%; the majority baseline achieved 73.91%.
- 11 exact prompt overlaps exist between train and test.
- The historical `result.csv` reports the same accuracy as my fresh model because both use the same leaky holdout.

### Correlations

- The only ML false positive in the test set (`256.0.1.26`) is an invalid IPv4 address that the rule caught and the ML missed, linking the validation output to the classifier's failure mode.
- The class imbalance observed in profiling (77% Class A) corresponds to the leakage-inflated accuracy in the JSONL-based binary run.
- The historical accuracy (0.9783) and my fresh ML accuracy (0.9783) are identical because both are evaluated on the same split with the same overlaps.

### Alternative Explanations Considered

- The 97.83% accuracy may be partly due to the 11 leaked examples rather than a genuinely strong classifier. The clean-subset accuracy of 93.75% supports this.
- The `0.0.0.0` cases where ML beat the rule could reflect a lucky pattern match rather than a general learned context rule. With only two test examples, the two cannot be distinguished.
- The `256.0.1.26` failure could reflect a fundamental limitation of the ML approach (no notion of IP validity) rather than a fixable training gap.

## 12. Scope and Limitations

This work is complete for the evidence sets it documents: evidence integrity check, dataset and JSONL profiling, IP extraction and validation, offline classifier, historical fine-tuning analysis and the 8-record analyst review. Conclusions are limited to what the records show:

- The findings rest on a single 183/46 split. A different split could produce different misclassifications and metrics.
- 11 of 46 test prompts are identical to training prompts, so 97.83% should be read alongside the clean-subset 93.75%.
- The classifier used only the prompt text as a feature; no external features were used.
- The historical logs do not identify which dataset was used for fine-tuning or how its splits were built.
- `256.0.1.26` appears in the test split because the JSONL split is fixed. In a production pipeline, strict validation would run first and flag this record before it reached the model.
- The model treats IP addresses as text patterns and does not validate them. Any claim that the model identifies valid network addresses must be checked against the validation output.

## 13. Intelligence Memorandum

| Field | Detail |
|---|---|
| **Memorandum** | ICDFA CIP-B107-LAB01 |
| **To** | Intelligence Review Panel |
| **From** | Adekunle Ogunyemi, Digital Forensics Examiner |
| **Date** | 18 November 2026 |

**Remit.** To preserve the evidence, extract IP candidates, train an offline classifier and produce defensible findings for an intelligence review.

**Evidence manifest.** All five core evidence files were hashed with SHA-256 and recorded in `evidence_manifest.csv`. The examination time was 2026-11-18 13:50:38 WAT, and post-examination integrity rechecks passed for all files.

**Methodology.** Regex extraction pulled IP-like strings from the JSONL prompts; strict IPv4 validation used Python's `ipaddress` module; a rule-based baseline and a Logistic Regression classifier were built with Scikit-Learn; the JSONL split was preserved as the holdout.

**Results.** 97.83% accuracy on the full holdout and 93.75% on the clean subset; 95.65% for the rule baseline; 73.91% for the majority baseline. Eleven exact prompt overlaps exist, and accuracy falls by about 4 percentage points when they are removed.

**Limitations.** The JSONL split contains leakage. The historical `result.csv` is evaluated on the same leaky split and must not be presented as a fresh student result. The historical logs show memorisation, not generalisation.

**Recommendation.** Deploy the model as analyst assistance only, not automated decision-making. Route disagreements between ML and rule to an analyst. Do not classify any recovered address as malicious without corroborating evidence.

**Next investigative actions**

- Build a clean holdout by removing the 11 overlapping prompts before training.
- Expand the negative class with more context-aware examples (URLs, version strings, wildcard addresses).
- Add a hard `ipaddress` validation step in front of the model so it never sees an invalid address.
- Review the historical fine-tuning scripts for credential removal and API version updates before any use.

## 14. Executive Briefing

**Purpose.** This briefing summarises the CIP-B107 findings on AI-assisted IP evidence identification.

**Methodology.** Evidence was preserved and hashed. IP candidates were extracted with regex and validated with `ipaddress`. A Logistic Regression classifier was trained on the supplied JSONL data and compared with a rule-based baseline and a majority-class baseline.

| Model | Accuracy |
|---|---|
| Logistic Regression (full holdout) | 97.83% |
| Logistic Regression (clean subset) | 93.75% |
| Rule-based baseline | 95.65% |
| Majority baseline | 73.91% |

**Key findings**

- The ML classifier is not dramatically better than the rule baseline; well-designed rules get most of the way there.
- The headline accuracy is inflated by train/test leakage (11 of 46 test prompts appear in training).
- The ML model cannot tell whether an IP is technically valid: it classified `256.0.1.26` as positive because it matched a digit pattern.
- The rule baseline cannot tell whether an IP is forensically meaningful: it classified `0.0.0.0` as positive because it passes `ipaddress.IPv4Address`.
- The historical fine-tuning run shows 100% training accuracy and perfect validation accuracy after a few steps, which indicates memorisation rather than real-world reliability.

**Recommendation.** Deploy as analyst assistance only. Escalate ML/rule disagreements to an analyst. Never classify an IP as malicious without corroborating evidence.

## 15. Key Learning Outcomes

- Preserving evidence and recording integrity with hashes and a chain-of-custody manifest.
- Profiling a dataset with Pandas and JSONL files with Python's `json` module.
- Using regular expressions to extract candidate IP addresses from raw text.
- Using Python's `ipaddress` module to validate IPv4 candidates and reject false positives.
- Training an offline machine learning classifier with Scikit-Learn and TF-IDF vectorisation.
- Evaluating a classifier with per-class metrics, a confusion matrix and a majority baseline.
- Distinguishing training accuracy from real-world reliability when reviewing historical logs.
- Detecting and disclosing train/test leakage and its effect on measured accuracy.
- Tracing misclassifications back to specific records and explaining them in forensic terms.

## 16. Repository Contents

```text
CIP-B107-LAB01/
├── README.md
├── CIP-B107_Report.docx
├── evidence_manifest.csv
├── outputs/
│   ├── candidate_review.csv
│   ├── predictions.csv
│   ├── analyst_review.csv
│   ├── phase_c_classification_report.txt
│   ├── phase_c_confusion_matrix_v2.png
│   └── phase_d_loss_plot.png
└── reports/
    └── phase_d_historical_analysis.txt
```

## 17. Author

**Adekunle Ogunyemi**
Digital Forensics Examiner, ICDFA CIP-B107

---

*Prepared for the ICDFA CIP-B107 module. Findings are limited to the supplied evidence and the test split described in this document.*
