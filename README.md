# NETWATCH-IDS — ML-Based Network Intrusion Detection System

A desktop GUI application that trains and evaluates machine learning models to detect malicious network traffic, built on the UNSW-NB15 intrusion detection dataset.

## Overview

NETWATCH-IDS loads network traffic data, trains two classifiers (Random Forest and Decision Tree), and presents the results through an interactive dashboard — including live performance gauges, a model comparison table, confusion matrix, ROC curve, feature importance map, and a filterable, severity-ranked alert log.

The goal was to simulate how a lightweight SOC (Security Operations Centre) tool might flag and triage suspicious traffic using ML-driven classification rather than static rule sets.

## Features

- **Dual-model training** — Random Forest (primary) and Decision Tree (baseline) for comparison
- **Live dashboard** — animated gauges for accuracy, precision, recall, F1 score, ROC AUC, and false positive rate
- **Confusion matrix & ROC curve visualisation**
- **Feature importance map** — shows which network traffic features most influenced predictions
- **Severity-ranked alert log** — predicted attacks are classified as Critical / High / Medium / Low based on model confidence, with filterable views
- **Background threading** — model training runs off the main thread so the UI stays responsive

## Dataset

Trained and evaluated on **[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)**, a widely used benchmark dataset for network intrusion detection research, containing both normal traffic and a range of modern attack types.

## Tech Stack

- **Python**
- **scikit-learn** — Random Forest & Decision Tree classifiers, preprocessing, evaluation metrics
- **pandas / NumPy** — data loading and preprocessing
- **Tkinter** — desktop GUI
- **Matplotlib** — charts (confusion matrix, ROC curve, feature importance)

## How It Works

1. User selects training and testing CSV files via the sidebar
2. Clicking **"Initiate Scan"** runs the pipeline in a background thread:
   - Load → Preprocess (cleaning, encoding, scaling) → Train → Evaluate → Display
3. Results populate the dashboard in real time: performance metrics, model comparison, and a ranked list of detected threats

## Getting Started

```bash
git clone https://github.com/Bishurestz/GUI_ML_IDS.git
cd GUI_ML_IDS
python install_dependencies.py
```

Then select a UNSW-NB15 training CSV and testing CSV from the sidebar and click **Initiate Scan**.

## Project Background

This project was built to apply machine learning techniques to a practical cyber security problem: distinguishing malicious network traffic from normal activity. It combines data preprocessing, model training/evaluation, and a custom-built GUI to make the results interpretable at a glance — similar to a simplified SIEM-style interface.

## Notes

- See `NETWATCH_capstone_project.docx` for extended write-up and technical documentation.
