# 🔍 Telco Customer Churn — Multi-Model Classification Benchmark

**Author:** Your Name  
**Course:** Big Data & Machine Learning  
**Problem Domain:** Customer Churn Prediction  

---

## Executive Summary

This project benchmarks multiple classification algorithms to predict telecom customer churn. It includes:

- End-to-end preprocessing and feature engineering  
- Training and hyperparameter tuning for all major classifiers  
- Full evaluation using Accuracy, F1, ROC-AUC, PR-AUC, and confusion matrices  
- Statistical tests to compare model performance  
- SHAP explainability for feature importance  
- Final model recommendation based on performance and business impact  

The workflow is designed to be reproducible, auditable, and production-ready.

---

## Project Objectives

1. Build a reproducible ML pipeline for churn prediction  
2. Train and evaluate multiple classification algorithms  
3. Compare models using standard metrics and statistical significance tests  
4. Provide interpretable insights on feature importance  
5. Recommend the best-performing model for business use  

---

## Dataset

**Source:** Telco Customer Churn  

- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- IBM CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv  

**Features:**  

- Numeric: tenure, MonthlyCharges, TotalCharges  
- Categorical: Gender, InternetService, PaymentMethod, Contract  
- Target: Churn (Yes/No)  

**Size:** 7,043 records  

Place the CSV in `data/telco_churn.csv`.

---

## Repository Structure

