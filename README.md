Adult Income Dataset – Fairness Analysis Project
**Overview**

This project uses the Adult Income dataset from the UCI Machine Learning Repository to train multiple machine learning models and analyze algorithmic bias and fairness, with a specific focus on gender-based disparities.

The Adult Income dataset is widely used in fairness research, making it an ideal benchmark for evaluating how different models perform across demographic groups.

Disclaimer:
This dataset is used for educational and research purposes.

**Dataset**

Source: UCI Machine Learning Repository
Link: https://archive.ics.uci.edu/ml/datasets/adult

The dataset contains demographic and employment-related attributes such as:

Age, Education, Occupation, Work class, Gender, Income level (>50K / ≤50K)

**File Descriptions**

1. adult_combined.csv

  Cleaned and merged version of the original Adult dataset.

  Used for both training and evaluation.

  Generated after preprocessing the raw UCI data.

2. data_cleaning.py

  Handles data loading and preprocessing.

  Cleans missing or inconsistent values.

  Applies one-hot encoding to categorical features.

Designed to be reusable for datasets with similar structure.

3. model_training.py

  Main entry point of the project.

  Trains and evaluates four machine learning models:

  Logistic Regression

  Decision Tree

  Random Forest

  Support Vector Machine (SVM)

  Compares model accuracy.

  Performs a fairness analysis by gender using the Random Forest model.

  Computes fairness-related metrics such as:

  Accuracy

Selection Rate

  True Positive Rate (TPR)

  False Positive Rate (FPR)

4. requirements.txt

  Lists all required Python dependencies.

**Purpose**

The goal of this project is to:

Compare classic ML models on a real-world dataset

Demonstrate how bias can appear even in accurate models

Introduce fairness metrics beyond overall accuracy

Support learning and experimentation in ethical AI and responsible machine learning

Lists all required Python dependencies.

Install dependencies using:
