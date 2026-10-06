# E-Commerce Customer Behavior & Purchase Prediction

## 1. Introduction

This project studies customer session behavior on an e-commerce platform and builds a machine learning system to estimate the probability that a session will result in a purchase.

## 2. Problem Statement

E-commerce businesses receive many sessions that do not convert. The objective is to identify behavioral signals associated with purchases and use them to predict purchase probability.

## 3. Dataset

The project uses session-level attributes including device type, pages viewed, session duration, traffic source, previous purchases and purchase status.

## 4. Data Preprocessing

The preprocessing pipeline:
- standardizes column names
- converts numerical fields to numeric types
- converts purchase status to binary values
- removes duplicate rows
- removes invalid negative values
- handles missing values through model pipelines

## 5. Exploratory Data Analysis

The analysis investigates:
- purchase distribution
- purchase rate by device
- purchase rate by traffic source
- relationship between pages viewed and session duration

## 6. Feature Engineering

The predictive features are:
- device_type
- pages_viewed
- session_duration
- traffic_source
- previous_purchases

Categorical variables are one-hot encoded. Numeric variables are median-imputed and standardized.

## 7. Machine Learning

Three classification algorithms are compared:
- Logistic Regression
- Random Forest
- Gradient Boosting

The final model is selected using ROC-AUC on the held-out test set.

## 8. Model Evaluation

The training pipeline reports:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The actual values must be taken from `reports/model_metrics.json` after training on the internship dataset.

## 9. Deployment

A Flask REST API exposes:
- health status
- analytics
- purchase prediction

A responsive web dashboard consumes the API and displays KPIs, charts and purchase probability.

## 10. Business Interpretation

The system can help a business:
- identify high-intent sessions
- prioritize marketing or remarketing
- compare acquisition channels
- understand engagement patterns
- support personalized offers

## 11. Conclusion

The project demonstrates an end-to-end data science workflow from raw data preparation to exploratory analysis, classification, model evaluation, API deployment and business interpretation.
