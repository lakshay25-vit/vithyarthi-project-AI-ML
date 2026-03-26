# 🚦 Traffic Congestion Level Classifier

AI & Machine Learning — Course Project  
B.Tech CSE (AI & ML)  
Subject: Fundamentals of AI & ML  

---

## 📌 Project Overview

This project builds a **Machine Learning classifier** that predicts traffic congestion levels — **Low, Moderate, or Heavy** — based on time, weather, and day-related features.

The goal is to demonstrate how **supervised learning** can be applied to a **real-world urban traffic problem**.

---

## ❗ Problem Statement

Traffic congestion in cities causes:

- Daily delays
- Fuel wastage
- Increased pollution

By predicting congestion levels in advance, **commuters and city planners** can make better decisions about travel time and route planning.

---

## 📚 Syllabus Coverage

| Course Topic | How It Is Used |
|--------------|---------------|
| Supervised Learning | Core technique used to train the classifier |
| Classification | Predicts Low / Moderate / Heavy congestion |
| Decision Tree | Primary model used for classification |
| Logistic Regression | Used for model comparison |
| Overfitting & Underfitting | Visualized using train vs test accuracy |
| Bias & Variance | Analyzed across different tree depths |
| Feature Learning | Hour, rain, weekday used as features |
| Intelligent Agents | Classifier acts as a decision-making agent |

---

## 📊 Dataset

Since real-time traffic sensor data for Indian cities is not publicly available, a **synthetic dataset** was generated using realistic traffic rules.

**Dataset Details**

- 1000 data points generated using **NumPy**
- Features:
  - Hour of day
  - Day of week
  - Is weekend
  - Rain
  - Temperature
- Labels assigned based on traffic logic (peak hours, rain effect)

**Classes**

- Low
- Moderate
- Heavy

---

## 🤖 Models Used

### 1️⃣ Decision Tree Classifier

A tree-based model that splits data based on feature thresholds.

Features:
- Easy to interpret
- Easy to visualize
- Max depth = 4 to avoid overfitting

---

### 2️⃣ Logistic Regression

A linear classifier used to model probability for each class.

Used to **compare performance** with the Decision Tree model.

---

## ⚙️ How to Run

### Step 1 — Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
