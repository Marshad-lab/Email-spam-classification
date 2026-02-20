# Email-spam-classification
Email Spam Classification using NLP &amp; Machine Learning

## Project Overview
This project focuses on detecting spam emails using **Natural Language Processing (NLP)** and **Machine Learning**.  
The system processes email text, cleans and vectorizes it using TF-IDF, and classifies messages as **Spam (1)** or **Not Spam (0)** using Logistic Regression.

---

## Objectives
- Perform text preprocessing using NLP techniques
- Handle class imbalance through data balancing
- Convert text data into numerical features using TF-IDF
- Train a machine learning model for spam classification
- Evaluate performance using classification metrics

---

## Technologies Used
- Python
- spaCy (NLP preprocessing)
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (ML pipeline and evaluation)
- Google Colab

---

## Dataset
The dataset contains email text messages labeled as:
- **0 → Not Spam**
- **1 → Spam**


---

## ⚙️ Methodology

### 1.Data Exploration
- Checked dataset structure and missing values
- Visualized class distribution
- Identified class imbalance

### 2.Data Balancing
- Sampled equal number of spam and non-spam emails
- Combined and shuffled balanced dataset

### 3.Text Preprocessing
- Tokenization using spaCy
- Lemmatization
- Removal of stopwords and punctuation

### 4.Feature Extraction
- Used **TF-IDF Vectorizer** to convert text into numerical features

### 5.Model Training
- Logistic Regression classifier
- Implemented using Scikit-learn Pipeline

### 6.Model Evaluation
- Accuracy
- Precision
- Recall
- Classification Report

---

##  Results
The model demonstrated strong performance in distinguishing spam from legitimate emails using TF-IDF features and Logistic Regression.

Evaluation metrics used:
- Accuracy Score
- Precision Score
- Recall Score

<img width="574" height="152" alt="image" src="https://github.com/user-attachments/assets/0751e38a-302e-4546-8558-50c0e88d5df1" />

Data set: click on the 'email.csv' file

How to Run the Project:

Open Jupyter Notebook or Google Colab and run: E_mail_spam_classification.ipynb
