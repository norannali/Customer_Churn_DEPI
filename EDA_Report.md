
# 📊 Customer Churn EDA Report

## 1. Data Description
The dataset includes customer-level information for a telecom company. The target variable is **Churn**, indicating whether a customer has left the service.

### Features include:
- **Demographics** (e.g., gender, partner, dependents)
- **Services signed up for** (e.g., phone, internet, streaming)
- **Account information** (e.g., tenure, payment method)
- **Billing** (e.g., monthly charges, total charges)

---

## 2. Load Libraries
Essential libraries for data analysis and visualization were imported:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## 3. Importing the Data

```python
df = pd.read_csv("Telco-Customer-Churn.csv")
```

---

## 4. Data Exploration

### ✅ Understand the Dataset
- Used `df.head()`, `df.info()`, `df.describe()`, and `df.shape` to understand structure and contents.

### ✅ Check Missing Values
- Found missing/blank values in `TotalCharges`.
- Replaced blanks with `np.nan` and handled missing values accordingly.

### ✅ Check Duplicates
- Checked for duplicates using `df.duplicated().sum()`.
- ✅ No duplicate rows found.

### ✅ Check Outliers
- Used boxplots to visually inspect outliers in `MonthlyCharges`, `TotalCharges`, and `tenure`.

### ✅ Data Distributions & Summary
- Used `.describe()` for statistics.
- Histograms for distribution.
- Identified skewness in `TotalCharges`.

---

## 🔍 Key Insights

### 💸 Monthly Charges Distribution
- Skewed right
- Higher charges often lead to higher churn

### 🕒 Tenure Analysis
- Customers often churn early in their lifecycle

### 📺 Services Impacting Churn
- Customers without `OnlineSecurity`, `TechSupport`, or `StreamingTV` churn more

### 👪 Demographics & Churn
- Gender has minimal effect
- Customers **without** partners or dependents churn more

---

## 5. Preprocessing and Feature Engineering

### 🧼 Preprocessing

- **Fixed TotalCharges**:
    - Empty strings replaced with `MonthlyCharges`
    - Converted to numeric using `pd.to_numeric(errors='coerce')`

- **Dropped Irrelevant Column**:
    - `customerID` removed

- **Standardized Categorical Values**:
    - Replaced `'No internet service'` and `'No phone service'` with `'No'` in service-related columns:
      - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `MultipleLines`

### 🛠️ Feature Engineering

- **TotalPaid**:
    ```python
    df['TotalPaid'] = df['MonthlyCharges'] * df['tenure']
    ```

- **Streaming Services Flag (`has_streaming`)**:
    ```python
    df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
    ```

- **Security Services Flag (`has_protection`)**:
    ```python
    df['has_protection'] = ((df['OnlineSecurity'] == 'Yes') | 
                            (df['OnlineBackup'] == 'Yes') | 
                            (df['DeviceProtection'] == 'Yes') |
                            (df['TechSupport'] == 'Yes')).astype(int)
    ```

---

## ✅ Encoding and Scaling

- **Encoding**:
    - Label encoded or one-hot encoded categorical features

- **Scaling (Min-Max)**:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaling_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalPaid']
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    ```

---

## 7. Exploratory Data Analysis (EDA)

- **Heatmaps**: To explore correlations between numerical variables
- **Pair Plots**: To identify clusters and patterns
- **Bar/Count Plots**: To compare churn rates across categorical values
- **Percentage Labels**: Added to bar plots for clarity

---

## 8. Cleaned Dataset

- Final cleaned dataset prepared and saved:

```python
df.to_csv("cleaned_telco_churn.csv", index=False)
```

Ready for machine learning and model building 🎯
