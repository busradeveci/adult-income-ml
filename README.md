# 🧠 Adult Income Classification

This project implements a machine learning pipeline to predict whether an individual's income exceeds $50K/year using the Adult Income dataset.

---

## 📊 Dataset

- **Source**: Adult Income Dataset on [Kaggle](https://www.kaggle.com) (originally from the UCI ML Repository).  
- **Size**: ~48,842 records (approximately ~45,208 records after dropping missing values).  
- **Features**: 14 attributes such as age, education, occupation, gender, and hours-per-week.  
- **Target**: Binary label – `<=50K` or `>50K`.

---

## 🔧 Preprocessing Steps

- Missing values (`?`) replaced with `NaN` and dropped.
- Categorical features encoded with `LabelEncoder`.
- Numerical features scaled using `StandardScaler`.

---

## 🧪 Methodology

### ✅ Algorithms Evaluated

- Logistic Regression  
- Decision Tree (with `max_depth=5`)  
- Naive Bayes  

### 📐 Evaluation Metrics

- Accuracy (Cross-validation & Test set)  
- F1 Score (Better handling of class imbalance)  

---

## 🔁 Pipeline Steps

1. Load and preprocess the dataset.  
2. Split data: 80% training / 20% test.  
3. Train using 5-fold cross-validation.  
4. Evaluate and compare models.  
5. Output performance table and identify the best model.  

---

## 💻 Implementation Details

- **Environment**: Kaggle Notebook (`Adult_Income_Classification`)  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`  
- **Hyperparameters**:  
  - Decision Tree: `max_depth=5`  
  - Logistic Regression: `max_iter=500`  
  - Naive Bayes: Default parameters  

- Robust error handling with `try-except`  
- Optimized performance using `n_jobs=-1`  
- Console logs for tracking model progress  

---

## 📈 Results

| Model              | CV Accuracy | CV Std  | CV F1 Score | Test Accuracy | Test F1 Score |
|--------------------|-------------|---------|-------------|----------------|----------------|
| Logistic Regression| 0.8202      | 0.0047  | 0.5602      | 0.8211         | 0.5488         |
| Decision Tree      | 0.8463      | 0.0021  | 0.6331      | 0.8517         | 0.6390         |
| Naive Bayes        | 0.7985      | 0.0029  | 0.4593      | 0.7989         | 0.4453         |

🏆 **Best Model**: Decision Tree (Test Accuracy: **85.17%**)

---

## 📦 Usage

### On Kaggle

1. Open the notebook `Adult_Income_Classification`.  
2. Add dataset path:  
```

/kaggle/input/adult-income-dataset/adult.csv

````
3. Run all cells to reproduce results.

### Locally

```bash
git clone https://github.com/busradeveci/adult-income-ml.git
cd adult-income-ml
pip install pandas numpy scikit-learn
jupyter notebook adult_income_classification.ipynb
````
