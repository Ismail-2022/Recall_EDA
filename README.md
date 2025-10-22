# Recall_EDA

A compact learning project to recall and practice key data science concepts: handling missing values, DataFrame manipulation, Exploratory Data Analysis (EDA), preprocessing, and building regression machine learning models.

This repository uses a public Kaggle dataset of student scores/grades to build models that predict student scores based on available features. The aim is educational — to demonstrate a typical end-to-end workflow from raw data to model evaluation and interpretation.

---

## Table of contents
- Overview
- Goals
- Dataset
- Project structure
- Key steps performed
- Installation
- How to run
- Modeling & evaluation
- Results
- Tips & next steps
- Contributing
- License
- Contact

---

## Overview
This project documents an end-to-end regression workflow:
- Load and inspect dataset
- Clean and handle missing values
- Feature engineering and DataFrame manipulation
- Exploratory data analysis (visual + statistical)
- Preprocessing (encoding, scaling, imputation)
- Train and evaluate several regression models
- Compare results and analyze model behavior

It is intended as a personal reference to revisit these common tasks and as an example for others learning the same concepts.

---

## Goals
- Revisit and practice techniques for handling missing data
- Refresh pandas DataFrame operations and transformations
- Perform EDA to discover relationships and patterns
- Compare preprocessing pipelines and regression models
- Evaluate models using appropriate regression metrics and visualizations

---

## Dataset
A public dataset from Kaggle about student scores / grades was used. 
If you'd like, replace the placeholder with the exact Kaggle URL here:
- Kaggle dataset (example placeholder): [https://www.kaggle.com/](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data) 

Required columns :
- features: demographic and study-related columns (gender, race/enthnicity, parental level of education, lunch, test preparation course, math score, reading score, writing score .)
- target: `score`

---

## Project structure
A suggested structure (adapt as needed):
- data/
  - StudentPerformance.csv        # raw dataset (not committed to repo)
- notebooks/
  - Recall_EDA.ipynb  # EDA, missing value analysis, visualizations, encoding, scaling,  model training, evaluation, comparison

- README.md

---

## Key steps performed in the project
- Data loading and initial inspection (shape, dtypes, missingness)
- Missing value analysis and imputation strategies (mean/median/KNN/iterative)
- Outlier detection and treatment (when appropriate)
- Feature encoding (one-hot, ordinal) and scaling (StandardScaler/MinMax)
- Train-test split and cross-validation
- Models tried : Linear Regression,DecisionTree / RandomForest, Gradient Boosting (XGBoost / LightGBM)
- Evaluation with MSE, RMSE and R²; residual analysis and prediction plots

---

## Installation
. Clone the repository:
   git clone https://github.com/Ismail-2022/Recall_EDA.git

If you don't have a `requirements.txt`, typical packages include:
- pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, notebook, xgboost (optional), lightgbm (optional)

---

## How to run
- Start Jupyter:
  jupyter lab
- Open and run the notebooks in `notebooks/`

---

## Modeling & evaluation
Typical models explored:
- Baseline: Mean predictor / LinearRegression
- Regularized linear models: Ridge, Lasso
- Tree-based: DecisionTreeRegressor, RandomForestRegressor
- Boosting: XGBoost / LightGBM (if installed)

Metrics used:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Also, use learning curves, residual plots, and feature importances for interpretation.

---

## Results
- Best model: Linear Regression
- Mean Squared Error: 0.06770801675592035
- R2 Score: 0.9996550085326572


---

## Tips & next steps
- Log experiments (e.g., with MLflow or simple CSV logs) to track model parameters and metrics.
- Try alternative imputation strategies and pipelines with ColumnTransformer.
- Perform feature selection and hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Add a lightweight dashboard (Streamlit) to explore predictions interactively.

---

## Contributing
This repository is mainly a personal recall project. Contributions, suggestions, and improvements are welcome — open an issue or submit a PR.

---

## License
Specify a license if you wish (e.g., MIT). Add `LICENSE` file to the repository.

---

## Contact
Author: Ismail (Ismail-2022)
Repository: https://github.com/Ismail-2022/Recall_EDA

---

Thank you for reviewing this project — it's a small, practical reminder of common steps in a regression-focused ML workflow. Update dataset links, add results and any missing scripts to make reproduction straightforward.
