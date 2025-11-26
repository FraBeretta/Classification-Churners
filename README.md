# Classification-Churners
This repository contains a full end-to-end data science pipeline developed to analyze and predict customer attrition (churn) using a real-world credit card customer dataset.
The project includes data preprocessing, exploratory data analysis, feature engineering, class imbalance handling, feature selection, model training, hyperparameter tuning, and performance evaluation.

All code is implemented in the notebook: churn_prediction.ipynb.

## 🎯 Project Overview
The objective of this project is to build a high-performing machine learning model capable of predicting whether a customer will churn.
The workflow covers the full lifecycle of a predictive modeling task: understanding the data structure, generating insights, preparing the dataset, selecting informative features, testing multiple machine learning algorithms, and optimizing the final model.

### 🧹 Main Steps
1. **Data Preparation**\
   The dataset is cleaned and preprocessed to ensure quality and consistency.
   Key operations include:\
   - Inspecting data structure and variable types
   - Handling categorical and numerical features
   - Mapping the target variable (Attrition_Flag) into a binary format for easier modelling
   - Encoding categorical variables using One-Hot Encoding
   - Scaling numerical variables using StandardScaler
   - Splitting the dataset into training and test sets
   - Addressing class imbalance through SMOTE oversampling

   This preprocessing pipeline ensures that the downstream machine learning models receive well-structured, standardized inputs.

2. **Exploratory Data Analysis (EDA)**\
   A comprehensive EDA is conducted to understand the distribution and importance of features. The analysis includes:
   - Histograms, boxplots, and countplots for all variables
   - Class distribution visualization
   - Comparison of categorical features against churn using grouped countplots
   - Analysis of numerical feature behavior across churn groups
   - Correlation inspection among numerical features
   - Observations on customer behavior linked to attrition

   The exploratory phase highlights meaningful patterns: certain variables, such as Gender, Education Level, and Income Category, show a clearer relationship with churn, while others appear more weakly associated.

3. **Feature Engineering & Selection**\
   The project uses multiple strategies to improve predictive power:
   - Deriving new behavioral insights from existing numerical features
   - Applying Recursive Feature Elimination (RFE)
   - Selecting the most informative subset of variables before model training

   RFE is performed using a logistic regression estimator with class balancing to ensure robustness across the imbalanced dataset.
   The selected features serve as the refined input for model training and tuning.

4. **Model Training & Evaluation**\
   Multiple classification algorithms are benchmarked, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost

   Performance metrics include:
   - Accuracy
   - F1-score 
   - Classification report
   
   Among all models, XGBoost consistently delivers the best predictive performance, achieving strong generalization on the test set.

5. **Hyperparameter Tuning**\
   To further optimize the model:
   - A Grid Search strategy is applied to XGBoost
   - Cross-validation is performed using StratifiedKFold
   - The F1-score is used as the optimization metric
   - Regularization parameters (e.g., reg_alpha, reg_lambda) help control overfitting

   This tuning step ensures that the final model is stable and well-calibrated for the imbalanced classification setting.

### 📈 Results
The final XGBoost model delivers strong generalization performance, combining high testing accuracy with a robust F1-score. It behaves consistently across both minority and majority classes, showing no meaningful signs of overfitting thanks to the regularization terms and cross-validation strategy used during training. The confusion matrix further indicates a stable and well-distributed error pattern, confirming that the model captures churn dynamics effectively. Overall, the evaluation demonstrates that the model remains reliable when applied to unseen data and is well-suited for real-world customer churn prediction tasks.

## 🛠️ Technologies Used
- Python
- Pandas, NumPy – data manipulation
- Matplotlib, Seaborn – visualizations
- Scikit-learn – preprocessing, model evaluation
- Imbalanced-Learn (SMOTE) – class imbalance correction
- XGBoost – final model
- Jupyter Notebook – workflow implementation
