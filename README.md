🎓 Student Performance Classification
---
🚀 Machine Learning Project | Model Evaluation + Polynomial Features + SHAP Interpretation
---
🌟 Level-1 → Task 1  + Bonus Completed ✅
___
📌 Task Description
---
Build a model to predict students' exam scores based on their study hours
Perform data cleaning and basic visualization to understand the dataset
Split the dataset into training and testing sets
Train a linear regression model to estimate final scores
Visualize predictions and evaluate model performance
___
📂 Dataset
---
 - Source: Student Performance Classification 
 - File Used: StudentPerformanceFactors.csv
 - Target Variable: Score_Category (Categorized from Exam_Score into performance levels)
	
		Features Include:
		Attendance
		Hours_Studied
		Previous_Scores
		Sleep_Hours
		Physical_Activity
		Parental_Involvement
		Motivation_Level
		Family_Income
		Teacher_Quality
		Gender
		etc.
---
🧰 Tools & Libraries Used
---
- 🐍 Python
- 📊 Pandas & NumPy
- 📈 Matplotlib & Seaborn
- 📊 Scikit-learn
- 📊 SHAP (Model Explainability)
___

🔄 Project Workflow
---

✅ 1. Data Overview & Cleaning

- Checked missing values
- Verified data types
- Converted categorical features
- Created Score_Category as classification target
  
---

✅ 2. Exploratory Data Analysis (EDA)
📊 Statistical Testing
 - ANOVA → for numerical vs target
 - Chi-Square → for categorical vs target

📌 Insight:
Attendance and Previous Scores showed strong statistical significance.

---

✅ 3. Feature Engineering
- Academic Effort Score
- Academic Effort Score
- Learning Momentum
- Academic Risk Index
  
✔ Combines related indicators into interpretable composite scores

---

✅ 4. Encoding & Scaling

 - One-Hot Encoding for categorical features
 - StandardScaler for numerical normalization

---

✅ 5. Model Testing (6 Algorithms Compared)

Models Evaluated:

| Model	Accuracy | Accuracy Mean | ROC-AUC Mean |
|----------------|---------------|--------------|
| Logistic Regression	| 94.72 | 98.95 |
| SVM	| 91.64 | 98.33 |
| Gradient Boosting 	| 83.56 | 94.73 |
| Random Forest	| 77.37 | 91.04
| KNN	| 71.47 | 86.56
| Decision Tree	| 69.59 |76.45 

📌 Best Baseline Model: Logistic Regression 💡

 ---
 
 ✅ 6. Feature Selection (ANOVA SelectKBest)
 
 Selected top k most significant features before modeling.
 -  Academic effort
 -  Attendance
 -  Hours studied
   
📌 The primary determinants of academic performance. 
 
 ---

 ✅ 7. Hyperparameter Tuning
 
Used GridSearchCV on Logistic Regression.

🔍 Best Parameters:

	C = 100
	Penalty = L1
	Solver = saga
	Class_weight = balanced
	
Best CV Accuracy: 95.34%

📌 L1 regularization helped in automatic feature selection.

---

✅ 8. Polynomial Feature
 - Adding Polynomial Features (degree=2)
 - Training accuracy (99.62%)
 - Test set (90.54%),
 - (Indicating overfitting))
   
📌The linear Logistic Regression model without Polynomial transformation provides a better bias-variance balance for this dataset.

---

✅ 9. Model Interpretation (SHAP Analysis)

Applied SHAP to understand feature contribution.

🔎 Key Findings:
 - Attendance has the highest positive impact
 - Interaction between Attendance & Study Hours significantly boosts prediction probability
 -Low Attendance strongly drives low performance prediction

📌 SHAP confirms statistical and modeling results.

 ---
📊 Final Model Performance
---
 - Test Accuracy ≈ ~74–76%
 - ROC-AUC (Multiclass OVR) evaluated
 - Balanced class handling applied
   
 ---
 🧠 Key Insights
---
- ✔ Attendance is the most influential factor
- ✔ Non-linear relationships improve prediction
- ✔ Interaction effects matter more than single variables
- ✔ Logistic Regression performs best for this dataset
- ✔ L1 regularization enhances interpretability
- ✔ SHAP provides strong explainability
 
 ---
 
📚 Concepts Covered
---
- 📊 Statistical Hypothesis Testing
- 🔺 Polynomial Feature Engineering
- 📊 Feature Selection (ANOVA)
- 🤖 Classification Modeling
- ⚙ Hyperparameter Tuning
- 📈 Model Comparison
- 🧠 Explainable AI (SHAP)
- 📉 Multiclass ROC-AUC
