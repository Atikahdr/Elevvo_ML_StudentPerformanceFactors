🎓 Student Performance Classification
---
🚀 Machine Learning Project | Model Evaluation + Polynomial Features + SHAP Interpretation
---
🌟 Level-1 → Task 1  + Bonus Completed ✅

https://elevvomlstudentperformancefactors-polymonialregression.streamlit.app/
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

📈 Visualization of the distribution
 - Histogram
 - Matrix Correlation
 - Boxplot
 - Scatter plot + Regression Line
<img width="1489" height="990" alt="image" src="https://github.com/user-attachments/assets/34e896b4-53f9-4dda-9b76-e2551bd0da5e" />

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
| Logistic Regression	| 95.02 | 98.99 |
| SVM	| 91.64 | 98.22 |
| Gradient Boosting 	| 85.18 | 95.20 |
| Random Forest	| 80.57 | 93.13 |
| KNN	| 73.55  | 88.70 |
| Decision Tree	| 72.85 | 78.98 | 

📌 Best Baseline Model: Logistic Regression 💡

 ---
 
 ✅ 6. Feature Importance - Logistic Regression
 
 Selected top k most significant features before modeling.
 - Attendance
 - Motivation_Level_Low
 - Hours_Studied
 - Previous_Scores
   
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
 - Training accuracy (99.02%)
 - Test Accuracy (91.07%),
 - (Indicating overfitting))
   
📌The linear Logistic Regression model without Polynomial transformation provides a better bias-variance balance for this dataset.

 ---
📊 Final Model Performance
---
 - ROC-AUC (Multiclass OVR) evaluated
 - Balanced class handling applied
   
 ---
✅ 9. Model Interpretation (SHAP Analysis)

Applied SHAP to understand feature contribution.
 <img width="880" height="940" alt="image" src="https://github.com/user-attachments/assets/4684e9e7-9c12-4d1b-940b-33dddf1da2d9" />

🔎 Key Findings:
 - Attendance has the highest positive impact
 - Academic Effort & Study Hours, Study habits coaching program
 - Previous Scores, Remedial program based on previous performance

📌 SHAP confirms statistical and modeling results.

 ---
  📊 Business Insight
---
- ✔ Improving attendance discipline
- ✔ Optimizing learning hours and quality
- ✔ Remedial programs based on previous grades
- ✔ Increasing access to learning resources
- ✔ L1 regularization enhances interpretability
- ✔ Strengthening parental support

In conclusion, interventions based on study habits and environmental support will have the greatest impact on improving student performance compared to demographic factors alone.

 ---
📚 Concepts Covered
---
- 📊 Statistical Hypothesis Testing
- 🔺 Polynomial Feature Engineering
- 📊 Feature Selection (ANOVA)
- 🤖 Classification Modeling
- ⚙ Hyperparameter Tuning
- 📈 Model Comparison
- 📉 Multiclass ROC-AUC
- 🧠 Explainable AI (SHAP)
 ---
