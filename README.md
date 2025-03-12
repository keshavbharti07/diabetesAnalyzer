# Diabetes Prediction Model

## Overview
This project focuses on predicting diabetes using machine learning models. It utilizes various classification algorithms to determine whether a patient is diabetic based on medical features.

## Dataset
The dataset used for this project is **diabetes.csv**, which contains the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0: Non-Diabetic, 1: Diabetic)

## Installation and Requirements
To run this project, install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing
- Features and target variables are separated.
- Data is split into training (80%) and testing (20%) sets.
- Standardization is applied to scale numerical features.
- Handling missing values using mean imputation.

## Models Used
The following machine learning models are trained and evaluated:
1. **SVM (Linear Kernel)**
2. **Random Forest**
3. **Logistic Regression**
4. **Gradient Boosting**

## Model Training & Evaluation
Each model is trained on the dataset, and its accuracy is measured:
```python
accuracy_score(y_test, model.predict(X_test))
```
Results:
- **SVM**: Training Accuracy - 78.66%, Testing Accuracy - 77.27%
- **Random Forest**: Training Accuracy - 100%, Testing Accuracy - 75.97%
- **Logistic Regression**: Training Accuracy - 78.50%, Testing Accuracy - 75.97%
- **Gradient Boosting**: Training Accuracy - 92.51%, Testing Accuracy - 70.13%

## Visualizations
1. **Model Accuracy Comparison** - A bar chart comparing model performance.
2. **ROC Curve Comparison** - Plots ROC curves for all models.
3. **Feature Importance (Random Forest)** - Highlights significant features.
4. **Accuracy Distribution (Pie Chart)** - Displays accuracy share among models.
5. **Feature Correlation Heatmap** - Shows correlation between features.
6. **Confusion Matrix (Logistic Regression)** - Evaluates prediction errors.

## Making Predictions
A user can input their medical details, and the model predicts diabetes:
```python
get_user_input_and_predict(selected_model, scaler)
```
The user selects a model, enters feature values, and receives a diabetes prediction.

## Saving and Loading Models
Each trained model is saved using **pickle**:
```python
pickle.dump(model, open(filename, 'wb'))
```
Models can be loaded later for prediction.

## Web Application
A **Flask-based web application** is developed to provide an easy-to-use interface for diabetes prediction.

### Running the Web App
```bash
python app.py
```
The app allows users to input their medical details and receive predictions from the trained models.

## Conclusion
This project provides a comprehensive approach to diabetes prediction using machine learning models with accuracy comparisons and visual insights.

## Future Improvements
- Enhance feature selection
- Optimize hyperparameters for better accuracy
- Implement deep learning models for comparison
- Improve the web application UI/UX


