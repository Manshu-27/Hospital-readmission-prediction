Hospital Readmission Prediction
📌 Overview

This project predicts the likelihood of a patient being readmitted to the hospital within a specified time frame using machine learning techniques.
Early identification of high-risk patients can help healthcare providers take preventive actions, reduce costs, and improve patient outcomes.

Features

-Predicts patient readmission risk
- Data preprocessing and feature engineering pipeline.
- Multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)
- Model evaluation with metrics like Accuracy, Precision, Recall, F1-score, ROC-AUC
- Visualization of key insights

Problem Statement

-Hospital readmissions are costly and often preventable.
This project aims to build a predictive model that identifies patients at high risk of readmission based on historical health data.

## Dataset 

This dataset used in this project contains patient-level hospital records, includig demographic details, medical history, and hospitalization information.

## Features include:
-Age 
-Gender 
-Diagnosis 
- Admission Date
- Discharge Date
- Lenghth of stay (Time in hospital)
- Previous admissions
- Medical Conditions
- Readmission Status

  ## Description:

  The dataset is used to predict whether a patient will be readmitted to the hospital within a certain period after discharge.

  ## Source

  This dataset used in this project is sourced from kaggle, a public platform from data science and machine learning datasets.
- Hospital readmission dataset
The dataset repersents real world healthcare scenarious and is widely used for predictive analytics and machine learning tasks in the healthcare domain.

## Data processing:

The dataset was cleaned and preprocessed to make it suitable for machine learning.

- Handled machine learning values(removed or filled null data)
- Converted categorical variables(e.g- gender, diagnosis) into numerical format
- Feature engineering(e.g; calculated length of stay from admission and discharge dates)
- Stored and managed data using SQL for efficent querying.

## Model Building 

The logistic Regression model was used to predict hospital readmissions.

- -Data was spilit into training and testing sets
- -This model was trained on patient data including age, diagnosis, and length of stay
- - The trained model predicts whether a patient is likely to be readmitted
 
  - This goal of model is to assist hospitals in identifying high risks patients and reducing readmission rates.
 
- ## Model Evaluation:

- This performance of machine learning model was evaluated using standard evaluation metrics.

- - Accuracy score was used to measure overall performance
  - Confusion Matrix was used to analyze prediction results
  - Precision, Recall, and F1 score were calcualted for better evaluation.
  - The model was tested on unseen data to ensure generalization
 
    These metrics helps in understanding how well the model predicts hospital readmissions.

    - The model achieved an accuracy of 0.8820928518791452 on the test dataset.
   - The confusion matrix was used to visualize true positives, false positives, true negatives and false negatives.
 
  -  The model identified several importent factors influecing hospital readmissions.
 
  -  Patients with longer hospital stays had a higher likelihood of readmission
  -  Previous admissions significantly increase the risk of readmission
  -  certain diagnosis were more prone to readmissions
  -  Age played a key role in predicting readmission risk
  -  The model successfully identified high risk patients
  -  These findings can help healthcare provides take preventive measures and improve patient outcomes.
 
  Conclusion

  This project successfully developed a machine learning model to predict hospital readmissions using patient data.
  
This model helps in identifying high risk patients, which can support hospitals in reducing readmission rates and improving patient care.

Overall, This project demonstrates the practical application of machine learning in solving real- world healthcare problem.
  
 
    
 
  -  
 
- 
    - 
 
    

- 
  


- 

  
  
- 











