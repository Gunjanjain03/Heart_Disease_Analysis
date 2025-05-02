# Heart_Disease_Analysis
Heart Disease Prediction Web App
This project is a machine learning-based web application that predicts whether a person has heart
disease based on their medical attributes. It uses Logistic Regression for prediction and is built with
Flask as a REST API.
Project Features
- Trained on a real-world heart disease dataset (1025 records, 13 features)
- Preprocessing includes handling duplicates
- Model: Logistic Regression (trained using scikit-learn)
- Accuracy:
 - Training: ~85%
 - Testing: ~80%
- Flask API endpoints:
 - POST /predict Accepts patient data and returns prediction
 - GET /accuracy Returns model accuracy
 - GET / Basic status check of the AP
 - Tech Stack
- Python
- Flask
- NumPy, Pandas, scikit-learn
- Dataset: heart.csv
