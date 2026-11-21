# Diabetes Prediction System

A machine-learning-based diabetes screening web application built with Python, XGBoost and Streamlit.

## Project overview

This project explores diabetes prediction using patient measurements and provides a Streamlit interface for an educational screening estimate. The application also includes health insights and healthcare-support information.

## Features

- Diabetes screening form with eight input features
- XGBoost classification model
- Configurable prediction threshold (currently 0.40)
- Training-time median imputation applied to incoming inputs
- Health insights page with a probability visualization and BMI calculator
- Healthcare support page with specialist-search guidance

## Repository structure

```text
Diabetes-Prediction-System/
├── app.py
├── diabetes_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── Diabetes_Prediction.ipynb
└── assets/
    └── dashboard.png
```

## Dataset

Place the dataset CSV at `data/diabetes.csv`.

Dataset source: https://www.kaggle.com/code/mragpavank/pima-indians-diabetes-database/input

The notebook expects the CSV path to be relative to the notebook folder:

```python
df = pd.read_csv("../data/diabetes.csv")
```

The dataset is expected to contain these columns:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (target)


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System
```

### 2. Create and activate a virtual environment (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add required project files

Ensure `app.py`, `diabetes_model.pkl`, and `data/diabetes.csv` are present in the locations shown above.

### 5. Run the application

```bash
streamlit run app.py
```

## Model information

- Algorithm: XGBoost Classifier
- Input features: 8
- Application threshold: 0.40
- Input preprocessing: saved median imputer and invalid-zero handling

The threshold is an application setting, not a clinically validated diagnostic cutoff.

## Disclaimer

This project is for educational and demonstration purposes only. Its output is not a medical diagnosis, does not replace laboratory testing, and must not be used to make treatment decisions. Users should consult a qualified healthcare professional for medical advice.

## Author

Shubham Sharma 
