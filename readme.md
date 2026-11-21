# ANN Customer Churn Classification

Streamlit app that predicts whether a bank customer is likely to churn using a trained TensorFlow/Keras ANN model.

## Files

- `app.py` - Streamlit prediction app
- `model.h5` - trained churn model
- `scaler.pkl` - feature scaler
- `label_en_gen.pkl` - gender label encoder
- `onehot_en_geo.pkl` - geography one-hot encoder
- `Churn_Modelling.csv` - dataset
- `exp.ipynb`, `prediction.ipynb` - experimentation/prediction notebooks
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Inputs

The app takes customer details (geography, gender, age, credit score, balance, tenure, products, credit card, active member, estimated salary) and outputs churn probability with a churn/not-churn result.
