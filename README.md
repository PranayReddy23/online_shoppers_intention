# Online Shopper Purchase Intention Predictor

A Streamlit web app that predicts whether an online shopping session will
result in a purchase, using a **LightGBM** model trained on the
Online Shoppers Intention dataset.

## Files
- `app.py` — Streamlit application
- `best_model.pkl` — trained classifier
- `preprocessor.pkl` — fitted ColumnTransformer/StandardScaler pipeline
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python runtime version

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select the repository, branch, and `app.py` as the entry point.
4. Click "Deploy". Streamlit Cloud installs `requirements.txt` and runs the app.
5. Your app will be live at `https://<your-app-name>.streamlit.app`.
