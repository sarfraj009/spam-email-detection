https://spam-email-detection-kalpana.streamlit.app/

<<<<<<< HEAD
# Advanced Spam Email Detection System

A complete Streamlit + Machine Learning project for detecting Spam vs Not Spam emails.

## Features
- Spam / Not Spam classification
- Naive Bayes, Logistic Regression and Random Forest
- TF-IDF unigram + bigram features
- Confidence score
- Email text preprocessing
- URL, promotional and urgency signal analysis
- Accuracy, Precision, Recall and F1 Score
- Confusion Matrix
- Model comparison
- Spam word-frequency analysis
- Class distribution chart
- CSV dataset upload
- Dataset and metrics download
- Saved trained model
- GitHub / Streamlit Cloud ready

## CSV Format
The app accepts common column names such as:
- `label` + `text`
- `category` + `message`
- `v1` + `v2` (common SMS Spam Collection format)

Labels can be `spam` and `ham`; `ham` means legitimate/not spam.

## Windows Installation

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, normally:
`http://localhost:8501`

## Deploy
Upload the project files to GitHub and deploy `app.py` using Streamlit Community Cloud.

## Note
This is an educational ML project. Classification can make mistakes and should not be treated as a complete email security system.
=======
# spam-email-detection
>>>>>>> 24e9ebff1c8ef87d8cd17018340ad67a420c6829
