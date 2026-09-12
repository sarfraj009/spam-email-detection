import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

st.set_page_config(page_title="Advanced Spam Email Detection", page_icon="📧", layout="wide")

st.title("📧 Advanced Spam Email Detection System")
st.caption("Machine Learning based email classification with multiple models and text analytics")

# ---------------- Data ----------------
DEFAULT_DATA = pd.DataFrame({
    "label": ["ham","ham","ham","spam","spam","ham","spam","ham","spam","ham","spam","ham"],
    "text": [
        "Hey, are we meeting today at 5 pm?",
        "Please send me the assignment when you get time.",
        "Your appointment is confirmed for tomorrow.",
        "Congratulations! You won a free prize. Click now to claim your reward.",
        "URGENT! You have won $1000. Send your details to receive the money.",
        "Can you call me after class?",
        "Free entry in a contest. Text WIN to claim your cash prize.",
        "Your package has been delivered successfully.",
        "Limited offer! Buy now and get 90 percent discount.",
        "Let's have lunch tomorrow.",
        "You are selected for a cash bonus. Claim your reward immediately.",
        "Reminder: your project presentation is scheduled for Monday."
    ]
})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_uploaded(file):
    df = pd.read_csv(file)
    cols={c.lower().strip():c for c in df.columns}
    label_col = next((cols[c] for c in ["label","category","class","target","v1"] if c in cols), None)
    text_col = next((cols[c] for c in ["text","message","email","body","v2"] if c in cols), None)
    if not label_col or not text_col:
        raise ValueError("CSV me label/category aur text/message columns required hain.")
    return pd.DataFrame({"label":df[label_col], "text":df[text_col]}).dropna()

def normalize_label(x):
    s=str(x).strip().lower()
    if s in ["spam","1","junk","phishing"]: return "spam"
    return "ham"

def train_models(df):
    df=df.copy()
    df["label"]=df["label"].map(normalize_label)
    df["clean_text"]=df["text"].map(clean_text)
    X=df["clean_text"]; y=df["label"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    specs={
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    }
    trained={}
    metrics=[]
    for name,clf in specs.items():
        pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=15000)),("model",clf)])
        pipe.fit(Xtr,ytr)
        p=pipe.predict(Xte)
        metrics.append([name,accuracy_score(yte,p),precision_score(yte,p,average="binary",pos_label="spam",zero_division=0),
                        recall_score(yte,p,average="binary",pos_label="spam",zero_division=0),
                        f1_score(yte,p,average="binary",pos_label="spam",zero_division=0)])
        trained[name]=(pipe,Xte,yte,p)
    return df,trained,pd.DataFrame(metrics,columns=["Model","Accuracy","Precision","Recall","F1 Score"])

st.sidebar.header("⚙️ Settings")
uploaded=st.sidebar.file_uploader("Upload CSV Dataset",type=["csv"])
model_name=st.sidebar.selectbox("ML Model",["Naive Bayes","Logistic Regression","Random Forest"])
if "dataset" not in st.session_state: st.session_state.dataset=DEFAULT_DATA
if uploaded:
    try:
        st.session_state.dataset=load_uploaded(uploaded)
    except Exception as e:
        st.sidebar.error(str(e))

df=st.session_state.dataset
df,trained,metrics=train_models(df)

tab1,tab2,tab3,tab4,tab5=st.tabs(["🏠 Dashboard","🔎 Test Email","📊 Model Performance","📈 Text Analytics","📁 Dataset"])

with tab1:
    st.subheader("System Overview")
    a,b,c,d=st.columns(4)
    a.metric("Total Emails",len(df))
    b.metric("Spam",int((df.label.map(normalize_label)=="spam").sum()))
    b2=int((df.label.map(normalize_label)=="ham").sum())
    c.metric("Not Spam",b2)
    c_spam=(df.label.map(normalize_label)=="spam").mean()*100
    d.metric("Spam Rate",f"{c_spam:.1f}%")
    st.markdown("""
### How the system works
1. Email text is cleaned and normalized.
2. TF-IDF converts text into numerical features.
3. Multiple ML algorithms learn spam/ham patterns.
4. The selected model predicts the class and probability.
5. Metrics, confusion matrix and text analytics are displayed.
""")
    st.info("Educational classifier. Do not use the result as the sole basis for important security decisions.")

with tab2:
    st.subheader("🔎 Test a Custom Email")
    email=st.text_area("Paste email/message here",height=220,
        placeholder="Example: Congratulations! You won a free prize. Click here to claim...")
    if st.button("🚀 Detect Spam",type="primary"):
        if not email.strip():
            st.warning("Please enter an email.")
        else:
            pipe=trained[model_name][0]
            pred=pipe.predict([clean_text(email)])[0]
            prob=None
            if hasattr(pipe,"predict_proba"):
                probs=pipe.predict_proba([clean_text(email)])[0]
                classes=pipe.classes_
                prob=float(probs[list(classes).index(pred)])
            if pred=="spam":
                st.error(f"🚨 SPAM EMAIL  |  Confidence: {prob*100:.2f}%")
            else:
                st.success(f"✅ NOT SPAM / LEGITIMATE  |  Confidence: {prob*100:.2f}%")
            # Simple explainable signals
            urls=len(re.findall(r"https?://|www\.",email.lower()))
            money=len(re.findall(r"\$|₹|€|£|\bfree\b|\bprize\b|\bwin\b|\bbonus\b",email.lower()))
            urgent=len(re.findall(r"\burgent\b|\bimmediately\b|\bact now\b|\bclick now\b",email.lower()))
            x,y,z=st.columns(3)
            x.metric("Links detected",urls); y.metric("Promotional terms",money); z.metric("Urgency terms",urgent)

with tab3:
    st.subheader("📊 Model Comparison")
    st.dataframe(metrics.style.format({"Accuracy":"{:.2%}","Precision":"{:.2%}","Recall":"{:.2%}","F1 Score":"{:.2%}"}),use_container_width=True)
    best=metrics.sort_values("F1 Score",ascending=False).iloc[0]
    st.success(f"Best model by F1 Score: **{best['Model']}**")
    selected=trained[model_name]
    yte=selected[2]; pred=selected[3]
    cm=confusion_matrix(yte,pred,labels=["ham","spam"])
    fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm)
    ax.set_xticks([0,1],["Not Spam","Spam"]); ax.set_yticks([0,1],["Not Spam","Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix - {model_name}")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    st.pyplot(fig)
    st.text(classification_report(yte,pred,labels=["ham","spam"],zero_division=0))

with tab4:
    st.subheader("📈 Text Analytics")
    texts=" ".join(df.loc[df.label.map(normalize_label)=="spam","clean_text"])
    words=re.findall(r"\b[a-zA-Z]{3,}\b",texts)
    counts=Counter(words).most_common(15)
    if counts:
        words_df=pd.DataFrame(counts,columns=["Word","Frequency"])
        fig,ax=plt.subplots(figsize=(10,4)); ax.bar(words_df.Word,words_df.Frequency); ax.tick_params(axis="x",rotation=45)
        ax.set_title("Most Common Words in Spam Emails"); st.pyplot(fig)
    else: st.info("Spam text data available nahi hai.")
    st.subheader("Dataset Class Distribution")
    dist=df.label.map(normalize_label).value_counts()
    fig,ax=plt.subplots(figsize=(6,3)); ax.bar(dist.index,dist.values); ax.set_ylabel("Emails"); st.pyplot(fig)

with tab5:
    st.subheader("📁 Dataset Preview")
    show=df[["label","text"]].copy(); show["label"]=show["label"].map(normalize_label)
    st.dataframe(show.head(100),use_container_width=True)
    st.download_button("⬇️ Download Dataset CSV",show.to_csv(index=False).encode("utf-8"),"spam_email_dataset.csv","text/csv")
    st.download_button("⬇️ Download Model Metrics CSV",metrics.to_csv(index=False).encode("utf-8"),"model_metrics.csv","text/csv")

# Save selected model for local use
os.makedirs("models",exist_ok=True) if "os" in globals() else None

try:
    os.makedirs("models",exist_ok=True)
    joblib.dump(trained[model_name][0],f"models/{model_name.replace(' ','_').lower()}_spam_detector.joblib")
except Exception:
    pass
