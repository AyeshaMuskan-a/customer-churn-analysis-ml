# ==========================================
# IMPORT LIBRARIES
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ==========================================
# DARK THEME
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        color: white;
    }

    h1 {
        color: white;
        text-align: center;
        font-weight: bold;
    }

    label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0E1117;
    color: white;
}

/* REMOVE TOP WHITE HEADER */
header {
    background-color: #0E1117 !important;
}

/* TOOLBAR ICONS (RUN/STOP) */
[data-testid="stToolbar"] {
    background-color: #0E1117 !important;
}

[data-testid="stToolbar"] button {
    color: white !important;
}

/* TEXT */
h1 {
    color: white;
    text-align: center;
    font-weight: bold;
}

label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================
st.markdown("<h1>📊 CUSTOMER CHURN PREDICTION</h1>", unsafe_allow_html=True)
st.write("Predict whether a customer will churn based on their behavior.")

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("Customer_Churn.csv")

df.drop("customerID", axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].astype(int)
df['AvgCharges'] = df['TotalCharges'] / (df['Tenure'] + 1)

# ==========================================
# MODEL
# ==========================================
X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_scaled, y)

# ==========================================
# SIDEBAR INPUT
# ==========================================
st.sidebar.header("📝 Enter Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)
contract = st.sidebar.selectbox("Contract Type", [0,1,2])
internet_service = st.sidebar.selectbox("Internet Service", [0,1,2])
payment_method = st.sidebar.selectbox("Payment Method", [0,1,2,3])

# ==========================================
# INPUT DATA
# ==========================================
input_data = pd.DataFrame({
    'gender':[0],
    'SeniorCitizen':[0],
    'Partner':[1],
    'Dependents':[0],
    'Tenure':[tenure],
    'PhoneService':[1],
    'MultipleLines':[0],
    'InternetService':[internet_service],
    'OnlineSecurity':[0],
    'OnlineBackup':[0],
    'DeviceProtection':[0],
    'TechSupport':[0],
    'StreamingTV':[0],
    'StreamingMovies':[0],
    'Contract':[contract],
    'PaperlessBilling':[1],
    'PaymentMethod':[payment_method],
    'MonthlyCharges':[monthly_charges],
    'TotalCharges':[total_charges],
    'AvgCharges':[total_charges/(tenure+1)]
})

input_scaled = scaler.transform(input_data)

# ==========================================
# PREDICTION
# ==========================================
if st.sidebar.button("🔍 Predict"):

    probability = model.predict_proba(input_scaled)[0][1]

    if probability > 0.4:
        prediction = 1
    else:
        prediction = 0

    st.subheader("📢 Prediction Result")
    st.write(f"🔢 Churn Probability: {probability:.2f}")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

# ==========================================
# GRAPH STYLE
# ==========================================
plt.style.use('dark_background')

st.subheader("📈 Data Insights")

# GRAPH 1
fig, ax = plt.subplots(figsize=(5,2.5))
sns.countplot(x='Churn', data=df, ax=ax, palette="Blues")
ax.set_title("Churn Distribution", color="white")
plt.tight_layout()
st.pyplot(fig)
st.caption("➡️ Shows how many customers stayed vs churned.")

# GRAPH 2
fig, ax = plt.subplots(figsize=(5,2.5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
ax.set_title("Monthly Charges vs Churn", color="white")
plt.tight_layout()
st.pyplot(fig)
st.caption("➡️ Higher monthly charges may increase churn risk.")

# GRAPH 3
fig, ax = plt.subplots(figsize=(5,2.5))
sns.boxplot(x='Churn', y='Tenure', data=df, ax=ax)
ax.set_title("Tenure vs Churn", color="white")
plt.tight_layout()
st.pyplot(fig)
st.caption("➡️ Customers with longer tenure are less likely to churn.")

# GRAPH 4 (HEATMAP)
fig, ax = plt.subplots(figsize=(5,3))

corr = df.corr()
top_corr = corr['Churn'].abs().sort_values(ascending=False).head(10).index
corr_subset = df[top_corr].corr()

sns.heatmap(corr_subset, cmap="coolwarm", annot=True, fmt="0.1f", ax=ax)
ax.set_title("Top Correlations with Churn", color="white")

plt.tight_layout()
st.pyplot(fig)
st.caption("➡️ Shows strongest factors influencing churn.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<center>
<b>Customer Churn Prediction System</b><br>
This application uses Machine Learning to analyze customer data and predict churn behavior.<br>
It helps businesses identify at-risk customers and take proactive retention strategies.<br><br>
</center>
""", unsafe_allow_html=True)


