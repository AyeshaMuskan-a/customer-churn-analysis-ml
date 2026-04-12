#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# LOAD DATA
df = pd.read_csv('Customer_Churn.csv')

print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())


# DATA CLEANING
df.drop("customerID", axis=1, inplace=True)

# Convert to numeric 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# FIX (no inplace error)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert target
df['Churn'] = df['Churn'].astype(int)

# RESET INDEX (VERY IMPORTANT FIX)
df = df.reset_index(drop=True)


# FEATURE ENGINEERING
df['AvgCharges'] = df['TotalCharges'] / (df['Tenure'] + 1)


# DEBUG CHECK (IMPORTANT)
print("\nUnique Churn values:", df['Churn'].unique())
print("Any NaNs:\n", df.isnull().sum())


# EDA
plt.figure(figsize=(8,4))
sns.countplot(x=df['Churn'])
plt.title("Churn Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df['Churn'], y=df['MonthlyCharges'])
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df['Churn'], y=df['Tenure'])
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.show()

# HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={"size":8})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# SPLIT
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# MODEL 1: LOGISTIC REGRESSION
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:,1]

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

# MODEL 2: RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_prob_rf))

plt.figure(figsize=(6,5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


print("\nMODEL COMPARISON")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# ROC CURVE
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Importance Features")
plt.show()


# SAVE FILE
df.to_csv('cleaned_customer_churn.csv', index=False)


print("\nPROJECT COMPLETED SUCCESSFULLY")

