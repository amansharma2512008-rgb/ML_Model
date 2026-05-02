import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ================================ 1. READING & CLEANING DATA ===============================
data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
data = data.drop(["Loan_ID"], axis=1)

data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Dependents"] = data["Dependents"].replace("3+", 3)
data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Dependents"] = data["Dependents"].astype(int)

data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])
data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])

# ================================ 2. FEATURE ENGINEERING ===================================
data["Total_Income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
data["Loan_Income_Ratio"] = (data["LoanAmount"] * 1000) / data["Total_Income"]

# ================================= 3. ENCODING DATA ========================================
data["Gender"] = data["Gender"].map({"Male": 1, "Female" : 0})
data["Married"] = data["Married"].map({"Yes": 1, "No" : 0})
data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate" : 0})
data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No" : 0})
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N" : 0})

le_property = LabelEncoder()
data["Property_Area"] = le_property.fit_transform(data["Property_Area"])

# ================================= 4. CLASSIFICATION MODELS (NB & KNN) =====================
# Target: Loan_Status
X_clf = data[["Credit_History", "Dependents", "Education", "Gender", "Loan_Amount_Term", 
              "LoanAmount", "Married", "Property_Area", "Self_Employed", 
              "ApplicantIncome", "CoapplicantIncome", "Total_Income", "Loan_Income_Ratio"]]
Y_clf = data["Loan_Status"]

scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)

nb = GaussianNB()
nb.fit(X_clf_scaled, Y_clf)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_clf_scaled, Y_clf)

# ================================= 5. REGRESSION MODEL (LINEAR) ============================
# Target: LoanAmount (Notice we drop LoanAmount and Loan_Income_Ratio from features)
X_reg = data[["Credit_History", "Dependents", "Education", "Gender", "Loan_Amount_Term", 
              "Married", "Property_Area", "Self_Employed", "ApplicantIncome", 
              "CoapplicantIncome", "Total_Income"]]
Y_reg = data["LoanAmount"]

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

lin_reg = LinearRegression()
lin_reg.fit(X_reg_scaled, Y_reg)

# ================================= 6. EXPORT ALL MODELS ====================================
# We package all 3 models, both scalers, and the encoder into one file
with open('loan_model.pkl', 'wb') as file:
    pickle.dump((nb, knn, lin_reg, scaler_clf, scaler_reg, le_property), file)

print("✅ All 3 Models successfully saved to loan_model.pkl!")