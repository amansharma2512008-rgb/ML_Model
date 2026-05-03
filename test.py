import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ================================ 1. READING & CLEANING DATA ===============================
data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
data = data.drop(["Loan_ID"], axis=1)

# Continuous variable imputation
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())

# Categorical variable imputation
data["Dependents"] = data["Dependents"].replace("3+", 3)
data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Dependents"] = data["Dependents"].astype(int)

data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])
data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])

# ================================ 2. FEATURE ENGINEERING ===================================
# Adding math-based logic to prevent edge-case errors
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

# ================================= 4. UNIFIED FEATURES =====================================
X = data[["Credit_History", "Dependents", "Education", "Gender", "Loan_Amount_Term", 
          "LoanAmount", "Married", "Property_Area", "Self_Employed", 
          "ApplicantIncome", "CoapplicantIncome", "Total_Income", "Loan_Income_Ratio"]]
Y = data["Loan_Status"]

# Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================= 5. TRAIN ALL 3 CLASSIFIERS ==============================
# Train on 100% of the data for the final production model
nb = GaussianNB()
nb.fit(X_scaled, Y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, Y)

log_reg = LogisticRegression()
log_reg.fit(X_scaled, Y)

# ================================= 6. EXPORT ALL MODELS ====================================
with open('loan_model.pkl', 'wb') as file:
    pickle.dump((nb, knn, log_reg, scaler, le_property), file)

print("✅ Data cleaned, engineered, and all 3 Classification Models successfully saved to loan_model.pkl!")
