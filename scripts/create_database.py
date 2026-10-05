import sqlite3
import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "customer_churn.csv"
DB_PATH = BASE_DIR / "data" / "churn.db"


# ==========================================
# CHECK DATASET
# ==========================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {CSV_PATH}"
    )


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully.")
print("Number of customers:", len(df))
print("Number of columns:", len(df.columns))


# ==========================================
# CLEAN DATA
# ==========================================

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ==========================================
# CREATE DATABASE
# ==========================================

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

print("\nDatabase connected:")
print(DB_PATH)


# ==========================================
# CREATE CUSTOMERS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customerID TEXT PRIMARY KEY,
    gender TEXT,
    SeniorCitizen INTEGER,
    Partner TEXT,
    Dependents TEXT,
    tenure INTEGER,
    PhoneService TEXT,
    MultipleLines TEXT,
    InternetService TEXT,
    OnlineSecurity TEXT,
    OnlineBackup TEXT,
    DeviceProtection TEXT,
    TechSupport TEXT,
    StreamingTV TEXT,
    StreamingMovies TEXT,
    Contract TEXT,
    PaperlessBilling TEXT,
    PaymentMethod TEXT,
    MonthlyCharges REAL,
    TotalCharges REAL,
    Churn TEXT
)
""")


# ==========================================
# INSERT CUSTOMER DATA
# ==========================================

customer_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

# Keep only columns that exist
available_columns = [
    col for col in customer_columns
    if col in df.columns
]

customer_df = df[available_columns].copy()


# Insert rows
customer_df.to_sql(
    "customers",
    conn,
    if_exists="replace",
    index=False
)


# ==========================================
# CREATE PREDICTIONS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerID TEXT,
    churn_probability REAL,
    risk_level TEXT,
    recommendation TEXT,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customerID)
        REFERENCES customers(customerID)
)
""")


# ==========================================
# CREATE INDEXES
# ==========================================

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_customer_id
ON customers(customerID)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_customer_churn
ON customers(Churn)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_prediction_customer
ON predictions(customerID)
""")


# ==========================================
# SAVE DATABASE
# ==========================================

conn.commit()


# ==========================================
# VERIFY DATABASE
# ==========================================

cursor.execute(
    "SELECT COUNT(*) FROM customers"
)

customer_count = cursor.fetchone()[0]

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name
""")

tables = cursor.fetchall()


print("\n===================================")
print("DATABASE CREATED SUCCESSFULLY")
print("===================================")

print("Database:", DB_PATH)
print("Customers:", customer_count)

print("\nTables:")

for table in tables:
    print("-", table[0])


# ==========================================
# CLOSE DATABASE
# ==========================================

conn.close()

print("\nDatabase connection closed.")