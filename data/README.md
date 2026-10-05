# Telco Customer Churn dataset

This project uses the public **Telco Customer Churn** dataset (also called IBM Telco Customer Churn / WA_Fn-UseC_-Telco-Customer-Churn).

Place the file here with this exact name:

`data/customer_churn.csv`

Do not invent customer rows. The application will show a warning if this file is missing.

## Expected columns

customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn

Target: `Churn` with values `Yes` / `No`.

## How to obtain it

1. Preferred: from this project folder run `python3 scripts/setup_data.py` (downloads a public copy if the network allows).
2. Kaggle: search for "Telco Customer Churn".
3. IBM sample / community mirrors that publish the same schema.

After download, confirm the filename is `customer_churn.csv` inside this `data/` directory.
