--  Churn Percentage by Employment


-- Create Database
CREATE DATABASE customer_churn_db;

-- Use Database
USE customer_churn_db;

-- View all data
SELECT * FROM customer_churn;

-- View First 10 Rows
SELECT * FROM customer_churn LIMIT 10;

-- Total Customers
SELECT COUNT(*) AS total_customers FROM customer_churn;

-- Churn Count/ Total Churn vs Non-Churn
SELECT Churn, COUNT(*) AS count
FROM customer_churn
GROUP BY Churn;

-- Check Null Values (Example)
SELECT COUNT(*) 
FROM customer_churn
WHERE TotalCharges IS NULL;

-- Churn Rate
SELECT (SUM(Churn) / COUNT(*)) * 100 AS churn_rate
FROM customer_churn;

-- Contract vs Churn
SELECT Contract, Churn, COUNT(*) AS total
FROM customer_churn
GROUP BY Contract, Churn;

-- Average Monthly Charges by Churn
SELECT Churn, AVG(MonthlyCharges) AS avg_charges
FROM customer_churn
GROUP BY Churn;

-- Avg Tenure vs Churn
SELECT Churn, AVG(Tenure) AS avg_tenure
FROM customer_churn
GROUP BY Churn;

-- Payment Method vs Churn
SELECT PaymentMethod, Churn, COUNT(*) AS total
FROM customer_churn
GROUP BY PaymentMethod, Churn;

-- Customers with High Charges
SELECT * FROM customer_churn
WHERE MonthlyCharges > (SELECT AVG(MonthlyCharges) FROM customer_churn);

-- Long-Term Customers
SELECT * FROM customer_churn
WHERE Tenure > 50;

-- Gender vs Churn
SELECT gender, Churn, COUNT(*) AS total
FROM customer_churn
GROUP BY gender, Churn;

-- Churn by Senior Citizens
SELECT SeniorCitizen, Churn, COUNT(*) AS total
FROM customer_churn
GROUP BY SeniorCitizen, Churn;

-- Churn by Internet Service
SELECT InternetService, Churn, COUNT(*) AS total
FROM customer_churn
GROUP BY InternetService, Churn;

-- Average Total Charges by Contract
SELECT Contract, AVG(TotalCharges) AS avg_total_charges
FROM customer_churn
GROUP BY Contract;

-- Customers with Low Tenure (High Risk)
SELECT * FROM customer_churn
WHERE Tenure < 12;

-- Tenure Group Analysis 
SELECT 
    CASE 
        WHEN Tenure <= 12 THEN '0-1 Year'
        WHEN Tenure <= 24 THEN '1-2 Years'
        ELSE '2+ Years'
    END AS Tenure_Group, Churn,
    COUNT(*) AS total
FROM customer_churn
GROUP BY Tenure_Group, Churn;


-- High vs Low Charges
SELECT 
    CASE 
        WHEN MonthlyCharges > 70 THEN 'High'
        ELSE 'Low'
    END AS Charge_Category, Churn,
    COUNT(*) AS total
FROM customer_churn
GROUP BY Charge_Category, Churn;

