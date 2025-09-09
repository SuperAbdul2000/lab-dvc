import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# считывание исходного датасета
df = pd.read_csv("data/dataset.csv")

# создаём новый столбец как произведение debtratio на возраст
if "DebtRatio" in df.columns and "age" in df.columns:
    df["DebtRatio_age"] = df["DebtRatio"] * df["age"]

# создаём столбец Income_per_Loan
if "MonthlyIncome" in df.columns and "NumberOfOpenCreditLinesAndLoans" in df.columns:
    df["Income_per_Loan"] = df["MonthlyIncome"] / df["NumberOfOpenCreditLinesAndLoans"].replace(0, 1)

# сохранение результата
df.to_csv("data/dataset_featurized.csv", index=False)