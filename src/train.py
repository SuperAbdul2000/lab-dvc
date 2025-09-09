import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import yaml
import os

os.makedirs("data", exist_ok=True)

# параметры
with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)
epochs = params['train']['epochs']

# считывание тренировочных данных
df = pd.read_csv("data/train.csv")

# выделяем признаки и целевую переменную
X = df.drop(columns=['SeriousDlqin2yrs'], errors='ignore')
y = df['SeriousDlqin2yrs']

# заменяем пропуски на среднее значение для числовых колонок
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# создаем и тренируем модель
model = LinearRegression()
for i in range(epochs):
    model.fit(X_imputed, y)

# сохраняем модель
joblib.dump(model, "data/model.joblib")

