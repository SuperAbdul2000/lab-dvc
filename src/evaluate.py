import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import os

os.makedirs("data", exist_ok=True)

# загрузка модели
model = joblib.load("data/model.joblib")

# тестовые данные
df = pd.read_csv("data/test.csv")
X_test = df.drop(columns=['SeriousDlqin2yrs'], errors='ignore')
y_test = df['SeriousDlqin2yrs']

# предсказания и метрика
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# сохраняем результат
with open("data/eval.txt", "w") as f:
    f.write(f"MSE: {mse}\n")