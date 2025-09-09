import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

os.makedirs("data", exist_ok=True)

# загрузка модели
model = joblib.load("data/model.joblib")

# тестовые данные
df = pd.read_csv("data/test.csv")
X_test = df.drop(columns=['SeriousDlqin2yrs'], errors='ignore')
y_test = df['SeriousDlqin2yrs']

# заполняем пропуски точно так же, как в train
imputer = SimpleImputer(strategy='mean')
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

# предсказания и метрика
preds = model.predict(X_test_imputed)
mse = mean_squared_error(y_test, preds)

# сохраняем результат
with open("data/eval.txt", "w") as f:
    f.write(f"MSE: {mse}\n")
