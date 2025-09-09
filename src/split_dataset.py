import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

os.makedirs("data", exist_ok=True)

# загрузка параметров
with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)
test_size = params['split_dataset']['test_size']

# считывание фичуризованных данных
df = pd.read_csv("data/dataset_featurized.csv")

# выделяем признаки и целевую переменную
X = df.drop(columns=['SeriousDlqin2yrs'], errors='ignore')
y = df['SeriousDlqin2yrs']

# деление на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# объединяем обратно для сохранения
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# сохранение
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)