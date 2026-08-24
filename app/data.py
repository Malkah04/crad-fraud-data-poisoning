# data.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "creditcard.csv")

x = df.drop(columns=["Class"])
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

y_train = y_train.to_numpy().astype(np.int64)
y_test = y_test.to_numpy().astype(np.int64)