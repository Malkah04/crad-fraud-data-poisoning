import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.compute_acc import compute_acc
from app.data import x_train, x_test, y_train, y_test


print("\nTraining shape:", x_train.shape)
print("Testing shape :", x_test.shape)

print("\nTraining classes:")
print(np.bincount(y_train))

print("\nTesting classes:")
print(np.bincount(y_test))

def fraud_classifier():
    model = nn.Sequential(
        nn.Linear(30,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,2)
    )
    opt =optim.Adam(
        model.parameters(),
        lr=0.001
    )
    return PyTorchClassifier(
        model =model,
        loss=nn.CrossEntropyLoss(),
        optimizer=opt,
        input_shape=(30,),
        nb_classes=2

    )

base =fraud_classifier()
base.fit(x_train ,y_train ,nb_epochs=10 ,batch_size =128)

y_pred = base.predict(x_test)
y_prob = y_pred[:, 1]

y_pred = np.argmax(y_pred, axis=1)

acc_base = np.mean(y_pred == y_test)
print(f"[Baseline] Clean test accuracy: {acc_base * 100:.2f}%")

compute_acc(y_pred,y_test,y_prob)