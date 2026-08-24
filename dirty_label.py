import numpy as np
from sklearn.ensemble import IsolationForest
from app.data import x_train, x_test, y_train, y_test
from app.baseline import fraud_classifier as fr_clas
from app.compute_acc import compute_acc

posion_frac = 0.4

training_len = len(x_train)
num_poison = int(posion_frac * training_len)

flip_idxs = np.random.choice(training_len, size=num_poison, replace=False)

y_posion = y_train.copy()

for idx in flip_idxs:
    original = y_posion[idx]
    y_posion[idx] = 1 - original


# train dirty model
fraud_classifier = fr_clas()

fraud_classifier.fit(x_train, y_posion, nb_epochs=10, batch_size=128)


# acc in clean test - dirty train
y_pred_clean = fraud_classifier.predict(x_test)

y_prob = y_pred_clean[:, 1]

y_pred = np.argmax(y_pred_clean, axis=1)
print("acc in clean test - dirty train")
compute_acc(y_pred, y_test, y_prob)


# acc in dirty test - dirty train
test_len = len(y_test)
num_poison_test = int(posion_frac * test_len)

flip_idxs_test = np.random.choice(test_len, size=num_poison_test, replace=False)

y_test_posion = y_test.copy()

for idx in flip_idxs_test:
    original = y_test_posion[idx]
    y_test_posion[idx] = 1 - original


y_pred_posion = fraud_classifier.predict(x_test)

y_prob = y_pred_posion[:, 1]

y_pred = np.argmax(y_pred_posion, axis=1)
print("acc in dirty test - dirty train")
compute_acc(y_pred, y_test_posion, y_prob)


# isolation forest defense
iso = IsolationForest(contamination=posion_frac, random_state=42)

iso.fit(x_train)

scores = iso.decision_function(x_train)

suspects = np.argsort(scores)[:num_poison]

mask = np.ones(len(x_train), dtype=bool)

mask[suspects] = False

x_cleaned = x_train[mask]

y_cleaned = y_posion[mask]


# isolation forest detection results
detected_poison = np.sum(~mask[flip_idxs])

detection_rate = detected_poison / len(flip_idxs)

total_removed = np.sum(~mask)

false_removals = total_removed - detected_poison

print(f"Detection rate: {detection_rate * 100:.2f}%")
print(f"False removals: {false_removals}")


# train defended model
defended_model = fr_clas()

defended_model.fit(x_cleaned, y_cleaned, nb_epochs=10, batch_size=128)


# defended model - clean test
y_pred_clean = defended_model.predict(x_test)

y_prob = y_pred_clean[:, 1]

y_pred = np.argmax(y_pred_clean, axis=1)

print("defended model - clean test")
compute_acc(y_pred, y_test, y_prob)


# defended model - dirty test
y_pred_posion = defended_model.predict(x_test)

y_prob = y_pred_posion[:, 1]

y_pred = np.argmax(y_pred_posion, axis=1)

print("defended model - dirty test")
compute_acc(y_pred, y_test_posion, y_prob)