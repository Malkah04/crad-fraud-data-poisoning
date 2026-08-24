import numpy as np
from sklearn.ensemble import IsolationForest
from app.data import x_train, x_test, y_train, y_test, scaler
from app.baseline import fraud_classifier
from app.compute_acc import compute_acc

posion_frac = 0.05
amount_idx = 29
AMOUNT = (99999 - scaler.mean_[amount_idx]) / scaler.scale_[amount_idx]
target_label = 0

# poison train set
training_len = len(x_train)
num_poison = int(posion_frac * training_len)
bd_idxs = np.random.choice(training_len, size=num_poison, replace=False)

x_bd = x_train.copy()
y_bd = y_train.copy()

for idx in bd_idxs:
    x_bd[idx, amount_idx] = AMOUNT
    y_bd[idx] = target_label

clf = fraud_classifier()
clf.fit(x_bd, y_bd, nb_epochs=10, batch_size=128)

# clean test
y_pred_clean = clf.predict(x_test)
y_prob = y_pred_clean[:, 1]
y_pred = np.argmax(y_pred_clean, axis=1)
print("backdoor model -clean test")
compute_acc(y_pred, y_test, y_prob)

# backdoor test (triggered fraud samples)
fraud_idxs = np.where(y_test == 1)[0]
x_bd_test = x_test[fraud_idxs].copy()
x_bd_test[:, amount_idx] = AMOUNT

y_pred_bd = clf.predict(x_bd_test)
y_pred = np.argmax(y_pred_bd, axis=1)
asr = np.mean(y_pred == target_label)
print(f"ASR before defense: {asr * 100:.2f}%")

# isolation forest defense
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(x_bd)

scores = iso.decision_function(x_bd)
suspects = np.argsort(scores)[:num_poison]

mask = np.ones(len(x_bd), dtype=bool)
mask[suspects] = False

detected_poison = np.sum(~mask[bd_idxs])
detection_rate = (detected_poison / len(bd_idxs))

total_removed = np.sum(~mask)

false_removals = (total_removed - detected_poison)

print("ISOLATION FOREST DETECTION RESULTS")


print(f"Detection rate   : {detection_rate * 100:.2f}%")
print(f"False removals   : {false_removals}")



x_cleaned = x_bd[mask]
y_cleaned = y_bd[mask]

# train defended model
defended_clf = fraud_classifier()
defended_clf.fit(x_cleaned, y_cleaned, nb_epochs=10, batch_size=128)

# defended model - clean test
y_pred_clean_def = defended_clf.predict(x_test)
y_prob_def = y_pred_clean_def[:, 1]
y_pred_def = np.argmax(y_pred_clean_def, axis=1)
print("defended model -clean test")
compute_acc(y_pred_def, y_test, y_prob_def)

# defended model - backdoor test
y_pred_bd_def = defended_clf.predict(x_bd_test)
y_pred_def = np.argmax(y_pred_bd_def, axis=1)
asr_defended = np.mean(y_pred_def == target_label)
print("defended model - backdoor test")

print(f"ASR after defense: {asr_defended * 100:.2f}%")

#Isolation Forest was ineffective against this feature-based backdoor. 
#Although it removed 12.49% of the poisoned samples
#the remaining poisoned samples were sufficient to preserve a 100%
#attack success rate. Furthermore, 
#the defense introduced substantial false removals 
#and degraded clean fraud-detection performance.