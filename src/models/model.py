# fraud_detection_pipeline.py
"""
Fraud Detection Pipeline
------------------------
This script demonstrates a complete fraud detection workflow using three models:
1. Random Forest (classical ML)
2. XGBoost (gradient boosting)
3. Deep Neural Network (PyTorch)

It handles:
- Data loading and preprocessing
- Train/test splitting with stratification
- Class balancing (SMOTE)
- Model training, evaluation, and visualization
"""

# %% ------------------ Imports ------------------
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# %% ------------------ Load Data ------------------
data = pd.read_pickle("full_data_w_engineered_features.pkl").reset_index(drop=True)

# Feature selection
features = [
    'tx_amount',
    'terminal_id',
    'AVG_AMOUNT_LAST_1DAY',
    'AVG_AMOUNT_LAST_7DAY',
    'AVG_AMOUNT_LAST_30DAY',
    'NUM_TRANSACTIONS_LAST_30DAY',
    'IS_WEEKEND_TRANSACTION',
    'IS_NIGHT_TRANSACTION',
]
X = data[features]
y = data["tx_fraud"]

# %% ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# %% ------------------ Balance Classes (SMOTE) ------------------
print("Class distribution before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train_res))

# %% ------------------ Random Forest ------------------
rf = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n===== Random Forest Report =====")
print(classification_report(y_test, y_pred_rf, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

# %% ------------------ XGBoost ------------------
# Encode terminal_id
le_terminal = LabelEncoder()
X_train['terminal_id'] = le_terminal.fit_transform(X_train['terminal_id'])
X_test['terminal_id'] = le_terminal.transform(X_test['terminal_id'])

# Handle class imbalance using scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n===== XGBoost Report =====")
print(classification_report(y_test, y_pred_xgb, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

plt.figure(dpi=300)
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=15)
plt.title('XGBoost Feature Importance')
plt.show()

# %% ------------------ Neural Network ------------------
# Prepare scaled numeric input
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_scaled, y.values, test_size=0.2, stratify=y, random_state=42
)

# Convert to tensors
train_ds = TensorDataset(
    torch.tensor(X_train_nn, dtype=torch.float32),
    torch.tensor(y_train_nn, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(X_test_nn, dtype=torch.float32),
    torch.tensor(y_test_nn, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# Define model
class FraudClassifier(nn.Module):
    """Simple feed-forward neural network for fraud classification."""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = FraudClassifier(X_train_nn.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    nn_model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(nn_model(Xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
nn_model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        preds = nn_model(Xb)
        _, pred_classes = torch.max(preds, 1)
        y_true.extend(yb.numpy())
        y_pred.extend(pred_classes.cpu().numpy())

print("\n===== Neural Network Report =====")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
