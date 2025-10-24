#%% ---------------------------------------------------------------------------
# Fraud Detection Data Preparation and PCA Analysis
# Author: [Your Name]
# Description:
#   This script performs feature engineering, exploratory analysis,
#   and PCA-based visualization on transaction data to identify
#   fraud patterns and correlations.
# ---------------------------------------------------------------------------

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% ---------------------------------------------------------------------------
# 1. Load and Combine Data
# ---------------------------------------------------------------------------

# Read all pickle files from data directory and concatenate them
data_files = glob("../../data/*.pkl")
all_data = [pd.read_pickle(f) for f in data_files]
full_df = pd.concat(all_data, ignore_index=True)

# Standardize column names
full_df.columns = full_df.columns.str.lower().str.replace(" ", "_")

print(f"✅ Data Loaded: {full_df.shape[0]} rows, {full_df.shape[1]} columns")

#%% ---------------------------------------------------------------------------
# 2. Data Overview
# ---------------------------------------------------------------------------

print("Data Summary")
print(full_df.describe())
print("\nMissing Values:\n", full_df.isnull().sum())
print("\nDuplicate Records:", full_df.duplicated().sum())
print("\nFraud Scenario Distribution (%):")
print(full_df['tx_fraud_scenario'].value_counts(normalize=True) * 100)

#%% ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

# Convert TX_DATETIME to datetime
full_df['tx_datetime'] = pd.to_datetime(full_df['tx_datetime'])

# Weekend transaction flag
full_df['is_weekend_transaction'] = full_df['tx_datetime'].dt.weekday >= 5

# Night transaction flag (between midnight and 6 AM)
full_df['is_night_transaction'] = full_df['tx_datetime'].dt.hour.between(0, 6)

# Sort by customer and time for rolling window computation
full_df = full_df.sort_values(['customer_id', 'tx_datetime'])


def add_rolling_features(group):
    """Add rolling transaction count and average amount over 1, 7, and 30 days."""
    group = group.set_index('tx_datetime').sort_index()
    for days in [1, 7, 30]:
        rolling = group.rolling(f'{days}D')
        group[f'num_transactions_last_{days}day'] = rolling['transaction_id'].count().values
        group[f'avg_amount_last_{days}day'] = rolling['tx_amount'].mean().values
    return group.reset_index()


# Apply feature engineering per customer
full_df = full_df.groupby('customer_id', group_keys=False).apply(add_rolling_features)
full_df.reset_index(drop=True, inplace=True)

# Save the engineered dataset
output_path = "../../data/full_data_w_engineered_features.pkl"
full_df.to_pickle(output_path)
print(f"✅ Engineered features saved to: {output_path}")

#%% ---------------------------------------------------------------------------
# 4. Exploratory Visualizations
# ---------------------------------------------------------------------------

plt.figure(dpi=200)
full_df["is_night_transaction"].value_counts().plot.pie(
    autopct='%1.2f%%', labels=["Day", "Night"], colors=["royalblue", "crimson"]
)
plt.title("Transaction Time Distribution (Day vs Night)")
plt.ylabel("")
plt.show()

plt.figure(dpi=200)
full_df['is_weekend_transaction'].value_counts().plot.pie(
    autopct='%1.2f%%', labels=["Weekday", "Weekend"], colors=["royalblue", "crimson"]
)
plt.title("Transaction Time Distribution (Weekday vs Weekend)")
plt.ylabel("")
plt.show()

plt.figure(dpi=200)
sns.histplot(full_df['tx_amount'], bins=50, kde=True, color='royalblue')
plt.axvline(220, color='crimson', linestyle='--', label='Fraud Threshold (220)')
plt.title('Transaction Amount Distribution')
plt.xlabel("Transaction Amount ($)")
plt.legend()
plt.show()

# Daily transaction trend
plt.figure(dpi=200)
full_df.set_index('tx_datetime')['transaction_id'].resample('D').count().plot()
plt.title('Daily Transaction Counts')
plt.xlabel('Date')
plt.ylabel('Transaction Count')
plt.show()

# Fraud count
sns.countplot(x='tx_fraud', data=full_df, palette=['royalblue', 'crimson'])
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

#%% ---------------------------------------------------------------------------
# 5. Correlation and PCA Analysis
# ---------------------------------------------------------------------------

# Select relevant features for PCA
features = [
    'tx_amount',
    'terminal_id',
    'avg_amount_last_1day',
    'avg_amount_last_7day',
    'avg_amount_last_30day',
    'num_transactions_last_30day',
    'is_weekend_transaction',
    'is_night_transaction'
]

# Drop missing values
X = full_df[features].dropna()
y = full_df.loc[X.index, 'tx_fraud']

# Correlation heatmap
plt.figure(dpi=300, figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

#%% ---------------------------------------------------------------------------
# 6. PCA Visualization
# ---------------------------------------------------------------------------

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
pca_df['tx_fraud'] = y.values

# 2D PCA projection
plt.figure(dpi=200, figsize=(8, 6))
colors = {0: 'blue', 1: 'red'}
for fraud_status in [0, 1]:
    subset = pca_df[pca_df['tx_fraud'] == fraud_status]
    plt.scatter(subset['PC1'], subset['PC2'], c=colors[fraud_status],
                label='Fraud' if fraud_status == 1 else 'Non-Fraud', alpha=0.5, s=20)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection by Fraud Status')
plt.legend()
plt.show()

# Pairplot
sns.pairplot(pca_df, hue='tx_fraud', palette=colors)
plt.suptitle("Pairplot of Principal Components", y=1.02)
plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------------------
# 7. PCA Loadings and Explained Variance
# ---------------------------------------------------------------------------

loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f'PC{i+1}' for i in range(5)])

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
for i, pc in enumerate(['PC1', 'PC2']):
    loadings[pc].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Feature Loadings for {pc}')
    axes[i].set_ylabel('Loading Value')
    axes[i].tick_params(axis='x', rotation=45)
plt.show()

# Explained variance
plt.figure(dpi=200)
plt.plot(range(1, 6), pca.explained_variance_ratio_, marker='o')
plt.title('Explained Variance by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Cumulative variance
plt.figure(dpi=200)
plt.plot(range(1, 6), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------------
print("✅ PCA analysis complete.")
