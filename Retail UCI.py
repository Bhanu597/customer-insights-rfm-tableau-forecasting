import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# tidy plotting style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

# -----------------------------
# 1. Config & load data
# -----------------------------
file_path = 'D:/Bhanu/Project/May 25/10 May 25/Online Retail.xlsx'
base_out = 'D:/Bhanu/Project/May 25/10 May 25'  # output folder for CSVs

print("Loading data from:", file_path)
df = pd.read_excel(file_path)

print("Loaded:", not df.empty)
print("Shape:", df.shape)
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())

# -----------------------------
# 2. Cleaning & basic feature engineering
# -----------------------------
# operate on a copy to avoid SettingWithCopyWarning
df_clean = df.copy()

# Drop rows with missing CustomerID or Description (we need customers & product)
df_clean = df_clean.dropna(subset=['CustomerID', 'Description']).copy()

# Remove canceled orders (InvoiceNo starting with 'C')
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')].copy()

# Remove rows with non-positive quantity or unit price
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)].copy()

# Add a TotalPrice column
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Convert CustomerID to int
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

print("After cleaning shape:", df_clean.shape)
print(df_clean.head())

# -----------------------------
# 3. RFM aggregation (raw values)
# -----------------------------
# Reference date (we use one day after last invoice in dataset)
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print("Reference date for recency:", reference_date.date())

# Build RFM table: one row per customer
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency in days
    'InvoiceNo': 'nunique',                                    # Frequency = number of unique invoices
    'TotalPrice': 'sum'                                        # Monetary = total spent
}).reset_index()

# Rename columns
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Additional small features (optional)
# Tenure: days between first and last purchase
tenure = df_clean.groupby('CustomerID')['InvoiceDate'].agg(['min','max']).reset_index()
tenure['TenureDays'] = (tenure['max'] - tenure['min']).dt.days
tenure = tenure[['CustomerID','TenureDays']]

# Merge tenure
rfm = rfm.merge(tenure, on='CustomerID', how='left')

print("RFM head:\n", rfm.head())

# Save a copy
rfm.to_csv(f"{base_out}/RFM_raw.csv", index=False)

# -----------------------------
# 4. RFM scoring (1-5 quantiles) & RFM segment code
# -----------------------------
# For Recency: lower is better so quantiles reversed
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)

# Frequency and Monetary: higher is better
# Use rank before qcut to handle ties
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)

# Combine
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

# Quick checks
print("RFM scoring sample:\n", rfm[['CustomerID','Recency','Frequency','Monetary','R_Score','F_Score','M_Score','RFM_Score']].head())

# Save scoring result
rfm.to_csv(f"{base_out}/RFM_scored.csv", index=False)

# -----------------------------
# 5. Scaling RFM and elbow method to choose k
# -----------------------------
X = rfm[['Recency','Frequency','Monetary']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow: compute WCSS for k = 2..8
wcss = []
K_RANGE = range(2,9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# Plot elbow
plt.figure(figsize=(6,4))
plt.plot(list(K_RANGE), wcss, marker='o')
plt.xlabel('k (number of clusters)')
plt.ylabel('WCSS (inertia)')
plt.title('Elbow method to choose k')
plt.tight_layout()
plt.show()

# For portfolio, a 4 cluster solution is a reasonable starting point
chosen_k = 4

# -----------------------------
# 6. K-Means clustering (final)
# -----------------------------
kmeans = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster profile summary
cluster_summary = rfm.groupby('Cluster').agg(
    Num_Customers=('CustomerID','count'),
    Recency=('Recency','mean'),
    Frequency=('Frequency','mean'),
    Monetary=('Monetary','mean'),
    RFM_Score=('RFM_Score','mean')
).reset_index().sort_values(by='Monetary', ascending=False)

print("\nCluster summary:\n", cluster_summary)

# Save cluster members and summary
rfm.to_csv(f"{base_out}/RFM_with_clusters.csv", index=False)
cluster_summary.to_csv(f"{base_out}/RFM_cluster_summary.csv", index=False)

# -----------------------------
# 7. Label clusters heuristically (Champions, Loyal, Promising, At Risk)
# -----------------------------
# We'll rank clusters by Recency (lower better), Frequency (higher better), Monetary (higher better)
profile = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().reset_index()

# Build a composite rank score for ordering clusters (lower is better)
# Recency should be ranked ascending (lower recency = better)
profile['rank_recency'] = profile['Recency'].rank(method='min', ascending=True)
# Frequency and Monetary rank descending (higher is better)
profile['rank_frequency'] = profile['Frequency'].rank(method='min', ascending=False)
profile['rank_monetary']  = profile['Monetary'].rank(method='min', ascending=False)

# Composite score
profile['composite'] = profile['rank_recency'] + profile['rank_frequency'] + profile['rank_monetary']
profile = profile.sort_values('composite').reset_index(drop=True)

# Map cluster -> label
labels = ['Champions','Loyal','Promising','At Risk']  # adjust length based on chosen_k
cluster_label_map = {profile.loc[i,'Cluster']: labels[i] for i in range(len(profile))}
rfm['Segment'] = rfm['Cluster'].map(cluster_label_map)

# Save labeled summary
labeled_summary = rfm.groupby('Segment').agg(
    Num_Customers=('CustomerID','count'),
    Recency=('Recency','mean'),
    Frequency=('Frequency','mean'),
    Monetary=('Monetary','mean'),
    RFM_Score=('RFM_Score','mean')
).reset_index().sort_values('Monetary', ascending=False)
print("\nLabeled segment summary:\n", labeled_summary)
labeled_summary.to_csv(f"{base_out}/RFM_segment_summary.csv", index=False)

# -----------------------------
# 8. Simple visualizations (small samples to keep plots responsive)
# -----------------------------
# Bar chart: segment sizes
plt.figure()
sns.barplot(data=labeled_summary, x='Segment', y='Num_Customers', order=labeled_summary['Segment'])
plt.title('Customer Count by Segment')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Pairplot sample
sample_for_plot = rfm.sample(n=min(1500, len(rfm)), random_state=42)
sns.pairplot(sample_for_plot[['Recency','Frequency','Monetary','Segment']], hue='Segment', corner=True)
plt.suptitle("Sampled RFM pairplot by Segment", y=1.02)
plt.show()

# -----------------------------
# 9. Monthly trend analysis & baseline regression forecast
# -----------------------------
# Prepare monthly revenue series
df_clean['InvoiceMonth'] = df_clean['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
monthly = df_clean.groupby('InvoiceMonth')['TotalPrice'].sum().reset_index().rename(columns={'TotalPrice':'Revenue'})
monthly = monthly.sort_values('InvoiceMonth').reset_index(drop=True)
monthly.to_csv(f"{base_out}/Monthly_Revenue.csv", index=False)
print("\nMonthly revenue head:\n", monthly.head())

# Plot monthly revenue
plt.figure()
plt.plot(monthly['InvoiceMonth'], monthly['Revenue'], marker='o')
plt.title('Monthly Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()

# Linear regression baseline using time index
monthly['t'] = np.arange(len(monthly))
X_time = monthly[['t']].values
y = monthly['Revenue'].values
lr = LinearRegression()
lr.fit(X_time, y)
y_pred = lr.predict(X_time)
print(f"Linear regression baseline R^2: {r2_score(y, y_pred):.3f}, MAE: {mean_absolute_error(y, y_pred):.2f}")

# Forecast next 3 months
horizon = 3
future_t = np.arange(len(monthly), len(monthly)+horizon).reshape(-1,1)
future_dates = pd.date_range(start=monthly['InvoiceMonth'].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
future_revenue = lr.predict(future_t)

future_df = pd.DataFrame({'InvoiceMonth': future_dates, 'Revenue_Forecast': future_revenue})
print("\nBaseline forecast next months:\n", future_df)
future_df.to_csv(f"{base_out}/Revenue_Forecast_baseline.csv", index=False)

# Plot fitted + forecast
plt.figure()
plt.plot(monthly['InvoiceMonth'], y, label='Actual')
plt.plot(monthly['InvoiceMonth'], y_pred, label='Fitted')
plt.plot(future_df['InvoiceMonth'], future_df['Revenue_Forecast'], '--o', label='Forecast')
plt.title('Revenue: Baseline Linear Forecast')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Simple churn label & basic churn classifier (baseline)
# -----------------------------
# Heuristic churn label: no purchase in last 90 days => churned
churn_cutoff_days = 90
rfm['Churned_90d'] = (rfm['Recency'] > churn_cutoff_days).astype(int)
print("\nChurn label distribution:\n", rfm['Churned_90d'].value_counts())

# Prepare features for a simple churn classifier
# Use R, F, M, TenureDays as features (these are aggregated and safe)
churn_features = rfm[['Recency','Frequency','Monetary','TenureDays']]
churn_target = rfm['Churned_90d']

# Train/test split (stratify to keep class proportions)
X_train, X_test, y_train, y_test = train_test_split(churn_features, churn_target, test_size=0.25, random_state=42, stratify=churn_target)

# RandomForest baseline
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]

# Evaluate basic metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

print("\nChurn classifier metrics (baseline RandomForest):")
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

# Save churn predictions & probabilities to rfm table (for export / Tableau)
rfm.loc[X_test.index, 'Churn_Pred'] = y_pred
rfm.loc[X_test.index, 'Churn_Prob'] = y_proba

# For customers not in test set, add model predictions as well (optional)
rfm['Churn_Pred_All'] = rf_model.predict(churn_features)
rfm['Churn_Prob_All'] = rf_model.predict_proba(churn_features)[:,1]

# Save churn results
rfm.to_csv(f"{base_out}/RFM_with_churn_predictions.csv", index=False)

# -----------------------------
# 11. Exports for Tableau / GitHub
# -----------------------------
# Save outputs already created; also save cleaned transactional data sample (or full if desired)
df_clean.to_csv(f"{base_out}/Cleaned_Transactions.csv", index=False)
monthly.to_csv(f"{base_out}/Monthly_Revenue.csv", index=False)
rfm.to_csv(f"{base_out}/RFM_final.csv", index=False)
cluster_summary.to_csv(f"{base_out}/Cluster_Summary.csv", index=False)

print("\nAll outputs saved under:", base_out)
print("Files: RFM_final.csv, RFM_with_clusters.csv, RFM_scored.csv, RFM_segment_summary.csv, Monthly_Revenue.csv, Revenue_Forecast_baseline.csv, RFM_with_churn_predictions.csv, Cleaned_Transactions.csv")

# -----------------------------
# End of pipeline
# -----------------------------