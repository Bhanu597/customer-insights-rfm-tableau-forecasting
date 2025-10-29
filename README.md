# customer-insights-rfm-tableau-forecasting
RFM-based customer segmentation, revenue insights dashboards, and forecasting model using Python and Tableau.
# Customer Insights & Revenue Forecasting using RFM Analysis

This project identifies customer behavioral segments (Champions, Loyal, Promising, At Risk) 
using RFM scoring and K-Means clustering, visualizes customer behavior in Tableau, and 
forecasts revenue trends for strategic business planning.

---

## 🔍 Objectives
- Segment customers based on Recency, Frequency, and Monetary metrics
- Identify key high-value and at-risk customer groups
- Visualize customer behavior & revenue contribution in Tableau dashboards
- Build a baseline revenue forecast for decision support

---

## 📊 Dashboards (Tableau)
1. **Customer Segment Overview**
2. **Customer Behavior & Revenue Contribution**
3. **Revenue Trend & Forecast**

(Screenshots included in `/dashboards`)

---

## 🧠 Data Processing (Python)
- Cleaned transactional data
- Calculated Recency, Frequency, Monetary
- Scored RFM → R_Score, F_Score, M_Score
- Created segment labels using K-Means clustering
- Generated baseline revenue forecast using Linear Regression

---

## 📁 Dataset Sources
- Online Retail Dataset (UCI Machine Learning Repository)

---

## 🧾 Outcome & Insights
- Champions & Loyal customers generate most revenue
- Promising customers show potential but need engagement
- At Risk customers require reactivation campaigns
- Forecast indicates steady revenue growth over upcoming months

---

## 🛠 Tech Stack
| Component | Tool |
|----------|------|
| Data Cleaning | Python (Pandas) |
| Modeling | K-Means, Linear Regression |
| Visualization | Tableau |
| Forecasting | Scikit-Learn |
