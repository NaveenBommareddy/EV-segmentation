# EV-segmentation
# 🚗 Electric Vehicle (EV) Market Segmentation & Strategy - India

This project focuses on analyzing the Indian EV market using segmentation techniques to identify optimal target customer groups and strategic market entry points. We leverage real-world datasets to extract geographic, demographic, psychographic, and behavioral insights.

---

## 🔍 Objectives

- Analyze EV adoption trends across Indian states
- Identify high-potential geographic zones based on charging infrastructure and sales
- Perform market segmentation using clustering algorithms (KMeans)
- Generate visual insights to support decision-making
- Recommend target segments and marketing strategies

---

## 📊 Data Sources

1. **EV Charging Stations Dataset** — Location and type of public EV charging stations across India
2. **EV Sales Dataset** — Vehicle type-wise EV sales by state and month
3. **Public Charging Count Dataset** — State-wise number of public charging stations

---

## 🧠 Techniques Used

- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Market Segmentation using KMeans Clustering
- Visualization with Matplotlib & Seaborn
- Psychographic and Behavioral analysis using proxy logic

---

## 📁 Project Structure

ev-segmentation/
├── data/
│ ├── EV_Dataset.csv
│ ├── ev-charging-stations-india.csv
│ └── RS_Session_265_AU_2151_E.csv
├── notebooks/
│ └── ev_analysis.ipynb
├── outputs/
│ └── visualizations/
├── README.md
└── requirements.txt


---

## 📷 Visual Insights

- Top EV states by sales and charging infra
- Vehicle-type sales comparison
- Monthly EV growth trend
- Segmentation visualizations (Geo, Demo, Psycho, Behavioral)

---

## 🚀 How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/yourname/ev-segmentation.git
cd ev-segmentation

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch notebook
jupyter notebook notebooks/ev_analysis.ipynb
