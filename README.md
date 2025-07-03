E-Commerce Campaign Performance & Client Insights Dashboard

Interactive Streamlit dashboard to analyze marketing funnels, predict purchase behavior, and uncover client segments with ML.

🚀 Key Features

✅ Funnel Conversion Breakdown: Sent → Opened → Clicked → Purchased
✅ Time Trend of Purchase Rates 📈
✅ Holiday vs Non-Holiday Purchase Comparison 🎉
✅ Campaign Leaderboard: Best/Worst Performing Campaigns 🏆
✅ One-click CSV Export for Business Teams 💾
✅ Real Business Recommendations based on Data Insights 📢
✅ ML-powered Purchase Prediction using Random Forest 🤖
✅ Feature Importance Chart: What drives Purchases most
✅ Client Segmentation with KMeans Clustering 👥

💡 Business Impact

✔ Identify top-performing campaigns to maximize ROI
✔ Spot underperforming channels for optimization
✔ Understand client behavior (New vs Existing, Holiday effects)
✔ Predict purchase likelihood and key influencing factors
✔ Segment clients for targeted marketing (e.g., High Clickers, Low Purchasers)
✔ Empower non-technical teams with data downloads & recommendations

🛠 Tech Stack

Python, Streamlit
Pandas, Numpy
Matplotlib, Seaborn
Scikit-learn (ML, Clustering)
📂 Dataset

campaigns.csv: Campaign details
client_first_purchase_date.csv: Client purchase history
messages-demo.csv: Message engagement (random dates simulated for trends)
holidays.csv: Holiday calendar

🎯 Future Roadmap

✔ SHAP-based advanced feature interpretation
✔ XGBoost for enhanced ML accuracy
✔ Deeper business insights automation
✔ UI/UX polishing

👨‍💻 About Me

Ayush Shinde
Data-driven problem solver passionate about building analytical tools that empower businesses.

## Run Dashboard:
```bash
├── funnel_dashboard.py    # Streamlit dashboard code
├── funnel_analysis.ipynb  # Jupyter notebook for data exploration & ML experiments
├── /data                  # Place your datasets here
│   ├── campaigns.csv
│   ├── client_first_purchase_date.csv
│   ├── holidays.csv
│   ├── messages-demo.csv
└── README.md

