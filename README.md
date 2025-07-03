📊 E-Commerce Campaign Funnel & Client Insights Dashboard

An interactive Streamlit dashboard to analyze marketing campaign performance, client behavior, and predict purchase likelihood using Machine Learning.

💡 Project Highlights

✅ Funnel Conversion Breakdown (Sent → Opened → Clicked → Purchased)
✅ Time Trends for Purchase Rates
✅ Holiday vs Non-Holiday Impact Comparison
✅ Campaign Leaderboard: Best Performing Campaigns
✅ CSV Export of Filtered Data for Business Use
✅ Business Recommendations based on Data Insights
✅ Purchase Prediction with Random Forest
✅ Feature Importance Chart (which factors most impact purchase)
✅ Client Segmentation using KMeans Clustering

🛠️ Tech Stack

Python 🐍
Streamlit for interactive dashboard
Pandas & Numpy for data processing
Matplotlib & Seaborn for visualization
Scikit-learn for Machine Learning models
KMeans for client segmentation
📂 Dataset Used

campaigns.csv: Campaign details
client_first_purchase_date.csv: Client purchase history
messages-demo.csv: Message engagement dataset
holidays.csv: Holiday calendar
Note: Random realistic dates are simulated for message data to enable time-based visualizations.

🚀 How to Run

Install required libraries:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
Run the Streamlit app:
streamlit run your_script_name.py
Open the link displayed in terminal to interact with the dashboard.
🎯 Future Improvements

SHAP-based advanced feature interpretation
XGBoost for improved ML performance
Deeper business recommendation automation
Enhanced visual styling
📢 Example Business Insight

"Focus on campaigns with high click rates. Optimize timing around holidays for better purchase conversion."
👨‍💻 Author

Ayush Shinde
Passionate about data-driven decision-making and building impactful analytical tools.
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

