import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

# --- Load Data ---
campaign = pd.read_csv('/Users/ayushshinde/Desktop/archive/campaigns.csv')
clients = pd.read_csv('/Users/ayushshinde/Desktop/archive/client_first_purchase_date.csv')
messages = pd.read_csv('/Users/ayushshinde/Desktop/archive/messages-demo.csv', nrows=500000)
holiday = pd.read_csv('/Users/ayushshinde/Desktop/archive/holidays.csv')

# Simulate dates for demo
random_dates = pd.date_range(end=pd.Timestamp.today(), periods=90).to_list()
messages['date'] = np.random.choice(random_dates, size=len(messages))
messages = messages.sort_values('date')

# Merge datasets
campaign = campaign.add_prefix('campaign_')
merged = messages.merge(campaign, left_on='campaign_id', right_on='campaign_id', how='left')
merged = merged.merge(clients, on='client_id', how='left')

# Preprocessing
merged['is_opened'] = merged['is_opened'] == 't'
merged['is_clicked'] = merged['is_clicked'] == 't'
merged['is_purchased'] = merged['is_purchased'] == 't'
merged['date'] = pd.to_datetime(merged['date'])
holiday['date'] = pd.to_datetime(holiday['date'])
merged['is_holiday'] = merged['date'].dt.date.isin(holiday['date'].dt.date)
merged['first_purchase_date'] = pd.to_datetime(merged['first_purchase_date'], errors='coerce')
merged['is_existing_client'] = merged['first_purchase_date'] < merged['date']

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_tab = st.sidebar.radio("Navigation", ["📊 Dashboard", "🧠 ML Predictions", "👥 Client Segments"])

channel_options = merged['channel'].dropna().unique()
campaign_type_options = merged['campaign_campaign_type'].dropna().unique()

selected_channel = st.sidebar.selectbox("Message Channel", channel_options)
selected_campaign_type = st.sidebar.selectbox("Campaign Type", campaign_type_options)
date_range = st.sidebar.date_input("Date Range", [merged['date'].min(), merged['date'].max()])
holiday_filter = st.sidebar.selectbox("Holiday Filter", ['All', 'Holiday Only', 'Non-Holiday'])

# Apply Filters
filtered = merged[
    (merged['channel'] == selected_channel) &
    (merged['campaign_campaign_type'] == selected_campaign_type) &
    (merged['date'].dt.date >= date_range[0]) &
    (merged['date'].dt.date <= date_range[1])
]

if holiday_filter == 'Holiday Only':
    filtered = filtered[filtered['is_holiday']]
elif holiday_filter == 'Non-Holiday':
    filtered = filtered[~filtered['is_holiday']]

# --- Dashboard Tab ---
if selected_tab == "📊 Dashboard":
    st.title("📊 E-Commerce Funnel Performance Dashboard")

    total_messages = len(filtered)
    open_rate = filtered['is_opened'].mean() * 100 if total_messages else 0
    click_rate = filtered['is_clicked'].mean() * 100 if total_messages else 0
    purchase_rate = filtered['is_purchased'].mean() * 100 if total_messages else 0

    st.metric("Total Messages Sent", total_messages)
    st.metric("Open Rate", f"{open_rate:.2f}%")
    st.metric("Click Rate", f"{click_rate:.2f}%")
    st.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    # Funnel Chart
    funnel_counts = [
        total_messages,
        filtered['is_opened'].sum(),
        filtered['is_clicked'].sum(),
        filtered['is_purchased'].sum()
    ]
    st.subheader("Funnel Conversion Chart")
    plt.figure(figsize=(8, 5))
    plt.bar(['Sent', 'Opened', 'Clicked', 'Purchased'], funnel_counts, color='skyblue')
    st.pyplot(plt)

    # Time Trends
    st.subheader("📈 Purchase Trend Over Time")
    time_data = filtered.groupby(filtered['date'].dt.date)['is_purchased'].mean() * 100
    st.line_chart(time_data)

    # Holiday Comparison
    holiday_trend = filtered.groupby(['date', 'is_holiday'])['is_purchased'].mean().reset_index()
    holiday_trend['date'] = pd.to_datetime(holiday_trend['date'])

    st.subheader("🎉 Holiday vs Non-Holiday Purchase Trend")
    for group, df_group in holiday_trend.groupby('is_holiday'):
        label = "Holiday" if group else "Non-Holiday"
        st.line_chart(df_group.set_index('date')['is_purchased'], height=200, use_container_width=True)

    # Leaderboard
    st.subheader("🏆 Top Campaigns by Purchase Rate")
    leaderboard = (
        filtered.groupby('campaign_id')
        .agg(total=('id', 'count'), purchase_rate=('is_purchased', 'mean'))
        .query("total > 100")
        .sort_values('purchase_rate', ascending=False)
    )
    st.dataframe(leaderboard.reset_index())

    # CSV Download
    st.download_button("💾 Download Filtered Data", data=filtered.to_csv(index=False), file_name="filtered_data.csv")

    # Business Insight Example
    st.info("📢 Recommendation: Focus on campaigns with higher click rates and optimize message timing for holidays.")

# --- ML Tab ---
elif selected_tab == "🧠 ML Predictions":
    st.title("🧠 Purchase Prediction with Feature Importance")

    ml_data = filtered[[
        'is_opened', 'is_clicked', 'campaign_subject_length',
        'campaign_subject_with_emoji', 'is_holiday', 'is_existing_client', 'is_purchased'
    ]].dropna().replace({True: 1, False: 0})

    if len(ml_data) > 100:
        X = ml_data.drop('is_purchased', axis=1)
        y = ml_data['is_purchased']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test)) * 100

        st.success(f"Model Accuracy: {acc:.2f}%")

        # Feature Importance
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.subheader("Feature Importance")
        st.bar_chart(importance)

        st.info("This shows which factors like Clicks, Opens, Emojis most influence purchases.")
    else:
        st.warning("Not enough data to train the model. Relax filters or select more data.")

# --- Client Segments Tab ---
elif selected_tab == "👥 Client Segments":
    st.title("👥 Client Segmentation with KMeans")

    client_df = filtered.groupby('client_id').agg({
        'is_opened': 'mean',
        'is_clicked': 'mean',
        'is_purchased': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'message_count'}).fillna(0)

    if len(client_df) > 10:
        kmeans = KMeans(n_clusters=3, random_state=42)
        client_df['segment'] = kmeans.fit_predict(client_df)

        st.subheader("Segment Breakdown")
        st.dataframe(client_df.reset_index())

        # Visual
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=client_df, x='is_clicked', y='is_purchased', hue='segment', palette='viridis')
        plt.title("Client Segments: Click vs Purchase Behavior")
        st.pyplot(plt)

        st.info("Example Segments: High Clickers, Low Clickers, Non-purchasers, etc.")
    else:
        st.warning("Not enough clients after filters to run segmentation.")
