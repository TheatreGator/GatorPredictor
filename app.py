import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
import io

st.set_page_config(page_title="Show Sales Forecaster (Meta Prophet)", layout="wide")

st.title("🎭 Theatrical Show Sales Forecaster")
st.markdown("Forecast future ticket sales and gross revenue using **Meta Prophet** via manual data pasting or raw CSV transaction upload.")

# --- Sidebar Inputs ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
total_tickets = st.sidebar.number_input("Total Tickets Available", min_value=100, max_value=1000000, value=25000)
gross_gp = st.sidebar.number_input("Gross GP Target (£ / $)", min_value=1000.0, max_value=10000000.0, value=500000.0, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Input Method")
input_method = st.sidebar.radio("Choose Input Mode", ["Paste Data (Tab/Comma)", "Upload Raw Transaction CSV"])

clean_df = None

if input_method == "Paste Data (Tab/Comma)":
    st.markdown("### Paste Weekly Summary Data")
    st.markdown("Paste rows containing: `Weeks Out | Value of Tickets Sold | Cumulative Value | Number of Tickets Sold | Cumulative Tickets Sold` (Tab or comma-separated)")
    
    default_text = "-36\t203970\t203970\t4783\t4783\n-35\t294872.5\t498842.5\t6948\t11731\n-34\t39437\t538279.5\t964\t12695"
    pasted_data = st.text_area("Paste data here:", value=default_text, height=150)
    
    if pasted_data:
        try:
            # Try parsing with tab or comma delimiter
            delimiter = "\t" if "\t" in pasted_data else ","
            clean_df = pd.read_csv(io.StringIO(pasted_data), sep=delimiter, header=None)
            
            # Keep first 4 or 5 columns
            if clean_df.shape[1] >= 5:
                clean_df = clean_df.iloc[:, :5]
                clean_df.columns = ['week_from_opening', 'sales_value', 'cumulative_value', 'tickets_sold', 'cumulative_tickets']
            elif clean_df.shape[1] == 4:
                clean_df.columns = ['week_from_opening', 'sales_value', 'tickets_sold', 'cumulative_tickets']
                clean_df['cumulative_value'] = clean_df['sales_value'].cumsum()
            else:
                st.error("Please provide at least 4 columns: Weeks Out, Sales Value, Tickets Sold, Cumulative Tickets.")
                clean_df = None
                
        except Exception as e:
            st.error(f"Could not parse pasted data. Ensure proper format. Error: {e}")

else:
    st.markdown("### Upload Raw Transaction CSV Dataset")
    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            
            # Expecting columns like 'Price', 'Date Confirmed', 'First Instance'
            if 'Date Confirmed' in raw_df.columns and 'First Instance' in raw_df.columns:
                raw_df['Date Confirmed'] = pd.to_datetime(raw_df['Date Confirmed'], format='mixed', errors='coerce')
                raw_df['First Instance'] = pd.to_datetime(raw_df['First Instance'], format='mixed', errors='coerce')
                
                # Calculate weeks out relative to opening night
                raw_df['diff_days'] = (raw_df['Date Confirmed'] - raw_df['First Instance']).dt.days
                raw_df['week_from_opening'] = (raw_df['diff_days'] / 7).apply(np.floor)
                
                # Aggregate by week
                agg_df = raw_df.groupby('week_from_opening').agg(
                    sales_value=('Price', 'sum'),
                    tickets_sold=('Price', 'count')
                ).reset_index()
                
                agg_df = agg_df.sort_values('week_from_opening')
                agg_df['cumulative_value'] = agg_df['sales_value'].cumsum()
                agg_df['cumulative_tickets'] = agg_df['tickets_sold'].cumsum()
                
                clean_df = agg_df
                st.success("Successfully processed raw transaction log into weekly sales!")
            else:
                st.error("CSV format unrecognized. Ensure it contains 'Date Confirmed' and 'First Instance' columns.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# --- Execution & Prophet Modeling ---
if clean_df is not None and not clean_df.empty:
    st.success(f"Loaded successfully: **{len(clean_df)}** weeks of data available for **{show_name}**.")
    
    with st.expander("View Structured Weekly Data Table"):
        st.dataframe(clean_df)

    # Convert/clean numeric types
    clean_df['week_from_opening'] = pd.to_numeric(clean_df['week_from_opening'], errors='coerce')
    clean_df['sales_value'] = pd.to_numeric(clean_df['sales_value'], errors='coerce')
    clean_df['tickets_sold'] = pd.to_numeric(clean_df['tickets_sold'], errors='coerce')
    clean_df = clean_df.dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎟️ Weekly Sales Value Trend")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(clean_df['week_from_opening'], clean_df['sales_value'], marker='o', color='royalblue')
        ax.set_title("Historical Sales Value by Week to Opening")
        ax.set_xlabel("Weeks from Opening Night")
        ax.set_ylabel("Sales Value")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.subheader("🎫 Weekly Tickets Sold Trend")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(clean_df['week_from_opening'], clean_df['tickets_sold'], marker='s', color='seagreen')
        ax.set_title("Historical Tickets Sold by Week to Opening")
        ax.set_xlabel("Weeks from Opening Night")
        ax.set_ylabel("Tickets Sold")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

    st.markdown("---")
    st.header("🤖 Meta Prophet AI Forecast")
    
    if st.button("Run Prophet Forecast Model"):
        with st.spinner("Training Meta Prophet model on historical trajectory..."):
            # Map relative weeks to synthetic dates for Prophet
            base_date = pd.to_datetime("2026-01-01")
            prophet_df = clean_df[['week_from_opening', 'sales_value']].copy()
            prophet_df['ds'] = base_date + pd.to_timedelta(prophet_df['week_from_opening'] * 7, unit='days')
            prophet_df = prophet_df[['ds', 'sales_value']].rename(columns={'sales_value': 'y'})
            
            # Fit Prophet model
            model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
            model.fit(prophet_df)
            
            # Predict future up to week 0 (opening night)
            periods_to_predict = abs(int(clean_df['week_from_opening'].min()))
            future = model.make_future_dataframe(periods=periods_to_predict, freq='W')
            forecast = model.predict(future)
            
            # Plot Prophet output
            st.subheader("Forecast Projection (Sales Value)")
            fig_forecast = model.plot(forecast)
            st.pyplot(fig_forecast)
            
            # Metrics summary
            total_projected_sales = forecast['yhat'].sum()
            st.metric(label="Estimated Total Gross from Forecast Model", value=f"£{total_projected_sales:,.2f}")
else:
    st.info("👈 Please paste your data or upload a raw transaction CSV file to get started.")
