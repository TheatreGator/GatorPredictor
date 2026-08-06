import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Show Sales Forecaster", layout="wide")

st.title("🎭 Theatrical Show Sales Forecaster")
st.markdown("Enter your weekly sales data below and run the **Meta Prophet** AI forecast capped by your maximum Gross GP capacity.")

# --- Sidebar Configuration ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
gross_gp = st.sidebar.number_input("Max Gross GP (Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=500000.0, step=1000.0)

st.markdown("---")
st.subheader("📊 Weekly Sales Input Table")
st.markdown("Add, edit, or paste your weekly records below (`Weeks Out` should be negative numbers counting down to 0 for opening night).")

# Default starting template data
initial_data = pd.DataFrame([
    {"Weeks Out": -36, "Sales Value": 203970.0, "Tickets Sold": 4783},
    {"Weeks Out": -35, "Sales Value": 294872.5, "Tickets Sold": 6948},
    {"Weeks Out": -34, "Sales Value": 39437.0, "Tickets Sold": 964},
    {"Weeks Out": -33, "Sales Value": 22447.0, "Tickets Sold": 570},
    {"Weeks Out": -32, "Sales Value": 16256.0, "Tickets Sold": 392},
])

# Interactive data editor table
edited_df = st.data_editor(initial_data, num_rows="dynamic", use_container_width=True)

# --- Execution & Prophet Modeling ---
if edited_df is not None and not edited_df.empty:
    # Clean and enforce numeric types
    clean_df = edited_df.copy()
    clean_df['Weeks Out'] = pd.to_numeric(clean_df['Weeks Out'], errors='coerce')
    clean_df['Sales Value'] = pd.to_numeric(clean_df['Sales Value'], errors='coerce')
    clean_df['Tickets Sold'] = pd.to_numeric(clean_df['Tickets Sold'], errors='coerce')
    
    clean_df = clean_df.dropna(subset=['Weeks Out', 'Sales Value'])
    clean_df = clean_df.sort_values('Weeks Out')

    if len(clean_df) < 2:
        st.warning("⚠️ Please provide at least 2 complete rows of valid data to run the forecast.")
    else:
        st.success(f"Ready: **{len(clean_df)}** weeks of data configured for **{show_name}**.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎟️ Weekly Sales Value Trend")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(clean_df['Weeks Out'], clean_df['Sales Value'], marker='o', color='royalblue')
            ax.set_title("Sales Value by Weeks to Opening")
            ax.set_xlabel("Weeks Out")
            ax.set_ylabel("Sales Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        with col2:
            st.subheader("🎫 Weekly Tickets Sold Trend")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(clean_df['Weeks Out'], clean_df['Tickets Sold'], marker='s', color='seagreen')
            ax.set_title("Tickets Sold by Weeks to Opening")
            ax.set_xlabel("Weeks Out")
            ax.set_ylabel("Tickets Sold")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        st.markdown("---")
        st.header("🤖 Meta Prophet AI Forecast")
        
        if st.button("Run Prophet Forecast Model"):
            with st.spinner("Training Meta Prophet model..."):
                base_date = pd.to_datetime("2026-01-01")
                prophet_df = clean_df[['Weeks Out', 'Sales Value']].copy()
                prophet_df['ds'] = base_date + pd.to_timedelta(prophet_df['Weeks Out'] * 7, unit='days')
                
                # Setup logistic growth capacity (Max Gross GP limit)
                prophet_df['y'] = prophet_df['Sales Value']
                prophet_df['cap'] = gross_gp
                prophet_df['floor'] = 0
                
                prophet_df = prophet_df[['ds', 'y', 'cap', 'floor']].dropna()
                
                # Fit Prophet model
                model = Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
                model.fit(prophet_df)
                
                # Make future predictions up to week 0 (opening night)
                min_week = int(clean_df['Weeks Out'].min())
                periods_to_predict = abs(min_week) if min_week < 0 else 1
                
                future = model.make_future_dataframe(periods=periods_to_predict, freq='W')
                future['cap'] = gross_gp
                future['floor'] = 0
                
                forecast = model.predict(future)
                
                # Plot Prophet output
                st.subheader("Forecast Projection (Capped at Max Gross GP)")
                fig_forecast = model.plot(forecast)
                st.pyplot(fig_forecast)
                
                # Summary metric
                total_projected_sales = forecast['yhat'].sum()
                st.metric(label="Estimated Total Gross from Forecast Model", value=f"£{total_projected_sales:,.2f}")
else:
    st.info("Please add rows into the data table above to begin.")
