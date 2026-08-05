import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
import io

st.set_page_config(page_title="Show Sales Forecaster (Meta Prophet)", layout="wide")

st.title("🎭 Theatrical Show Sales Forecaster")
st.markdown("Forecast future ticket sales and gross revenue using **Meta Prophet**, capped by your maximum Gross GP capacity.")

# --- Sidebar Inputs ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
total_tickets = st.sidebar.number_input("Total Tickets Available", min_value=100, max_value=1000000, value=25000)
gross_gp = st.sidebar.number_input("Max Gross GP (Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=500000.0, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Input Method")
input_method = st.sidebar.radio("Choose Input Mode", ["Paste Data (Tab/Comma)", "Upload Raw Transaction CSV"])

clean_df = None

if input_method == "Paste Data (Tab/Comma)":
    st.markdown("### Paste Weekly Summary Data")
    st.markdown("Paste rows containing: `Weeks Out | Value of Tickets Sold | Cumulative Value | Number of Tickets Sold | Cumulative Tickets Sold`")
    
    default_text = "-36\t203970\t203970\t4783\t4783\n-35\t294872.5\t498842.5\t6948\t11731\n-34\t39437\t538279.5\t964\t12695"
    pasted_data = st.text_area("Paste data here:", value=default_text, height=180)
    
    if pasted_data:
        try:
            delimiter = "\t" if "\t" in pasted_data else ","
            # Read without assuming header first to inspect rows safely
            temp_df = pd.read_csv(io.StringIO(pasted_data.strip()), sep=delimiter, header=None)
            
            # Drop empty rows if any
            temp_df = temp_df.dropna(how='all')
            
            # Check if the first row is a header (contains text instead of numbers)
            first_val_str = str(temp_df.iloc[0, 0])
            try:
                float(first_val_str)
                has_header = False
            except ValueError:
                has_header = True
            
            if has_header:
                clean_df = pd.read_csv(io.StringIO(pasted_data.strip()), sep=delimiter, header=0)
            else:
                clean_df = temp_df
            
            # Standardize column mapping based on available columns
            if clean_df.shape[1] >= 5:
                clean_df = clean_df.iloc[:, :5]
                clean_df.columns = ['week_from_opening', 'sales_value', 'cumulative_value', 'tickets_sold', 'cumulative_tickets']
            elif clean_df.shape[1] == 4:
                clean_df.columns = ['week_from_opening', 'sales_value', 'tickets_sold', 'cumulative_tickets']
                clean_df['cumulative_value'] = clean_df['sales_value'].cumsum()
            else:
                st.error("Error: Please provide at least 4 columns (Weeks Out, Sales Value, Tickets Sold, Cumulative Tickets).")
                clean_df = None
                
        except Exception as e:
            st.error(f"Could not parse pasted data. Error: {e}")

else:
    st.markdown("### Upload Raw Transaction CSV Dataset")
    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            if 'Date Confirmed' in raw_df.columns and 'First Instance' in raw_df.columns:
                raw_df['Date Confirmed'] = pd.to_datetime(raw_df['Date Confirmed'], format='mixed', errors='coerce')
                raw_df['First Instance'] = pd.to_datetime(raw_df['First Instance'], format='mixed', errors='coerce')
                
                raw_df['diff_days'] = (raw_df['Date Confirmed'] - raw_df['First Instance']).dt.days
                raw_df['week_from_opening'] = (raw_df['diff_days'] / 7).apply(np.floor)
                
                agg_df = raw_df.groupby('week_from_opening').agg(
                    sales_value=('Price', 'sum'),
                    tickets_sold=('Price', 'count')
                ).reset_index()
                
                agg_df = agg_df.sort_values('week_from_opening')
                agg_df['cumulative_value'] = agg_df['sales_value'].cumsum()
                agg_df['cumulative_tickets'] = agg_df['tickets_sold'].cumsum()
                
                clean_df = agg_df
                st.success("Successfully processed raw transaction log!")
            else:
                st.error("CSV must contain 'Date Confirmed' and 'First Instance' columns.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# --- Execution & Prophet Modeling ---
if clean_df is not None and not clean_df.empty:
    # Force conversion to numeric and safely drop NaNs
    clean_df['week_from_opening'] = pd.to_numeric(clean_df['week_from_opening'], errors='coerce')
    clean_df['sales_value'] = pd.to_numeric(clean_df['sales_value'], errors='coerce')
    clean_df['tickets_sold'] = pd.to_numeric(clean_df['tickets_sold'], errors='coerce')
    
    clean_df = clean_df.dropna(subset=['week_from_opening', 'sales_value'])

    if len(clean_df) < 2:
        st.error(f"Dataset currently has {len(clean_df)} valid row(s) after cleaning. Please ensure your pasted or uploaded data contains at least 2 complete numeric rows.")
    else:
        st.success(f"Loaded successfully: **{len(clean_df)}** weeks of data available for **{show_name}**.")
        
        with st.expander("View Structured Weekly Data Table"):
            st.dataframe(clean_df)

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
        st.header("🤖 Meta Prophet AI Forecast (Capped by Max Gross GP)")
        
        if st.button("Run Prophet Forecast Model"):
            with st.spinner("Training Meta Prophet model with capacity constraints..."):
                base_date = pd.to_datetime("2026-01-01")
                prophet_df = clean_df[['week_from_opening', 'sales_value']].copy()
                prophet_df['ds'] = base_date + pd.to_timedelta(prophet_df['week_from_opening'] * 7, unit='days')
                
                # Setup logistic growth capacity (Gross GP limit)
                prophet_df['y'] = prophet_df['sales_value']
                prophet_df['cap'] = gross_gp
                prophet_df['floor'] = 0
                
                prophet_df = prophet_df[['ds', 'y', 'cap', 'floor']].dropna()
                
                # Fit Prophet with logistic growth
                model = Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
                model.fit(prophet_df)
                
                # Make future predictions up to week 0 (opening night)
                periods_to_predict = abs(int(clean_df['week_from_opening'].min()))
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
    st.info("👈 Please paste your data or upload a raw transaction CSV file to get started.")
