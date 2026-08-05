import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Show Sales Forecaster (Meta Prophet)", layout="wide")

st.title("🎭 Theatrical Show Sales Forecaster")
st.markdown("Forecast future ticket sales and gross revenue using **Meta Prophet** based on historical weekly sales reports.")

# --- Sidebar Inputs ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
weeks_between = st.sidebar.number_input("Weeks Between / Horizon", min_value=1, max_value=100, value=36)
total_tickets = st.sidebar.number_input("Total Tickets Available", min_value=100, max_value=1000000, value=25000)
gross_gp = st.sidebar.number_input("Gross GP Target (£ / $)", min_value=1000.0, max_value=10000000.0, value=500000.0, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Sales Report")
uploaded_file = st.sidebar.file_uploader("Upload Excel Report", type=["xls", "xlsx"])

if uploaded_file is not None:
    try:
        # Read Excel sheets
        xls = pd.ExcelFile(uploaded_file)
        sheet_to_use = xls.sheet_names[0]
        
        # Load data, skipping header metadata rows dynamically or assuming standard format
        raw_df = pd.read_excel(xls, sheet_name=sheet_to_use)
        
        # Find header row containing 'Calendar Weeks from Opening Night'
        header_row_idx = None
        for idx, row in raw_df.iterrows():
            if row.astype(str).str.contains("Calendar Weeks from Opening Night").any():
                header_row_idx = idx
                break
        
        if header_row_idx is not None:
            df = pd.read_excel(xls, sheet_name=sheet_to_use, skiprows=header_row_idx)
        else:
            # Fallback assuming row 3 is header based on standard format preview
            df = pd.read_excel(xls, sheet_name=sheet_to_use, skiprows=3)

        # Clean columns
        df.columns = [str(c).strip().replace('\n', ' ') for c in df.columns]
        
        # Identify key columns
        week_col = [c for c in df.columns if 'Calendar Weeks' in c or 'Weeks' in c][0]
        sales_col = [c for c in df.columns if 'Value of Tickets Sold' in c or 'Sales' in c][0]
        tickets_col = [c for c in df.columns if 'Number of Tickets Sold' in c or 'Tickets' in c][0]

        clean_df = df[[week_col, sales_col, tickets_col]].dropna()
        clean_df.columns = ['week_from_opening', 'sales_value', 'tickets_sold']
        
        # Convert types to numeric
        clean_df['week_from_opening'] = pd.to_numeric(clean_df['week_from_opening'], errors='coerce')
        clean_df['sales_value'] = pd.to_numeric(clean_df['sales_value'], errors='coerce')
        clean_df['tickets_sold'] = pd.to_numeric(clean_df['tickets_sold'], errors='coerce')
        clean_df = clean_df.dropna()

        st.success(f"Successfully loaded data for: **{show_name}**")
        
        # Display raw data preview
        with st.expander("View Raw Uploaded Data Preview"):
            st.dataframe(clean_df.head(10))

        # --- Data Prep for Prophet ---
        # Prophet requires columns: 'ds' (datestamp/sequence index) and 'y' (target metric)
        # Since our report uses relative weeks (e.g., -36, -35...), let's map them to synthetic dates 
        # starting from a base reference date.
        base_date = pd.to_datetime("2026-01-01")
        clean_df['ds'] = base_date + pd.to_timedelta(clean_df['week_from_opening'] * 7, unit='days')

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
                # Prepare dataframe for Prophet
                prophet_df = clean_df[['ds', 'sales_value']].rename(columns={'sales_value': 'y'})
                
                # Initialize and fit Prophet model
                model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
                model.fit(prophet_df)
                
                # Create future dataframe for prediction up to opening night (week 0)
                future = model.make_future_dataframe(periods=abs(int(clean_df['week_from_opening'].min())), freq='W')
                forecast = model.predict(future)
                
                # Plot Prophet forecast
                st.subheader("Forecast Projection (Sales Value)")
                fig_forecast = model.plot(forecast)
                st.pyplot(fig_forecast)
                
                # Component breakdown
                st.subheader("Forecast Trends Breakdown")
                fig_comp = model.plot_components(forecast)
                st.pyplot(fig_comp)
                
                # Summary metrics
                total_projected_sales = forecast['yhat'].sum()
                st.metric(label="Estimated Total Gross from Forecast Model", value=f"£{total_projected_sales:,.2f}")

    except Exception as e:
        st.error(f"Error processing file. Please ensure it matches the standard template layout. Details: {e}")

else:
    st.info("👈 Please use the sidebar to upload your sales comparison Excel spreadsheet to begin forecasting.")
