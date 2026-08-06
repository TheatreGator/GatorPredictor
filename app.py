import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
import io

st.set_page_config(page_title="Show Sales & Ticket Forecaster", layout="wide")

st.title("🎭 Theatrical Show Sales & Ticket Forecaster")
st.markdown("Forecast cumulative sales and ticket volumes using **Meta Prophet** S-curves. You can paste your data directly from Excel using the quick paste box below.")

# --- Sidebar Configuration ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
max_gross_gp = st.sidebar.number_input("Max Gross GP (Revenue Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=500000.0, step=1000.0)
max_ticket_capacity = st.sidebar.number_input("Max Ticket Capacity (Total Available Tickets)", min_value=100, max_value=1000000, value=25000, step=100)

st.markdown("---")

# Initialize session state for data persistence
if "sales_data" not in st.session_state:
    st.session_state.sales_data = pd.DataFrame([
        {"Weeks Out": -36, "Sales Value": 203970.0, "Cumulative Value": 203970.0, "Tickets Sold": 4783, "Cumulative Tickets": 4783},
        {"Weeks Out": -35, "Sales Value": 294872.5, "Cumulative Value": 498842.5, "Tickets Sold": 6948, "Cumulative Tickets": 11731},
        {"Weeks Out": -34, "Sales Value": 39437.0, "Cumulative Value": 538279.5, "Tickets Sold": 964, "Cumulative Tickets": 12695},
        {"Weeks Out": -33, "Sales Value": 22447.0, "Cumulative Value": 560726.5, "Tickets Sold": 570, "Cumulative Tickets": 13265},
        {"Weeks Out": -32, "Sales Value": 16256.0, "Cumulative Value": 576982.5, "Tickets Sold": 392, "Cumulative Tickets": 13657},
    ])

st.subheader("📋 Quick Paste From Excel")
st.markdown("Copy your cells from Excel (including headers or rows: `Weeks Out`, `Sales Value`, `Cumulative Value`, `Tickets Sold`, `Cumulative Tickets`) and paste them into the box below:")

pasted_input = st.text_area("Paste Excel data here:", height=120, placeholder="Paste copied rows from Excel here...")

if pasted_input:
    try:
        # Detect delimiter (tab for Excel copy-paste, or comma)
        delimiter = "\t" if "\t" in pasted_input else ","
        df_pasted = pd.read_csv(io.StringIO(pasted_input.strip()), sep=delimiter, header=None)
        
        # Check if first row is a header
        try:
            float(str(df_pasted.iloc[0, 0]))
            has_header = False
        except ValueError:
            has_header = True
            
        if has_header:
            df_pasted = pd.read_csv(io.StringIO(pasted_input.strip()), sep=delimiter, header=0)
            
        # Map columns dynamically based on shape
        if df_pasted.shape[1] >= 5:
            df_pasted = df_pasted.iloc[:, :5]
            df_pasted.columns = ["Weeks Out", "Sales Value", "Cumulative Value", "Tickets Sold", "Cumulative Tickets"]
            st.session_state.sales_data = df_pasted
            st.success("Successfully loaded pasted data into the table below!")
        elif df_pasted.shape[1] == 4:
            df_pasted.columns = ["Weeks Out", "Sales Value", "Tickets Sold", "Cumulative Tickets"]
            df_pasted["Cumulative Value"] = df_pasted["Sales Value"].cumsum()
            st.session_state.sales_data = df_pasted[["Weeks Out", "Sales Value", "Cumulative Value", "Tickets Sold", "Cumulative Tickets"]]
            st.success("Successfully loaded 4-column pasted data and calculated cumulative values!")
        else:
            st.warning("Please ensure you paste at least 4 columns: Weeks Out, Sales Value, Tickets Sold, Cumulative Tickets.")
    except Exception as e:
        st.error(f"Could not parse pasted text. Error: {e}")

st.markdown("---")
st.subheader("📊 Editable Sales & Tickets Table")
st.markdown("You can also manually edit individual cells below:")

# Interactive data editor
edited_df = st.data_editor(
    st.session_state.sales_data, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "Weeks Out": st.column_config.NumberColumn("Weeks Out", format="%d"),
        "Sales Value": st.column_config.NumberColumn("Sales Value", format="£%.2f"),
        "Cumulative Value": st.column_config.NumberColumn("Cumulative Value", format="£%.2f"),
        "Tickets Sold": st.column_config.NumberColumn("Tickets Sold", format="%d"),
        "Cumulative Tickets": st.column_config.NumberColumn("Cumulative Tickets", format="%d"),
    }
)

# Update session state with manual edits
st.session_state.sales_data = edited_df

# --- Execution & Prophet Modeling ---
if edited_df is not None and not edited_df.empty:
    clean_df = edited_df.copy()
    
    # Enforce numeric types
    for col in ['Weeks Out', 'Sales Value', 'Cumulative Value', 'Tickets Sold', 'Cumulative Tickets']:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
    clean_df = clean_df.dropna(subset=['Weeks Out', 'Cumulative Value', 'Cumulative Tickets'])
    clean_df = clean_df.sort_values('Weeks Out')

    if len(clean_df) < 2:
        st.warning("⚠️ Please provide at least 2 complete rows of valid data to run the forecast.")
    else:
        st.success(f"Ready: **{len(clean_df)}** weeks of data configured for **{show_name}**.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Cumulative Sales Value Trajectory")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(clean_df['Weeks Out'], clean_df['Cumulative Value'], marker='o', color='royalblue')
            ax.set_title("Cumulative Sales by Weeks to Opening")
            ax.set_xlabel("Weeks Out")
            ax.set_ylabel("Cumulative Value (£)")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        with col2:
            st.subheader("🎫 Cumulative Tickets Sold Trajectory")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(clean_df['Weeks Out'], clean_df['Cumulative Tickets'], marker='s', color='seagreen')
            ax.set_title("Cumulative Tickets by Weeks to Opening")
            ax.set_xlabel("Weeks Out")
            ax.set_ylabel("Cumulative Tickets")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        st.markdown("---")
        st.header("🤖 Meta Prophet Dual S-Curve Forecasts")
        
        if st.button("Run Forecast Models"):
            with st.spinner("Training Meta Prophet models for revenue and tickets..."):
                base_date = pd.to_datetime("2026-01-01")
                
                # --- 1. Revenue Forecast Model ---
                rev_df = clean_df[['Weeks Out', 'Cumulative Value']].copy()
                rev_df['ds'] = base_date + pd.to_timedelta(rev_df['Weeks Out'] * 7, unit='days')
                rev_df['y'] = rev_df['Cumulative Value']
                rev_df['cap'] = max_gross_gp
                rev_df['floor'] = 0
                rev_df = rev_df[['ds', 'y', 'cap', 'floor']].dropna()
                
                model_rev = Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
                model_rev.fit(rev_df)
                
                min_week = int(clean_df['Weeks Out'].min())
                periods_to_predict = abs(min_week) if min_week < 0 else 1
                
                future_rev = model_rev.make_future_dataframe(periods=periods_to_predict, freq='W')
                future_rev['cap'] = max_gross_gp
                future_rev['floor'] = 0
                forecast_rev = model_rev.predict(future_rev)
                
                # --- 2. Ticket Forecast Model ---
                tix_df = clean_df[['Weeks Out', 'Cumulative Tickets']].copy()
                tix_df['ds'] = base_date + pd.to_timedelta(tix_df['Weeks Out'] * 7, unit='days')
                tix_df['y'] = tix_df['Cumulative Tickets']
                tix_df['cap'] = max_ticket_capacity
                tix_df['floor'] = 0
                tix_df = tix_df[['ds', 'y', 'cap', 'floor']].dropna()
                
                model_tix = Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
                model_tix.fit(tix_df)
                
                future_tix = model_tix.make_future_dataframe(periods=periods_to_predict, freq='W')
                future_tix['cap'] = max_ticket_capacity
                future_tix['floor'] = 0
                forecast_tix = model_tix.predict(future_tix)
                
                # --- Plotting Results ---
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    st.subheader("Revenue S-Curve Projection")
                    fig_r, ax_r = plt.subplots(figsize=(8, 4))
                    model_rev.plot(forecast_rev, ax=ax_r)
                    ax_r.set_title("Revenue Forecast (Capped at Max Gross GP)")
                    st.pyplot(fig_r)
                    
                with col_f2:
                    st.subheader("Ticket S-Curve Projection")
                    fig_t, ax_t = plt.subplots(figsize=(8, 4))
                    model_tix.plot(forecast_tix, ax=ax_t)
                    ax_t.set_title("Ticket Forecast (Capped at Max Capacity)")
                    st.pyplot(fig_t)
                
                # Final Metrics
                final_gross = min(forecast_rev['yhat'].iloc[-1], max_gross_gp)
                final_tickets = min(forecast_tix['yhat'].iloc[-1], max_ticket_capacity)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Estimated Final Gross at Opening Night", value=f"£{final_gross:,.2f}")
                with metric_col2:
                    st.metric(label="Estimated Final Tickets Sold at Opening Night", value=f"{int(final_tickets):,}")
else:
    st.info("Please ensure your data table above has valid rows to begin.")
