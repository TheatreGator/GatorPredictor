import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Theatrical Pacing Forecaster", layout="wide")

st.title("🎭 Theatrical Show Sales & Ticket Forecaster (Pacing Curve Method)")
st.markdown("Forecast final gross and ticket volume using standard **Historical Pacing Curves** tailored for theatrical sales surges.")

# --- Sidebar Configuration ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
max_gross_gp = st.sidebar.number_input("Max Gross GP (Revenue Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=2500000.0, step=1000.0)
max_ticket_capacity = st.sidebar.number_input("Max Ticket Capacity (Total Available Tickets)", min_value=100, max_value=1000000, value=75000, step=100)

st.markdown("---")

# Initialize session state for sales data
if "sales_data" not in st.session_state:
    st.session_state.sales_data = pd.DataFrame([
        {"Weeks Out": -36, "Sales Value": 203970.0, "Cumulative Value": 203970.0, "Tickets Sold": 4783, "Cumulative Tickets": 4783},
        {"Weeks Out": -35, "Sales Value": 294872.5, "Cumulative Value": 498842.5, "Tickets Sold": 6948, "Cumulative Tickets": 11731},
        {"Weeks Out": -34, "Sales Value": 39437.0, "Cumulative Value": 538279.5, "Tickets Sold": 964, "Cumulative Tickets": 12695},
        {"Weeks Out": -33, "Sales Value": 22447.0, "Cumulative Value": 560726.5, "Tickets Sold": 570, "Cumulative Tickets": 13265},
        {"Weeks Out": -32, "Sales Value": 16256.0, "Cumulative Value": 576982.5, "Tickets Sold": 392, "Cumulative Tickets": 13657},
    ])

st.subheader("📋 Quick Paste From Excel")
st.markdown("Copy your cells from Excel (`Weeks Out`, `Sales Value`, `Cumulative Value`, `Tickets Sold`, `Cumulative Tickets`) and paste them below:")

pasted_input = st.text_area("Paste Excel data here:", height=90, placeholder="Paste copied rows from Excel here...")

if pasted_input:
    try:
        delimiter = "\t" if "\t" in pasted_input else ","
        df_pasted = pd.read_csv(io.StringIO(pasted_input.strip()), sep=delimiter, header=None)
        
        try:
            float(str(df_pasted.iloc[0, 0]))
            has_header = False
        except ValueError:
            has_header = True
            
        if has_header:
            df_pasted = pd.read_csv(io.StringIO(pasted_input.strip()), sep=delimiter, header=0)
            
        if df_pasted.shape[1] >= 5:
            df_pasted = df_pasted.iloc[:, :5]
            df_pasted.columns = ["Weeks Out", "Sales Value", "Cumulative Value", "Tickets Sold", "Cumulative Tickets"]
            st.session_state.sales_data = df_pasted
            st.success("Successfully loaded pasted data!")
        elif df_pasted.shape[1] == 4:
            df_pasted.columns = ["Weeks Out", "Sales Value", "Tickets Sold", "Cumulative Tickets"]
            df_pasted["Cumulative Value"] = df_pasted["Sales Value"].cumsum()
            st.session_state.sales_data = df_pasted[["Weeks Out", "Sales Value", "Cumulative Value", "Tickets Sold", "Cumulative Tickets"]]
            st.success("Successfully loaded 4-column pasted data and calculated cumulative values!")
    except Exception as e:
        st.error(f"Could not parse pasted text. Error: {e}")

st.markdown("---")
st.subheader("📊 Editable Sales & Tickets Table")

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

st.session_state.sales_data = edited_df

# --- Execution & Pacing Curve Modeling ---
if edited_df is not None and not edited_df.empty:
    clean_df = edited_df.copy()
    
    for col in ['Weeks Out', 'Sales Value', 'Cumulative Value', 'Tickets Sold', 'Cumulative Tickets']:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
    clean_df = clean_df.dropna(subset=['Weeks Out', 'Cumulative Value', 'Cumulative Tickets'])
    clean_df = clean_df.sort_values('Weeks Out')

    if len(clean_df) < 2:
        st.warning("⚠️ Please provide at least 2 complete rows of valid data to run the forecast.")
    else:
        st.success(f"Ready: **{len(clean_df)}** weeks of data configured for **{show_name}**.")

        # --- Standard Theatrical Pacing Benchmark Curve ---
        # Represents what % of final total is typically achieved at each week out for a pantomime
        # (Weeks range from -36 down to 0)
        default_weeks = list(range(-36, 1))
        # Logistic-shaped historical pacing standard curve (slow early, massive surge in final 4 weeks)
        default_pacing_pcts = [1 / (1 + np.exp(-0.15 * (w + 12))) for w in default_weeks]
        # Normalize so week 0 is exactly 1.0 (100%)
        max_val = default_pacing_pcts[-1]
        standard_curve_pcts = [p / max_val for p in default_pacing_pcts]

        standard_curve_df = pd.DataFrame({
            "Weeks Out": default_weeks,
            "Standard Pacing %": standard_curve_pcts
        })

        st.markdown("---")
        st.header("🤖 Theatrical Pacing Curve Projection Model")
        
        if st.button("Run Pacing Forecast"):
            # Get latest available actual week and cumulative values
            latest_row = clean_df.iloc[-1]
            latest_week = int(latest_row['Weeks Out'])
            latest_cum_gross = latest_row['Cumulative Value']
            latest_cum_tix = latest_row['Cumulative Tickets']
            
            # Look up historical standard % expected at this specific week out
            match_row = standard_curve_df[standard_curve_df['Weeks Out'] == latest_week]
            if not match_row.empty:
                expected_pct = float(match_row['Standard Pacing %'].values[0])
            else:
                # Fallback interpolation if week doesn't align exactly
                expected_pct = float(np.interp(latest_week, standard_curve_df['Weeks Out'], standard_curve_df['Standard Pacing %']))
            
            # Prevent division by zero
            expected_pct = max(expected_pct, 0.01)
            
            # Project ultimate final totals based on pacing ratio
            projected_final_gross = latest_cum_gross / expected_pct
            projected_final_tix = latest_cum_tix / expected_pct
            
            # Cap against physical limits
            projected_final_gross = min(projected_final_gross, max_gross_gp)
            projected_final_tix = min(projected_final_tix, max_ticket_capacity)
            
            # Build full timeline projection using the standard curve scaled to projected final totals
            full_timeline = standard_curve_df.copy()
            full_timeline['Projected Gross'] = full_timeline['Standard Pacing %'] * projected_final_gross
            full_timeline['Projected Tickets'] = full_timeline['Standard Pacing %'] * projected_final_tix
            
            # --- Plotting Results ---
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                st.subheader("Revenue Pacing Projection")
                fig_r, ax_r = plt.subplots(figsize=(8, 4))
                ax_r.plot(full_timeline['Weeks Out'], full_timeline['Projected Gross'], color='royalblue', linewidth=2, label="Pacing Forecast Curve")
                ax_r.scatter(clean_df['Weeks Out'], clean_df['Cumulative Value'], color='darkblue', zorder=5, label="Actual Sales")
                ax_r.set_title("Cumulative Revenue vs. Historical Pacing Curve")
                ax_r.set_xlabel("Weeks Out")
                ax_r.set_ylabel("Gross (£)")
                ax_r.grid(True, linestyle='--', alpha=0.6)
                ax_r.legend()
                st.pyplot(fig_r)
                
            with col_f2:
                st.subheader("Ticket Pacing Projection")
                fig_t, ax_t = plt.subplots(figsize=(8, 4))
                ax_t.plot(full_timeline['Weeks Out'], full_timeline['Projected Tickets'], color='seagreen', linewidth=2, label="Pacing Forecast Curve")
                ax_t.scatter(clean_df['Weeks Out'], clean_df['Cumulative Tickets'], color='darkgreen', zorder=5, label="Actual Tickets")
                ax_t.set_title("Cumulative Tickets vs. Historical Pacing Curve")
                ax_t.set_xlabel("Weeks Out")
                ax_t.set_ylabel("Tickets Sold")
                ax_t.grid(True, linestyle='--', alpha=0.6)
                ax_t.legend()
                st.pyplot(fig_t)
            
            # Final Metrics Display
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="Estimated Final Gross at Opening Night", value=f"£{projected_final_gross:,.2f}")
            with metric_col2:
                st.metric(label="Estimated Final Tickets Sold at Opening Night", value=f"{int(projected_final_tix):,}")
else:
    st.info("Please ensure your data table above has valid rows to begin.")
