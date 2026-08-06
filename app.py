import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
import io

st.set_page_config(page_title="Theatrical Pacing Forecaster & Model Builder", layout="wide")

st.title("🎭 Theatrical Show Sales & Pacing Forecaster")
st.markdown("Forecast sales using historical pacing curves or build/upload custom models generated from your past venue data.")

# --- Sidebar Configuration ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
max_gross_gp = st.sidebar.number_input("Max Gross GP (Revenue Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=2500000.0, step=1000.0)
max_ticket_capacity = st.sidebar.number_input("Max Ticket Capacity (Total Available Tickets)", min_value=100, max_value=1000000, value=75000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Pacing Model Source")

# Option to upload a saved custom pacing model json file
uploaded_model_file = st.sidebar.file_uploader("Upload Saved Custom Pacing Model (.json)", type=["json"])

custom_pacing_df = None
if uploaded_model_file is not None:
    try:
        model_data = json.load(uploaded_model_file)
        custom_pacing_df = pd.DataFrame(model_data)
        st.sidebar.success("Loaded custom pacing model successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model file: {e}")

# Main Navigation Tabs
tab1, tab2 = st.tabs(["📈 Active Show Forecast", "🛠️ Build Custom Pacing Model from History"])

# ==========================================
# TAB 1: ACTIVE SHOW FORECAST
# ==========================================
with tab1:
    st.subheader("📊 Weekly Sales & Tickets Input Table")
    st.markdown("Tip: You can select cells and **paste directly from Excel** (`Ctrl+V` / `Cmd+V`), or add/edit rows manually.")

    # Initialize session state for sales data
    if "sales_data" not in st.session_state:
        st.session_state.sales_data = pd.DataFrame([
            {"Weeks Out": -36, "Sales Value": 203970.0, "Cumulative Value": 203970.0, "Tickets Sold": 4783, "Cumulative Tickets": 4783},
            {"Weeks Out": -35, "Sales Value": 294872.5, "Cumulative Value": 498842.5, "Tickets Sold": 6948, "Cumulative Tickets": 11731},
            {"Weeks Out": -34, "Sales Value": 39437.0, "Cumulative Value": 538279.5, "Tickets Sold": 964, "Cumulative Tickets": 12695},
            {"Weeks Out": -33, "Sales Value": 22447.0, "Cumulative Value": 560726.5, "Tickets Sold": 570, "Cumulative Tickets": 13265},
            {"Weeks Out": -32, "Sales Value": 16256.0, "Cumulative Value": 576982.5, "Tickets Sold": 392, "Cumulative Tickets": 13657},
        ])

    pasted_input = st.text_area("Quick Paste Excel Data here (Optional):", height=80, placeholder="Paste copied rows from Excel here...")

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
            # Determine pacing curve to use
            if custom_pacing_df is not None:
                pacing_curve_df = custom_pacing_df
                st.info("Using **Custom Uploaded Pacing Model** for projections.")
            else:
                # Default built-in theatrical curve
                default_weeks = list(range(-36, 1))
                default_pacing_pcts = [1 / (1 + np.exp(-0.15 * (w + 12))) for w in default_weeks]
                max_val = default_pacing_pcts[-1]
                pacing_curve_df = pd.DataFrame({
                    "Weeks Out": default_weeks,
                    "Pacing Pct": [p / max_val for p in default_pacing_pcts]
                })
                st.info("Using **Default Built-in Pacing Model** (Tip: Build or upload a custom model in the next tab for higher accuracy).")

            if st.button("Run Pacing Forecast", type="primary"):
                latest_row = clean_df.iloc[-1]
                latest_week = int(latest_row['Weeks Out'])
                latest_cum_gross = latest_row['Cumulative Value']
                latest_cum_tix = latest_row['Cumulative Tickets']
                
                # Match week in pacing curve
                match_row = pacing_curve_df[pacing_curve_df['Weeks Out'] == latest_week]
                if not match_row.empty:
                    expected_pct = float(match_row['Pacing Pct'].values[0])
                else:
                    expected_pct = float(np.interp(latest_week, pacing_curve_df['Weeks Out'], pacing_curve_df['Pacing Pct']))
                
                expected_pct = max(expected_pct, 0.01)
                
                projected_final_gross = latest_cum_gross / expected_pct
                projected_final_tix = latest_cum_tix / expected_pct
                
                projected_final_gross = min(projected_final_gross, max_gross_gp)
                projected_final_tix = min(projected_final_tix, max_ticket_capacity)
                
                full_timeline = pacing_curve_df.copy()
                full_timeline['Projected Gross'] = full_timeline['Pacing Pct'] * projected_final_gross
                full_timeline['Projected Tickets'] = full_timeline['Pacing Pct'] * projected_final_tix
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.subheader("Revenue Pacing Projection")
                    fig_r, ax_r = plt.subplots(figsize=(8, 4))
                    ax_r.plot(full_timeline['Weeks Out'], full_timeline['Projected Gross'], color='royalblue', linewidth=2, label="Pacing Forecast Curve")
                    ax_r.scatter(clean_df['Weeks Out'], clean_df['Cumulative Value'], color='darkblue', zorder=5, label="Actual Sales")
                    ax_r.set_title("Cumulative Revenue vs. Pacing Curve")
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
                    ax_t.set_title("Cumulative Tickets vs. Pacing Curve")
                    ax_t.set_xlabel("Weeks Out")
                    ax_t.set_ylabel("Tickets Sold")
                    ax_t.grid(True, linestyle='--', alpha=0.6)
                    ax_t.legend()
                    st.pyplot(fig_t)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Estimated Final Gross at Opening Night", value=f"£{projected_final_gross:,.2f}")
                with metric_col2:
                    st.metric(label="Estimated Final Tickets Sold at Opening Night", value=f"{int(projected_final_tix):,}")

# ==========================================
# TAB 2: BUILD CUSTOM PACING MODEL FROM HISTORY
# ==========================================
with tab2:
    st.subheader("🛠️ Historical Pacing Model Builder")
    st.markdown("Upload multiple past show sales reports (CSV or Excel) containing columns: `Weeks Out`, `Cumulative Value` (or `Sales Value`). The app will synthesize them into a custom master pacing curve.")

    uploaded_historical_files = st.file_uploader(
        "Upload Past Show Files (Select multiple CSV/Excel files)", 
        type=["csv", "xlsx", "xls"], 
        accept_multiple_files=True
    )

    if uploaded_historical_files:
        historical_curves = []
        
        for file in uploaded_historical_files:
            try:
                if file.name.endswith('.csv'):
                    df_hist = pd.read_csv(file)
                else:
                    df_hist = pd.read_excel(file)
                
                # Normalize column name checks
                df_hist.columns = [c.strip() for c in df_hist.columns]
                
                # Check required columns
                if 'Weeks Out' in df_hist.columns:
                    val_col = 'Cumulative Value' if 'Cumulative Value' in df_hist.columns else ('Sales Value' if 'Sales Value' in df_hist.columns else None)
                    if val_col:
                        df_hist['Weeks Out'] = pd.to_numeric(df_hist['Weeks Out'], errors='coerce')
                        df_hist[val_col] = pd.to_numeric(df_hist[val_col], errors='coerce')
                        df_hist = df_hist.dropna(subset=['Weeks Out', val_col]).sort_values('Weeks Out')
                        
                        if val_col == 'Sales Value':
                            df_hist['Cumulative Value'] = df_hist['Sales Value'].cumsum()
                            val_col = 'Cumulative Value'
                            
                        final_val = df_hist[val_col].iloc[-1]
                        if final_val > 0:
                            df_hist['Pacing_Pct'] = df_hist[val_col] / final_val
                            standard_weeks = np.arange(-36, 1)
                            interp_pcts = np.interp(standard_weeks, df_hist['Weeks Out'], df_hist['Pacing_Pct'], left=0.0, right=1.0)
                            historical_curves.append(interp_pcts)
            except Exception as e:
                st.warning(f"Could not parse file {file.name}: {e}")
                
        if historical_curves:
            mean_curve = np.mean(historical_curves, axis=0)
            mean_curve = np.clip(mean_curve, 0.0, 1.0)
            mean_curve[-1] = 1.0
            
            custom_model_df = pd.DataFrame({
                "Weeks Out": np.arange(-36, 1),
                "Pacing Pct": mean_curve
            })
            
            st.success(f"Successfully processed **{len(historical_curves)}** historical shows and generated a custom pacing model!")
            
            # Plot custom curve
            fig_m, ax_m = plt.subplots(figsize=(10, 4))
            ax_m.plot(custom_model_df['Weeks Out'], custom_model_df['Pacing Pct'], color='purple', linewidth=2.5, label="Custom Venue Baseline Curve")
            ax_m.set_title("Generated Custom Pacing Curve (% to Opening)")
            ax_m.set_xlabel("Weeks Out")
            ax_m.set_ylabel("Pacing Percentage (0.0 to 1.0)")
            ax_m.grid(True, linestyle='--', alpha=0.6)
            ax_m.legend()
            st.pyplot(fig_m)
            
            # Download button for model JSON
            model_json_str = custom_model_df.to_json(orient='records')
            st.download_button(
                label="📥 Download Custom Pacing Model (.json)",
                data=model_json_str,
                file_name="venue_custom_pacing_model.json",
                mime="application/json",
                type="primary"
            )
        else:
            st.error("None of the uploaded files could be successfully parsed. Please ensure they contain 'Weeks Out' and sales columns.")
