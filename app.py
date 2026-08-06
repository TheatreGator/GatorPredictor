import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
import io

st.set_page_config(page_title="Theatrical Dual Pacing Forecaster", layout="wide")

st.title("🎭 Theatrical Show Sales & Dual Pacing Forecaster")
st.markdown("Forecast revenue and ticket volumes independently using custom dual-pacing models built from your venue's historical data.")

# --- Sidebar Configuration ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
max_gross_gp = st.sidebar.number_input("Max Gross GP (Revenue Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=2500000.0, step=1000.0)
max_ticket_capacity = st.sidebar.number_input("Max Ticket Capacity (Total Available Tickets)", min_value=100, max_value=1000000, value=75000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Dual Pacing Model Source")

# Option to upload a saved dual pacing model json file
uploaded_model_file = st.sidebar.file_uploader("Upload Saved Dual Pacing Model (.json)", type=["json"])

custom_rev_pacing_df = None
custom_tix_pacing_df = None

if uploaded_model_file is not None:
    try:
        model_data = json.load(uploaded_model_file)
        if "revenue_model" in model_data and "ticket_model" in model_data:
            custom_rev_pacing_df = pd.DataFrame(model_data["revenue_model"])
            custom_tix_pacing_df = pd.DataFrame(model_data["ticket_model"])
            st.sidebar.success("Loaded custom dual pacing models successfully!")
        else:
            st.sidebar.error("Uploaded JSON does not contain valid dual model keys ('revenue_model', 'ticket_model').")
    except Exception as e:
        st.sidebar.error(f"Error loading model file: {e}")

# Main Navigation Tabs
tab1, tab2 = st.tabs(["📈 Active Show Forecast", "🛠️ Build Custom Dual Pacing Model"])

# ==========================================
# TAB 1: ACTIVE SHOW FORECAST
# ==========================================
with tab1:
    st.subheader("📊 Weekly Sales & Tickets Input Table")
    st.markdown("Tip: Select cells and **paste directly from Excel** (`Ctrl+V` / `Cmd+V`), or edit values manually below:")

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
                df_pasted["Cumulative Tickets"] = df_pasted["Tickets Sold"].cumsum()
                st.session_state.sales_data = df_pasted[["Weeks Out", "Sales Value", "Cumulative Value", "Tickets Sold", "Cumulative Tickets"]]
                st.success("Successfully loaded 4-column pasted data and calculated cumulative metrics!")
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
            # Set up default revenue and ticket pacing curves if no custom upload is active
            default_weeks = list(range(-36, 1))
            default_rev_pcts = [1 / (1 + np.exp(-0.15 * (w + 12))) for w in default_weeks]
            max_rev_val = default_rev_pcts[-1]
            default_rev_curve = [p / max_rev_val for p in default_rev_pcts]

            default_tix_pcts = [1 / (1 + np.exp(-0.14 * (w + 13))) for w in default_weeks] # slight variation for tickets if needed
            max_tix_val = default_tix_pcts[-1]
            default_tix_curve = [p / max_tix_val for p in default_tix_pcts]

            if custom_rev_pacing_df is not None:
                rev_pacing_df = custom_rev_pacing_df
                st.sidebar.info("Active Revenue Model: Custom Uploaded")
            else:
                rev_pacing_df = pd.DataFrame({"Weeks Out": default_weeks, "Pacing Pct": default_rev_curve})

            if custom_tix_pacing_df is not None:
                tix_pacing_df = custom_tix_pacing_df
                st.sidebar.info("Active Ticket Model: Custom Uploaded")
            else:
                tix_pacing_df = pd.DataFrame({"Weeks Out": default_weeks, "Pacing Pct": default_tix_curve})

            if st.button("Run Dual Pacing Forecast", type="primary"):
                latest_row = clean_df.iloc[-1]
                latest_week = int(latest_row['Weeks Out'])
                latest_cum_gross = latest_row['Cumulative Value']
                latest_cum_tix = latest_row['Cumulative Tickets']
                
                # --- Revenue Projection Match ---
                match_rev = rev_pacing_df[rev_pacing_df['Weeks Out'] == latest_week]
                if not match_rev.empty:
                    exp_rev_pct = float(match_rev['Pacing Pct'].values[0])
                else:
                    exp_rev_pct = float(np.interp(latest_week, rev_pacing_df['Weeks Out'], rev_pacing_df['Pacing Pct']))
                exp_rev_pct = max(exp_rev_pct, 0.01)
                
                projected_final_gross = latest_cum_gross / exp_rev_pct
                projected_final_gross = min(projected_final_gross, max_gross_gp)
                
                full_timeline_rev = rev_pacing_df.copy()
                full_timeline_rev['Projected Gross'] = full_timeline_rev['Pacing Pct'] * projected_final_gross

                # --- Ticket Projection Match ---
                match_tix = tix_pacing_df[tix_pacing_df['Weeks Out'] == latest_week]
                if not match_tix.empty:
                    exp_tix_pct = float(match_tix['Pacing Pct'].values[0])
                else:
                    exp_tix_pct = float(np.interp(latest_week, tix_pacing_df['Weeks Out'], tix_pacing_df['Pacing Pct']))
                exp_tix_pct = max(exp_tix_pct, 0.01)
                
                projected_final_tix = latest_cum_tix / exp_tix_pct
                projected_final_tix = min(projected_final_tix, max_ticket_capacity)
                
                full_timeline_tix = tix_pacing_df.copy()
                full_timeline_tix['Projected Tickets'] = full_timeline_tix['Pacing Pct'] * projected_final_tix
                
                # --- Plots ---
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.subheader("Revenue Pacing Projection")
                    fig_r, ax_r = plt.subplots(figsize=(8, 4))
                    ax_r.plot(full_timeline_rev['Weeks Out'], full_timeline_rev['Projected Gross'], color='royalblue', linewidth=2, label="Revenue Pacing Curve")
                    ax_r.scatter(clean_df['Weeks Out'], clean_df['Cumulative Value'], color='darkblue', zorder=5, label="Actual Revenue")
                    ax_r.set_title("Cumulative Revenue vs. Pacing Model")
                    ax_r.set_xlabel("Weeks Out")
                    ax_r.set_ylabel("Gross (£)")
                    ax_r.grid(True, linestyle='--', alpha=0.6)
                    ax_r.legend()
                    st.pyplot(fig_r)
                    
                with col_f2:
                    st.subheader("Ticket Pacing Projection")
                    fig_t, ax_t = plt.subplots(figsize=(8, 4))
                    ax_t.plot(full_timeline_tix['Weeks Out'], full_timeline_tix['Projected Tickets'], color='seagreen', linewidth=2, label="Ticket Pacing Curve")
                    ax_t.scatter(clean_df['Weeks Out'], clean_df['Cumulative Tickets'], color='darkgreen', zorder=5, label="Actual Tickets")
                    ax_t.set_title("Cumulative Tickets vs. Pacing Model")
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
# TAB 2: BUILD CUSTOM DUAL PACING MODEL
# ==========================================
with tab2:
    st.subheader("🛠️ Historical Dual Pacing Model Builder")
    st.markdown("Upload multiple past show sales reports (CSV or Excel) containing columns: `Weeks Out`, `Cumulative Value` (or `Sales Value`), and `Cumulative Tickets` (or `Tickets Sold`). The app will synthesize them into independent revenue and ticket pacing models.")

    uploaded_historical_files = st.file_uploader(
        "Upload Past Show Files (Select multiple CSV/Excel files)", 
        type=["csv", "xlsx", "xls"], 
        accept_multiple_files=True
    )

    if uploaded_historical_files:
        rev_curves = []
        tix_curves = []
        standard_weeks = np.arange(-36, 1)
        
        for file in uploaded_historical_files:
            try:
                if file.name.endswith('.csv'):
                    df_hist = pd.read_csv(file)
                else:
                    df_hist = pd.read_excel(file)
                
                df_hist.columns = [c.strip() for c in df_hist.columns]
                
                if 'Weeks Out' in df_hist.columns:
                    # Parse Revenue
                    rev_col = 'Cumulative Value' if 'Cumulative Value' in df_hist.columns else ('Sales Value' if 'Sales Value' in df_hist.columns else None)
                    # Parse Tickets
                    tix_col = 'Cumulative Tickets' if 'Cumulative Tickets' in df_hist.columns else ('Tickets Sold' if 'Tickets Sold' in df_hist.columns else None)
                    
                    df_hist['Weeks Out'] = pd.to_numeric(df_hist['Weeks Out'], errors='coerce')
                    df_hist = df_hist.dropna(subset=['Weeks Out']).sort_values('Weeks Out')
                    
                    # Process Revenue Curve
                    if rev_col:
                        df_hist[rev_col] = pd.to_numeric(df_hist[rev_col], errors='coerce')
                        temp_rev = df_hist.dropna(subset=['Weeks Out', rev_col])
                        if rev_col == 'Sales Value':
                            temp_rev['Cumulative Value'] = temp_rev['Sales Value'].cumsum()
                            rev_target = 'Cumulative Value'
                        else:
                            rev_target = rev_col
                            
                        if not temp_rev.empty:
                            final_rev = temp_rev[rev_target].iloc[-1]
                            if final_rev > 0:
                                temp_rev['Rev_Pct'] = temp_rev[rev_target] / final_rev
                                interp_rev = np.interp(standard_weeks, temp_rev['Weeks Out'], temp_rev['Rev_Pct'], left=0.0, right=1.0)
                                rev_curves.append(interp_rev)
                                
                    # Process Ticket Curve
                    if tix_col:
                        df_hist[tix_col] = pd.to_numeric(df_hist[tix_col], errors='coerce')
                        temp_tix = df_hist.dropna(subset=['Weeks Out', tix_col])
                        if tix_col == 'Tickets Sold':
                            temp_tix['Cumulative Tickets'] = temp_tix['Tickets Sold'].cumsum()
                            tix_target = 'Cumulative Tickets'
                        else:
                            tix_target = tix_col
                            
                        if not temp_tix.empty:
                            final_tix = temp_tix[tix_target].iloc[-1]
                            if final_tix > 0:
                                temp_tix['Tix_Pct'] = temp_tix[tix_target] / final_tix
                                interp_tix = np.interp(standard_weeks, temp_tix['Weeks Out'], temp_tix['Tix_Pct'], left=0.0, right=1.0)
                                tix_curves.append(interp_tix)
            except Exception as e:
                st.warning(f"Could not parse file {file.name}: {e}")
                
        if rev_curves or tix_curves:
            st.success(f"Successfully processed historical data from **{len(uploaded_historical_files)}** files!")
            
            dual_model_payload = {}
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                if rev_curves:
                    mean_rev = np.mean(rev_curves, axis=0)
                    mean_rev = np.clip(mean_rev, 0.0, 1.0)
                    mean_rev[-1] = 1.0
                    rev_model_df = pd.DataFrame({"Weeks Out": standard_weeks, "Pacing Pct": mean_rev})
                    dual_model_payload["revenue_model"] = rev_model_df.to_dict(orient="records")
                    
                    st.subheader("Custom Revenue Pacing Baseline")
                    fig_rm, ax_rm = plt.subplots(figsize=(7, 4))
                    ax_rm.plot(rev_model_df['Weeks Out'], rev_model_df['Pacing Pct'], color='royalblue', linewidth=2.5, label="Revenue Pacing Curve")
                    ax_rm.set_title("Revenue % to Opening")
                    ax_rm.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_rm)
                    
            with col_m2:
                if tix_curves:
                    mean_tix = np.mean(tix_curves, axis=0)
                    mean_tix = np.clip(mean_tix, 0.0, 1.0)
                    mean_tix[-1] = 1.0
                    tix_model_df = pd.DataFrame({"Weeks Out": standard_weeks, "Pacing Pct": mean_tix})
                    dual_model_payload["ticket_model"] = tix_model_df.to_dict(orient="records")
                    
                    st.subheader("Custom Ticket Pacing Baseline")
                    fig_tm, ax_tm = plt.subplots(figsize=(7, 4))
                    ax_tm.plot(tix_model_df['Weeks Out'], tix_model_df['Pacing Pct'], color='seagreen', linewidth=2.5, label="Ticket Pacing Curve")
                    ax_tm.set_title("Ticket % to Opening")
                    ax_tm.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_tm)
            
            if "revenue_model" in dual_model_payload or "ticket_model" in dual_model_payload:
                model_json_str = json.dumps(dual_model_payload, indent=2)
                st.markdown("---")
                st.download_button(
                    label="📥 Download Custom Dual Pacing Model Package (.json)",
                    data=model_json_str,
                    file_name="venue_dual_pacing_models.json",
                    mime="application/json",
                    type="primary"
                )
        else:
            st.error("Could not extract valid revenue or ticket columns from the uploaded files. Please check column headers.")
