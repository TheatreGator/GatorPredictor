import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
import io

st.set_page_config(page_title="Theatrical Full-Run Pacing Forecaster", layout="wide")

st.title("🎭 Theatrical Show Sales & Full-Run Pacing Forecaster")
st.markdown("Forecast cumulative revenue and ticket volume from pre-sale through to the end of the post-opening run.")

# --- Sidebar Configuration ---
st.sidebar.header("Show Configuration")
show_name = st.sidebar.text_input("Show Name", "The Pantomime Adventures of Peter Pan")
max_gross_gp = st.sidebar.number_input("Max Gross GP (Revenue Capacity Limit)", min_value=1000.0, max_value=10000000.0, value=2500000.0, step=1000.0)
max_ticket_capacity = st.sidebar.number_input("Max Ticket Capacity (Total Available Tickets)", min_value=100, max_value=1000000, value=75000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Run Timeline Settings")
post_opening_weeks = st.sidebar.number_input(
    "Post-Opening Run Length (Weeks)", 
    min_value=0, 
    max_value=26, 
    value=6, 
    step=1,
    help="Number of weeks the show continues running and selling after opening night."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Dual Pacing Model Source")
uploaded_model_file = st.sidebar.file_uploader("Upload Saved Dual Pacing Model (.json)", type=["json"])

custom_rev_pacing_df = None
custom_tix_pacing_df = None

if uploaded_model_file is not None:
    try:
        model_data = json.load(uploaded_model_file)
        if "revenue_model" in model_data and "ticket_model" in model_data:
            custom_rev_pacing_df = pd.DataFrame(model_data["revenue_model"])
            custom_tix_pacing_df = pd.DataFrame(model_data["ticket_model"])
            st.sidebar.success("Loaded custom full-run pacing models successfully!")
        else:
            st.sidebar.error("Uploaded JSON does not contain valid dual model keys.")
    except Exception as e:
        st.sidebar.error(f"Error loading model file: {e}")

# Helper function to dynamically stretch/normalize pacing curves across the entire timeline when run length changes
def stretch_pacing_curve(pacing_df, new_post_opening_weeks):
    if pacing_df is None:
        return None
    orig_max_week = int(pacing_df['Weeks Out'].max())
    min_week = int(pacing_df['Weeks Out'].min())
    if orig_max_week == new_post_opening_weeks:
        return pacing_df
        
    old_weeks = pacing_df['Weeks Out'].values
    old_pcts = pacing_df['Pacing Pct'].values
    
    orig_span = orig_max_week - min_week
    old_progress = (old_weeks - min_week) / max(orig_span, 1)
    
    new_weeks = np.arange(min_week, new_post_opening_weeks + 1)
    new_span = new_post_opening_weeks - min_week
    new_progress = (new_weeks - min_week) / max(new_span, 1)
    
    new_pcts = np.interp(new_progress, old_progress, old_pcts, left=0.0, right=1.0)
    new_pcts[-1] = 1.0 
    
    return pd.DataFrame({"Weeks Out": new_weeks, "Pacing Pct": new_pcts})

# Main Navigation Tabs
tab1, tab2 = st.tabs(["📈 Active Show Forecast", "🛠️ Build Custom Full-Run Pacing Model"])

# ==========================================
# TAB 1: ACTIVE SHOW FORECAST
# ==========================================
with tab1:
    st.subheader("📊 Weekly Sales & Tickets Input Table")
    st.markdown("Tip: Select cells and **paste directly from Excel** (`Ctrl+V` / `Cmd+V`). Weeks can be negative (pre-sale) and positive (during the run after opening at week 0).")

    if "sales_data" not in st.session_state:
        st.session_state.sales_data = pd.DataFrame([
            {"Weeks Out": -36, "Sales Value": 203970.0, "Cumulative Value": 203970.0, "Tickets Sold": 4783, "Cumulative Tickets": 4783},
            {"Weeks Out": -12, "Sales Value": 150000.0, "Cumulative Value": 1200000.0, "Tickets Sold": 3000, "Cumulative Tickets": 35000},
            {"Weeks Out": 0, "Sales Value": 80000.0, "Cumulative Value": 1850000.0, "Tickets Sold": 1500, "Cumulative Tickets": 58000},
            {"Weeks Out": 2, "Sales Value": 30000.0, "Cumulative Value": 2100000.0, "Tickets Sold": 600, "Cumulative Tickets": 67000},
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
            min_w = int(clean_df['Weeks Out'].min())
            max_w = max(int(clean_df['Weeks Out'].max()), int(post_opening_weeks))
            timeline_weeks = list(range(min_w, max_w + 1))

            # Default fallback curves spanning from min_w to max_w
            default_rev_pcts = [1 / (1 + np.exp(-0.15 * (w - (max_w / 2)))) for w in timeline_weeks]
            max_rev_val = default_rev_pcts[-1] if default_rev_pcts[-1] > 0 else 1.0
            default_rev_curve = [p / max_rev_val for p in default_rev_pcts]

            default_tix_pcts = [1 / (1 + np.exp(-0.14 * (w - (max_w / 2)))) for w in timeline_weeks]
            max_tix_val = default_tix_pcts[-1] if default_tix_pcts[-1] > 0 else 1.0
            default_tix_curve = [p / max_tix_val for p in default_tix_pcts]

            if custom_rev_pacing_df is not None:
                rev_pacing_df = stretch_pacing_curve(custom_rev_pacing_df, int(post_opening_weeks))
            else:
                rev_pacing_df = pd.DataFrame({"Weeks Out": timeline_weeks, "Pacing Pct": default_rev_curve})

            if custom_tix_pacing_df is not None:
                tix_pacing_df = stretch_pacing_curve(custom_tix_pacing_df, int(post_opening_weeks))
            else:
                tix_pacing_df = pd.DataFrame({"Weeks Out": timeline_weeks, "Pacing Pct": default_tix_curve})

            if st.button("Run Full-Run Pacing Forecast", type="primary"):
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
                
                # --- Extract Milestone Readouts (Opening Week 0 vs Closing Week) ---
                op_rev_row = full_timeline_rev[full_timeline_rev['Weeks Out'] == 0]
                opening_gross = float(op_rev_row['Projected Gross'].values[0]) if not op_rev_row.empty else 0.0
                closing_gross = projected_final_gross

                op_tix_row = full_timeline_tix[full_timeline_tix['Weeks Out'] == 0]
                opening_tix = float(op_tix_row['Projected Tickets'].values[0]) if not op_tix_row.empty else 0.0
                closing_tix = projected_final_tix

                # --- Plots ---
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.subheader("Full-Run Revenue Projection")
                    fig_r, ax_r = plt.subplots(figsize=(8, 4))
                    ax_r.plot(full_timeline_rev['Weeks Out'], full_timeline_rev['Projected Gross'], color='royalblue', linewidth=2, label="Revenue Pacing Curve")
                    ax_r.scatter(clean_df['Weeks Out'], clean_df['Cumulative Value'], color='darkblue', zorder=5, label="Actual Revenue")
                    ax_r.axvline(0, color='red', linestyle=':', label='Opening Night (Week 0)')
                    ax_r.set_title("Cumulative Revenue Through Closing Week")
                    ax_r.set_xlabel("Weeks Out (0 = Opening, Positive = Run)")
                    ax_r.set_ylabel("Gross (£)")
                    ax_r.grid(True, linestyle='--', alpha=0.6)
                    ax_r.legend()
                    st.pyplot(fig_r)
                    
                with col_f2:
                    st.subheader("Full-Run Ticket Projection")
                    fig_t, ax_t = plt.subplots(figsize=(8, 4))
                    ax_t.plot(full_timeline_tix['Weeks Out'], full_timeline_tix['Projected Tickets'], color='seagreen', linewidth=2, label="Ticket Pacing Curve")
                    ax_t.scatter(clean_df['Weeks Out'], clean_df['Cumulative Tickets'], color='darkgreen', zorder=5, label="Actual Tickets")
                    ax_t.axvline(0, color='red', linestyle=':', label='Opening Night (Week 0)')
                    ax_t.set_title("Cumulative Tickets Through Closing Week")
                    ax_t.set_xlabel("Weeks Out (0 = Opening, Positive = Run)")
                    ax_t.set_ylabel("Tickets Sold")
                    ax_t.grid(True, linestyle='--', alpha=0.6)
                    ax_t.legend()
                    st.pyplot(fig_t)
                
                # --- Milestone Readout Section ---
                st.markdown("---")
                st.subheader("🎯 Opening vs. Closing Milestone Readout")
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown("#### 💷 Revenue (Gross GP)")
                    st.metric(label="At Opening Night (Week 0)", value=f"£{opening_gross:,.2f}")
                    st.metric(label=f"At Season Close (Week +{int(post_opening_weeks)})", value=f"£{closing_gross:,.2f}")
                    rev_post_sales = closing_gross - opening_gross
                    st.caption(f"Projected post-opening walk-up/run sales: £{rev_post_sales:,.2f}")

                with m_col2:
                    st.markdown("#### 🎫 Ticket Volume")
                    st.metric(label="At Opening Night (Week 0)", value=f"{int(opening_tix):,}")
                    st.metric(label=f"At Season Close (Week +{int(post_opening_weeks)})", value=f"{int(closing_tix):,}")
                    tix_post_sales = closing_tix - opening_tix
                    st.caption(f"Projected post-opening tickets sold: {int(tix_post_sales):,}")

# ==========================================
# TAB 2: BUILD CUSTOM FULL-RUN PACING MODEL
# ==========================================
model_timeline_weeks = list(range(-36, int(post_opening_weeks) + 1))

with tab2:
    st.subheader("🛠️ Historical Full-Run Pacing Model Builder")
    st.markdown(f"Upload past show sales reports that include weeks up to **Week +{post_opening_weeks}** (closing week). The app normalizes each show so that its **ultimate final total at the end of the run equals 1.0 (100%)**.")

    uploaded_historical_files = st.file_uploader(
        "Upload Past Show Files (Select multiple CSV/Excel files)", 
        type=["csv", "xlsx", "xls"], 
        accept_multiple_files=True
    )

    if uploaded_historical_files:
        rev_curves = []
        tix_curves = []
        
        for file in uploaded_historical_files:
            try:
                if file.name.endswith('.csv'):
                    df_hist = pd.read_csv(file)
                else:
                    df_hist = pd.read_excel(file)
                
                df_hist.columns = [c.strip() for c in df_hist.columns]
                
                if 'Weeks Out' in df_hist.columns:
                    rev_col = 'Cumulative Value' if 'Cumulative Value' in df_hist.columns else ('Sales Value' if 'Sales Value' in df_hist.columns else None)
                    tix_col = 'Cumulative Tickets' if 'Cumulative Tickets' in df_hist.columns else ('Tickets Sold' if 'Tickets Sold' in df_hist.columns else None)
                    
                    df_hist['Weeks Out'] = pd.to_numeric(df_hist['Weeks Out'], errors='coerce')
                    df_hist = df_hist.dropna(subset=['Weeks Out']).sort_values('Weeks Out')
                    
                    # Process Revenue
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
                                interp_rev = np.interp(model_timeline_weeks, temp_rev['Weeks Out'], temp_rev['Rev_Pct'], left=0.0, right=1.0)
                                rev_curves.append(interp_rev)
                                
                    # Process Tickets
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
                                interp_tix = np.interp(model_timeline_weeks, temp_tix['Weeks Out'], temp_tix['Tix_Pct'], left=0.0, right=1.0)
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
                    rev_model_df = pd.DataFrame({"Weeks Out": model_timeline_weeks, "Pacing Pct": mean_rev})
                    dual_model_payload["revenue_model"] = rev_model_df.to_dict(orient="records")
                    
                    st.subheader("Custom Full-Run Revenue Baseline")
                    fig_rm, ax_rm = plt.subplots(figsize=(7, 4))
                    ax_rm.plot(rev_model_df['Weeks Out'], rev_model_df['Pacing Pct'], color='royalblue', linewidth=2.5, label="Revenue Pacing Curve")
                    ax_rm.axvline(0, color='red', linestyle=':', label='Opening')
                    ax_rm.set_title("Revenue % to Season Close")
                    ax_rm.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_rm)
                    
            with col_m2:
                if tix_curves:
                    mean_tix = np.mean(tix_curves, axis=0)
                    mean_tix = np.clip(mean_tix, 0.0, 1.0)
                    mean_tix[-1] = 1.0
                    tix_model_df = pd.DataFrame({"Weeks Out": model_timeline_weeks, "Pacing Pct": mean_tix})
                    dual_model_payload["ticket_model"] = tix_model_df.to_dict(orient="records")
                    
                    st.subheader("Custom Full-Run Ticket Baseline")
                    fig_tm, ax_tm = plt.subplots(figsize=(7, 4))
                    ax_tm.plot(tix_model_df['Weeks Out'], tix_model_df['Pacing Pct'], color='seagreen', linewidth=2.5, label="Ticket Pacing Curve")
                    ax_tm.axvline(0, color='red', linestyle=':', label='Opening')
                    ax_tm.set_title("Ticket % to Season Close")
                    ax_tm.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_tm)
            
            if "revenue_model" in dual_model_payload or "ticket_model" in dual_model_payload:
                model_json_str = json.dumps(dual_model_payload, indent=2)
                st.markdown("---")
                st.download_button(
                    label="📥 Download Custom Full-Run Model Package (.json)",
                    data=model_json_str,
                    file_name="venue_full_run_pacing_models.json",
                    mime="application/json",
                    type="primary"
                )
        else:
            st.error("Could not extract valid revenue or ticket columns from the uploaded files.")
