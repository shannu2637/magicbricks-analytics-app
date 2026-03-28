import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
 
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MagicBricks Explorer",
    page_icon="🏠",
    layout="wide",
)
 
# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e9ecef;
    }
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSelectbox"] label {
        font-weight: 500;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ── Data loading & cleaning (mirrors your notebook) ───────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("MagicBrick_Data.csv")
 
    # Drop unnamed index col if present
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
 
    # ── Price column ──────────────────────────────────────────────────────────
    for i in range(len(df["Price"])):
        val = str(df.at[i, "Price"]).strip()
        if "₹" in val:
            df.at[i, "Price"] = val.replace("₹", "").strip()
        else:
            df.at[i, "Price"] = np.nan
 
    for i in range(len(df["Price"])):
        val = str(df.at[i, "Price"]).strip()
        if "Cr" in val:
            df.at[i, "Price"] = float(val.replace("Cr", "").strip()) * 10_000_000
        elif "Lac" in val:
            df.at[i, "Price"] = float(val.replace("Lac", "").strip()) * 100_000
        else:
            df.at[i, "Price"] = np.nan
 
    df["Price"] = df["Price"].astype("float64")
 
    # ── EMI column ────────────────────────────────────────────────────────────
    for i in range(len(df["EMI"])):
        val = str(df.at[i, "EMI"]).strip()
        if "k" in val:
            df.at[i, "EMI"] = float(val.replace("k", "").strip()) * 1_000
        elif "L" in val:
            df.at[i, "EMI"] = float(val.replace("L", "").strip()) * 100_000
        else:
            df.at[i, "EMI"] = np.nan
 
    df["EMI"] = df["EMI"].astype("float64")
 
    # ── Carpet Area column ────────────────────────────────────────────────────
    for i in range(len(df["Carpet Area"])):
        val = str(df.at[i, "Carpet Area"]).lower().replace(",", "").strip()
        if val == "nan":
            continue
        elif val.replace(".", "").isdigit():
            df.at[i, "Carpet Area"] = float(val)
        elif "sqm" in val:
            df.at[i, "Carpet Area"] = float(val.replace("sqm", "").strip()) * 10.7639
        elif "sqyrd" in val:
            df.at[i, "Carpet Area"] = float(val.replace("sqyrd", "").strip()) * 9
        else:
            df.at[i, "Carpet Area"] = np.nan
 
    df["Carpet Area"] = df["Carpet Area"].astype("float64")
 
    # ── Drop nulls & duplicates ───────────────────────────────────────────────
    obj_cols = df.select_dtypes(include="object").columns
    df = df.dropna(subset=obj_cols)
    df.drop_duplicates(inplace=True)
    rooms_count = ["BHK", "Bathrooms", "Balconies"]
    df = df.dropna(subset=rooms_count)
 
    # ── Fill Price & EMI by locality median ───────────────────────────────────
    df["Price"] = df["Price"].fillna(df.groupby("Locality")["Price"].transform("median"))
    df["EMI"]   = df["EMI"].fillna(df.groupby("Locality")["EMI"].transform("median"))
 
    # ── Derived features ──────────────────────────────────────────────────────
    df["Price_per_sqft"]  = df["Price"] / df["Carpet Area"]
    df["Carpet_to_bhk"]   = df["Carpet Area"] / df["BHK"]
    df["Carpet_to_bath"]  = df["Carpet Area"] / df["Bathrooms"]
 
    # ── ROI score per locality (same as notebook cell 95) ─────────────────────
    locality_stats = (
        df.groupby(["City", "Locality"])
        .agg(Price_per_sqft=("Price_per_sqft", "median"),
             EMI=("EMI", "median"),
             Count=("Price", "count"))
        .reset_index()
    )
    locality_stats["ROI_score"] = locality_stats["Count"] / locality_stats["Price_per_sqft"]
 
    df = df.merge(locality_stats[["City", "Locality", "ROI_score"]], on=["City", "Locality"], how="left")
 
    return df
 
 
# ── Helper: format price ───────────────────────────────────────────────────────
def fmt_price(val):
    if pd.isna(val):
        return "N/A"
    if val >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    return f"₹{val/1e5:.1f} L"
 
 
# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.warning(
        "⚠️ `MagicBrick_Data.csv` not found in the same folder as app.py. "
        "Place the CSV next to this file and refresh.",
        icon="⚠️",
    )
    data_loaded = False
    st.stop()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏠 MagicBricks Property Explorer")
st.caption("Filter properties by City → Locality → Developer. All dropdowns stay in sync.")
st.divider()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CASCADING FILTERS (sidebar)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔍 Filters")
    st.caption("Selecting a city narrows localities and developers automatically.")
 
    # ── City ──────────────────────────────────────────────────────────────────
    cities = ["All"] + sorted(df["City"].unique())
    city = st.selectbox("🏙️ City", cities)
 
    city_df = df if city == "All" else df[df["City"] == city]
 
    # ── Locality (only those in selected city) ────────────────────────────────
    localities = ["All"] + sorted(city_df["Locality"].unique())
    locality = st.selectbox("📍 Locality", localities)
 
    loc_df = city_df if locality == "All" else city_df[city_df["Locality"] == locality]
 
    # ── Developer (only those in selected city + locality) ────────────────────
    developers = ["All"] + sorted(loc_df["Developer"].unique())
    developer = st.selectbox("🏗️ Developer", developers)
 
    st.divider()
 
    # ── BHK ───────────────────────────────────────────────────────────────────
    bhk_options = sorted(df["BHK"].dropna().unique())
    bhk_filter = st.multiselect("🛏️ BHK", bhk_options, default=bhk_options)
 
    # ── Price range ───────────────────────────────────────────────────────────
    price_min = float(df["Price"].min())
    price_max = float(df["Price"].max())
    price_range = st.slider(
        "💰 Price range (₹ Cr)",
        min_value=round(price_min / 1e7, 2),
        max_value=round(price_max / 1e7, 2),
        value=(round(price_min / 1e7, 2), round(price_max / 1e7, 2)),
        step=0.1,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# APPLY ALL FILTERS
# ══════════════════════════════════════════════════════════════════════════════
filtered = df.copy()
if city      != "All": filtered = filtered[filtered["City"]      == city]
if locality  != "All": filtered = filtered[filtered["Locality"]  == locality]
if developer != "All": filtered = filtered[filtered["Developer"] == developer]
filtered = filtered[filtered["BHK"].isin(bhk_filter)]
filtered = filtered[
    (filtered["Price"] >= price_range[0] * 1e7) &
    (filtered["Price"] <= price_range[1] * 1e7)
]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total listings",   f"{len(filtered):,}")
k2.metric("Median price",     fmt_price(filtered["Price"].median()))
k3.metric("Median ₹/sqft",    f"₹{filtered['Price_per_sqft'].median():,.0f}" if len(filtered) else "N/A")
k4.metric("Median carpet",    f"{filtered['Carpet Area'].median():,.0f} sqft" if len(filtered) else "N/A")
k5.metric("Cities covered",   filtered["City"].nunique())
 
st.divider()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Listings", "📊 Analytics", "💰 ROI Finder", "🏗️ Developer Index", "🔮 Price Predictor"])
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – LISTINGS TABLE
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if filtered.empty:
        st.info("No properties match the current filters.")
    else:
        display_cols = ["City", "Locality", "Developer", "Project", "BHK",
                        "Bathrooms", "Balconies", "Carpet Area", "Price", "Price_per_sqft", "EMI"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        show = filtered[display_cols].copy()
        show["Price"]          = show["Price"].apply(fmt_price)
        show["EMI"]            = show["EMI"].apply(lambda v: f"₹{v/1000:.0f}k/mo" if not pd.isna(v) else "N/A")
        show["Price_per_sqft"] = show["Price_per_sqft"].apply(lambda v: f"₹{v:,.0f}" if not pd.isna(v) else "N/A")
        show["Carpet Area"]    = show["Carpet Area"].apply(lambda v: f"{v:,.0f} sqft" if not pd.isna(v) else "N/A")
        st.dataframe(show.reset_index(drop=True), use_container_width=True, height=420)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if len(filtered) < 3:
        st.info("Need at least 3 listings for analytics. Broaden your filters.")
    else:
        col1, col2 = st.columns(2)
 
        # Price per sqft by locality
        with col1:
            st.markdown('<p class="section-header">Median ₹/sqft by locality</p>', unsafe_allow_html=True)
            loc_psf = (
                filtered.groupby("Locality")["Price_per_sqft"]
                .median().reset_index()
                .sort_values("Price_per_sqft", ascending=True)
            )
            fig = px.bar(
                loc_psf, x="Price_per_sqft", y="Locality", orientation="h",
                color="Price_per_sqft", color_continuous_scale="Blues",
                labels={"Price_per_sqft": "₹/sqft", "Locality": ""},
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=320)
            st.plotly_chart(fig, use_container_width=True)
 
        # BHK distribution
        with col2:
            st.markdown('<p class="section-header">BHK distribution</p>', unsafe_allow_html=True)
            bhk_dist = filtered["BHK"].value_counts().reset_index()
            bhk_dist.columns = ["BHK", "Count"]
            fig2 = px.pie(bhk_dist, values="Count", names="BHK",
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=320)
            st.plotly_chart(fig2, use_container_width=True)
 
        # Price vs Carpet Area scatter
        st.markdown('<p class="section-header">Price vs carpet area (coloured by BHK)</p>', unsafe_allow_html=True)
        fig3 = px.scatter(
            filtered, x="Carpet Area", y="Price", color="BHK",
            hover_data=["City", "Locality", "Developer"],
            labels={"Price": "Price (₹)", "Carpet Area": "Carpet Area (sqft)"},
            color_continuous_scale="Viridis",
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=360)
        st.plotly_chart(fig3, use_container_width=True)
 
        # City-wise price box
        if city == "All":
            st.markdown('<p class="section-header">Price distribution by city</p>', unsafe_allow_html=True)
            fig4 = px.box(
                filtered, x="City", y="Price",
                color="City",
                labels={"Price": "Price (₹)"},
            )
            fig4.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=340)
            st.plotly_chart(fig4, use_container_width=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – ROI FINDER
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.caption("ROI score = listing count ÷ median ₹/sqft — high demand, low price = high score.")
    roi = (
        filtered.groupby(["City", "Locality"])
        .agg(
            Median_price=("Price", "median"),
            Median_psf=("Price_per_sqft", "median"),
            Median_EMI=("EMI", "median"),
            Listings=("Price", "count"),
        )
        .reset_index()
    )
    roi["ROI_score"] = (roi["Listings"] / roi["Median_psf"]).round(4)
    roi = roi.sort_values("ROI_score", ascending=False)
    roi["Median_price"] = roi["Median_price"].apply(fmt_price)
    roi["Median_psf"]   = roi["Median_psf"].apply(lambda v: f"₹{v:,.0f}")
    roi["Median_EMI"]   = roi["Median_EMI"].apply(lambda v: f"₹{v/1000:.0f}k" if not pd.isna(v) else "N/A")
 
    if roi.empty:
        st.info("No data for ROI calculation with current filters.")
    else:
        min_val = 5
        max_val = max(min_val + 1, min(30, len(roi)))
        default_val = min(max_val, 10)
        top_n = st.slider("Show top N localities", min_val, max_val, default_val)
        st.dataframe(roi.head(top_n).reset_index(drop=True), use_container_width=True)
 
        fig5 = px.scatter(
            roi.head(top_n),
            x="Median_psf", y="ROI_score",
            color="City", size="Listings",
            hover_data=["Locality", "Median_price"],
            labels={"Median_psf": "Median ₹/sqft", "ROI_score": "ROI score"},
            title="ROI analysis — low price + high demand localities",
        )
        fig5.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
        st.plotly_chart(fig5, use_container_width=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – DEVELOPER INDEX
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.caption("Developer quality index: median ₹/sqft + avg carpet area signals premium vs budget positioning.")
    if len(filtered) < 3:
        st.info("Broaden your filters to see developer stats.")
    else:
        dev_pivot = (
            filtered.groupby("Developer")
            .agg(
                Median_psf=("Price_per_sqft", "median"),
                Avg_carpet=("Carpet Area", "mean"),
                Listings=("Price", "count"),
                Median_price=("Price", "median"),
            )
            .reset_index()
            .sort_values("Median_psf", ascending=False)
        )
        dev_pivot["Median_price"] = dev_pivot["Median_price"].apply(fmt_price)
        dev_pivot["Avg_carpet"]   = dev_pivot["Avg_carpet"].apply(lambda v: f"{v:,.0f} sqft")
        dev_pivot["Median_psf"]   = dev_pivot["Median_psf"].apply(lambda v: f"₹{v:,.0f}")
 
        st.dataframe(dev_pivot.reset_index(drop=True), use_container_width=True)
 
        # Bubble chart: listings vs psf
        raw_dev = (
            filtered.groupby("Developer")
            .agg(Median_psf=("Price_per_sqft", "median"),
                 Avg_carpet=("Carpet Area", "mean"),
                 Listings=("Price", "count"))
            .reset_index()
        )
        fig6 = px.scatter(
            raw_dev, x="Avg_carpet", y="Median_psf",
            size="Listings", color="Developer",
            hover_data=["Listings"],
            labels={"Avg_carpet": "Avg carpet area (sqft)", "Median_psf": "Median ₹/sqft"},
            title="Developer positioning — size = number of listings",
        )
        fig6.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=420)
        st.plotly_chart(fig6, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 – PRICE PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("🔮 Property Price & EMI Predictor")
    st.caption("Enter property details to predict the estimated price and monthly EMI.")
    
    # Load model and metadata
    try:
        model = joblib.load("model.pkl")
        with open("model_meta.json", "r") as f:
            model_meta = json.load(f)
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure model.pkl and model_meta.json exist in the folder.")
        st.stop()
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        # City selection
        available_cities = sorted(df["City"].unique())
        selected_city = st.selectbox("🏙️ City", available_cities)
        
        # Get localities for selected city
        city_localities = sorted(df[df["City"] == selected_city]["Locality"].unique())
        selected_locality = st.selectbox("📍 Locality", city_localities)
        
        # BHK
        bhk = st.number_input("🛏️ BHK", min_value=1, max_value=10, value=2, step=1)
        
        # Bathrooms
        bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2, step=1)
    
    with col2:
        # Carpet Area (in sqft)
        carpet_area = st.number_input("📐 Carpet Area (sqft)", min_value=100, max_value=50000, value=1000, step=100)
        
        # Balconies (optional)
        balconies = st.number_input("🪟 Balconies", min_value=0, max_value=5, value=1, step=1)
        
        # Get developer for selected locality
        available_devs = sorted(df[df["Locality"] == selected_locality]["Developer"].unique())
        selected_developer = st.selectbox("🏗️ Developer", available_devs)
    
    st.divider()
    
    # Make prediction
    if st.button("💡 Predict Price & EMI", use_container_width=True):
        try:
            # Calculate derived features (same as training)
            carpet_to_bhk = carpet_area / bhk
            carpet_to_bath = carpet_area / bathrooms
            log_carpet = np.log1p(carpet_area)
            
            # Prepare input dataframe with EXACT column order from training
            # Order: City, Locality, Developer (categorical), then numerical features
            input_data = pd.DataFrame({
                "City": [selected_city],
                "Locality": [selected_locality],
                "Developer": [selected_developer],
                "BHK": [float(bhk)],
                "Bathrooms": [float(bathrooms)],
                "Balconies": [float(balconies)],
                "Carpet Area": [float(carpet_area)],
                "Carpet_to_bhk": [float(carpet_to_bhk)],
                "Carpet_to_bath": [float(carpet_to_bath)],
                "log_carpet": [float(log_carpet)],
            })
            
            # Reorder columns to match training: categorical first, then numerical
            categor_cols = ["City", "Locality", "Developer"]
            num_cols = ["BHK", "Bathrooms", "Balconies", "Carpet Area", "Carpet_to_bhk", "Carpet_to_bath", "log_carpet"]
            input_data = input_data[categor_cols + num_cols]
            
            # Make prediction
            predicted_price_log = model.predict(input_data)[0]
            predicted_price = np.expm1(predicted_price_log)  # Convert from log space back to actual price
            
            # Calculate EMI (assuming 9% annual interest for 20 years)
            annual_rate = 0.09
            monthly_rate = annual_rate / 12
            num_months = 20 * 12
            if monthly_rate > 0:
                emi = (predicted_price * monthly_rate * (1 + monthly_rate)**num_months) / ((1 + monthly_rate)**num_months - 1)
            else:
                emi = predicted_price / num_months
            
            # Display results
            st.success("✅ Prediction generated successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "💰 Estimated Price",
                    fmt_price(predicted_price)
                )
                st.caption(f"₹ {predicted_price:,.0f}")
            
            with col2:
                st.metric(
                    "📊 Estimated Monthly EMI",
                    f"₹{emi/1000:.1f}k/mo"
                )
                st.caption(f"₹ {emi:,.0f}")
            
            # Summary box
            st.divider()
            st.markdown("### 📋 Property Summary")
            summary_df = pd.DataFrame({
                "Attribute": ["City", "Locality", "Developer", "BHK", "Bathrooms", "Carpet Area", "Balconies"],
                "Value": [
                    selected_city,
                    selected_locality,
                    selected_developer,
                    f"{bhk}",
                    f"{bathrooms}",
                    f"{carpet_area:,} sqft",
                    f"{balconies}"
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Price per sqft
            price_per_sqft = predicted_price / carpet_area
            st.markdown(f"**📊 Price per sqft:** ₹{price_per_sqft:,.0f}")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")