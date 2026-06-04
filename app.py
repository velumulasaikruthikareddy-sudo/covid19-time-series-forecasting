import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI-Powered COVID-19 Forecasting & Risk Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #ff4b4b;
    text-align: center;
}

h2, h3 {
    color: #31333F;
}

div[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("📊 AI-Powered COVID-19 Forecasting & Risk Analysis Dashboard")

st.markdown(
    "### Advanced Time Series Forecasting using AI Analytics"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    try:

        df = pd.read_csv(
            "dataset/owid-covid-data.csv"
        )

        return df

    except Exception as e:

        st.error(
            f"Error loading dataset: {e}"
        )

        st.stop()

df = load_data()

# =====================================================
# REQUIRED COLUMNS
# =====================================================

required_columns = [
    "location",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()

# =====================================================
# DATE CONVERSION
# =====================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Dashboard Controls")

countries = sorted(
    df["location"].dropna().unique()
)

country = st.sidebar.selectbox(
    "Select Country",
    countries
)

# =====================================================
# DATE RANGE
# =====================================================

min_date = df["date"].min()
max_date = df["date"].max()

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2020-03-01"),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("2021-12-31"),
    min_value=min_date,
    max_value=max_date
)

if start_date >= end_date:

    st.error(
        "Start Date must be earlier than End Date."
    )

    st.stop()

# =====================================================
# FORECAST DAYS
# =====================================================

forecast_days = st.sidebar.slider(
    "Forecast Days",
    7,
    30,
    14
)

show_data = st.sidebar.checkbox(
    "Show Dataset"
)

show_graphs = st.sidebar.checkbox(
    "Show Advanced Analytics",
    value=True
)

# =====================================================
# FILTER DATA
# =====================================================

country_data = df[
    df["location"] == country
].copy()

country_data = country_data[
    (country_data["date"] >= pd.to_datetime(start_date)) &
    (country_data["date"] <= pd.to_datetime(end_date))
]

country_data = country_data[[
    "date",
    "total_cases",
    "new_cases",
    "total_deaths"
]]

# =====================================================
# EMPTY CHECK
# =====================================================

if country_data.empty:

    st.error(
        "No data available for selected filters."
    )

    st.stop()

# =====================================================
# NUMERIC CONVERSION
# =====================================================

numeric_columns = [
    "total_cases",
    "new_cases",
    "total_deaths"
]

for col in numeric_columns:

    country_data[col] = pd.to_numeric(
        country_data[col],
        errors="coerce"
    )

country_data = country_data.fillna(0)

# =====================================================
# REMOVE NEGATIVES
# =====================================================

country_data["total_cases"] = (
    country_data["total_cases"]
    .clip(lower=0)
)

country_data["new_cases"] = (
    country_data["new_cases"]
    .clip(lower=0)
)

country_data["total_deaths"] = (
    country_data["total_deaths"]
    .clip(lower=0)
)

country_data = country_data.sort_values(
    "date"
)

# =====================================================
# METRICS
# =====================================================

latest_total_cases = int(
    country_data["total_cases"].iloc[-1]
)

latest_total_deaths = int(
    country_data["total_deaths"].iloc[-1]
)

latest_new_cases = int(
    country_data["new_cases"].iloc[-1]
)

# =====================================================
# MAIN METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Cases",
    f"{latest_total_cases:,}"
)

col2.metric(
    "Total Deaths",
    f"{latest_total_deaths:,}"
)

col3.metric(
    "Latest New Cases",
    f"{latest_new_cases:,}"
)

st.markdown("---")

# =====================================================
# ADVANCED KPI DASHBOARD
# =====================================================

st.subheader("📈 Advanced COVID-19 KPIs")

active_cases = max(
    latest_total_cases - latest_total_deaths,
    0
)

recovery_rate = round(
    (
        (
            latest_total_cases - latest_total_deaths
        )
        / max(latest_total_cases, 1)
    ) * 100,
    2
)

mortality_rate = round(
    (
        latest_total_deaths
        / max(latest_total_cases, 1)
    ) * 100,
    2
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Active Cases",
    f"{active_cases:,}"
)

k2.metric(
    "Recovery Rate",
    f"{recovery_rate}%"
)

k3.metric(
    "Mortality Rate",
    f"{mortality_rate}%"
)

if latest_new_cases > 100000:

    risk_level = "🔴 High Risk"

elif latest_new_cases > 10000:

    risk_level = "🟡 Medium Risk"

else:

    risk_level = "🟢 Low Risk"

k4.metric(
    "Risk Level",
    risk_level
)

st.markdown("---")

# =====================================================
# AI HEALTH RECOMMENDATIONS
# =====================================================

st.subheader("🤖 AI Health Recommendations")

if latest_new_cases > 100000:

    st.error("""
    AI Recommendation:
    - Increase testing capacity
    - Strengthen hospital preparedness
    - Promote vaccination campaigns
    - Encourage mask usage
    """)

elif latest_new_cases > 10000:

    st.warning("""
    AI Recommendation:
    - Monitor outbreaks
    - Improve healthcare allocation
    - Continue vaccination awareness
    """)

else:

    st.success("""
    AI Recommendation:
    - Maintain preventive measures
    - Continue monitoring trends
    - Encourage public awareness
    """)

st.markdown("---")

# =====================================================
# SHOW DATASET
# =====================================================

if show_data:

    st.subheader("Country Dataset")

    st.dataframe(
        country_data,
        use_container_width=True
    )

# =====================================================
# MOVING AVERAGE
# =====================================================

country_data["7_day_avg"] = (
    country_data["new_cases"]
    .rolling(window=7)
    .mean()
)

# =====================================================
# TOTAL CASES GRAPH
# =====================================================

st.subheader(
    f"COVID-19 Total Cases Trend in {country}"
)

fig1 = px.line(
    country_data,
    x="date",
    y="total_cases",
    title=f"COVID-19 Total Cases Trend - {country}"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =====================================================
# DAILY CASES GRAPH
# =====================================================

st.subheader(
    "Daily Cases with 7-Day Moving Average"
)

fig2 = px.line(
    country_data,
    x="date",
    y=["new_cases", "7_day_avg"],
    title="Daily Cases and Moving Average"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================================
# FORECASTING SECTION
# =====================================================

st.subheader(
    "COVID-19 Future Spread Prediction"
)

forecast_data = country_data[[
    "date",
    "total_cases"
]].copy()

forecast_data.columns = [
    "Date",
    "Cases"
]

forecast_data = forecast_data.dropna()

forecast_data["Day"] = np.arange(
    len(forecast_data)
)

X = forecast_data[["Day"]]
y = forecast_data["Cases"]

# =====================================================
# TRAIN MODEL
# =====================================================

model = LinearRegression()

model.fit(X, y)

# =====================================================
# FUTURE PREDICTION
# =====================================================

future_days = np.arange(
    len(forecast_data),
    len(forecast_data) + forecast_days
).reshape(-1, 1)

future_predictions = model.predict(
    future_days
)

future_predictions = np.round(
    future_predictions
).astype(int)

future_dates = pd.date_range(
    start=forecast_data["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

forecast_result = pd.DataFrame({

    "Date": future_dates,
    "Predicted Total Cases": future_predictions

})

# =====================================================
# ACTUAL + PREDICTED GRAPH
# =====================================================

actual_df = forecast_data.copy()

actual_df.columns = [
    "Date",
    "Cases",
    "Day"
]

actual_df = actual_df[[
    "Date",
    "Cases"
]]

actual_df["Type"] = "Actual Cases"

predicted_df = forecast_result.copy()

predicted_df.columns = [
    "Date",
    "Cases"
]

predicted_df["Type"] = "Predicted Cases"

combined_df = pd.concat([
    actual_df,
    predicted_df
])

forecast_fig = px.line(
    combined_df,
    x="Date",
    y="Cases",
    color="Type",
    title=f"Actual vs Predicted COVID-19 Cases - {country}"
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

# =====================================================
# FORECAST TABLE
# =====================================================

st.subheader(
    "Forecast Results"
)

st.dataframe(
    forecast_result,
    use_container_width=True
)

# =====================================================
# ACCURACY
# =====================================================

score = model.score(X, y)

accuracy = round(
    score * 100,
    2
)

st.subheader(
    "AI Forecasting Accuracy"
)

acc1, acc2 = st.columns(2)

acc1.metric(
    "Forecast Accuracy",
    f"{accuracy}%"
)

acc2.metric(
    "Forecast Days",
    f"{forecast_days} Days"
)

# =====================================================
# COUNTRY COMPARISON
# =====================================================

if show_graphs:

    st.subheader("🌍 Country Comparison Dashboard")

    comparison_countries = st.multiselect(
        "Select Countries for Comparison",
        countries,
        default=[country]
    )

    comparison_df = df[
        df["location"].isin(comparison_countries)
    ]

    comparison_df = comparison_df[[
        "date",
        "location",
        "total_cases"
    ]]

    comparison_fig = px.line(
        comparison_df,
        x="date",
        y="total_cases",
        color="location",
        title="COVID-19 Country Comparison"
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )

# =====================================================
# SMART INSIGHTS
# =====================================================

st.subheader("🧠 Smart COVID Insights")

max_cases = int(
    country_data["new_cases"].max()
)

avg_cases = int(
    country_data["new_cases"].mean()
)

st.info(
    f"""
📌 Peak daily new cases recorded:
{max_cases:,}

📌 Average daily new cases:
{avg_cases:,}

📌 Current mortality rate:
{mortality_rate}%

📌 Current recovery rate:
{recovery_rate}%

📌 AI Risk Classification:
{risk_level}
"""
)

# =====================================================
# DOWNLOAD CSV
# =====================================================

csv = forecast_result.to_csv(
    index=False
)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="covid_forecast.csv",
    mime="text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.success(
    "✅ AI-Powered COVID-19 Forecasting Dashboard Running Successfully"
)