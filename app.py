import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Marketing Study Dashboard",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("marketing_campaign.csv", sep="\t")
    return df


df = load_data()


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

df = df.copy()

# Age
df["Age"] = 2024 - df["Year_Birth"]

# Customer seniority
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
df["Customer_Seniority"] = 2024 - df["Dt_Customer"].dt.year

# Spending columns
spending_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spending_cols].sum(axis=1)

# Campaign response
campaign_cols = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response"
]

df["Total_Campaigns_Accepted"] = df[campaign_cols].sum(axis=1)
df["Campaign_Response"] = np.where(df["Total_Campaigns_Accepted"] > 0, 1, 0)

# Family and purchases
df["Children"] = df["Kidhome"] + df["Teenhome"]

df["Total_Purchases"] = (
    df["NumWebPurchases"]
    + df["NumCatalogPurchases"]
    + df["NumStorePurchases"]
)

df["Digital_Engagement"] = df["NumWebPurchases"] + df["NumWebVisitsMonth"]

# Cleaning
df = df.dropna(subset=["Income"])
df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

# Remove extreme income outliers for clearer visualization
df = df[df["Income"] <= df["Income"].quantile(0.99)]


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.title("Filters")

education_options = sorted(df["Education"].dropna().unique())
marital_options = sorted(df["Marital_Status"].dropna().unique())

education_filter = st.sidebar.multiselect(
    "Education",
    options=education_options,
    default=education_options
)

marital_filter = st.sidebar.multiselect(
    "Marital Status",
    options=marital_options,
    default=marital_options
)

# If the user clears a filter, keep all categories
if len(education_filter) == 0:
    education_filter = education_options

if len(marital_filter) == 0:
    marital_filter = marital_options

df_filtered = df[
    (df["Education"].isin(education_filter))
    & (df["Marital_Status"].isin(marital_filter))
]

if df_filtered.empty:
    st.error("No observations are available with the selected filters. Please adjust the filters.")
    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Marketing Study Dashboard")
st.subheader("Customer Behavior, Campaign Response and Segmentation Analysis")

st.markdown("""
This project analyzes customer-level marketing data to understand customer profiles, spending behavior,
campaign responsiveness, and customer segmentation.

The dashboard combines descriptive statistics, statistical tests, regression models, clustering,
interactive visualizations, and managerial recommendations.

**Important note:** this is an observational marketing study. The results show associations and patterns,
not causal effects.
""")


# --------------------------------------------------
# Key indicators
# --------------------------------------------------

st.markdown("---")
st.subheader("Key Marketing Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Customers", f"{len(df_filtered):,}")
col2.metric("Average Income", f"${df_filtered['Income'].mean():,.0f}")
col3.metric("Average Spending", f"${df_filtered['Total_Spending'].mean():,.0f}")
col4.metric("Campaign Response Rate", f"{df_filtered['Campaign_Response'].mean() * 100:.1f}%")
col5.metric("Average Purchases", f"{df_filtered['Total_Purchases'].mean():.1f}")


# --------------------------------------------------
# Research design
# --------------------------------------------------

st.markdown("---")
st.subheader("Research Design")

st.markdown("""
This marketing study is organized around six main research questions:

1. **Who are the customers?**
2. **What are the main spending patterns?**
3. **Are campaign-responsive customers more valuable?**
4. **Which factors are associated with customer spending?**
5. **Which factors are associated with campaign response?**
6. **Can customers be segmented into meaningful marketing groups?**
""")


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

st.markdown("---")
st.subheader("Dataset Overview")

st.markdown("""
The dataset contains individual-level customer information, including demographic characteristics,
income, family structure, product spending, purchase channels, and responses to previous marketing campaigns.
""")

with st.expander("View first rows of the dataset"):
    st.dataframe(df_filtered.head(100))

with st.expander("View variable types"):
    st.write(df_filtered.dtypes)

st.markdown(f"""
The filtered dataset contains **{df_filtered.shape[0]} customers** and **{df_filtered.shape[1]} variables**.
""")


# --------------------------------------------------
# Question 1
# --------------------------------------------------

st.markdown("---")
st.header("1. Who are the customers?")

st.markdown("""
### Research Question
Who are the customers in this marketing dataset?

### Objective
The objective is to understand the general profile of the customer base.

### Method
We use descriptive statistics and visualizations to analyze age, income, education, marital status,
family structure, customer seniority, purchasing frequency, and total spending.
""")

profile_vars = [
    "Age",
    "Income",
    "Children",
    "Customer_Seniority",
    "Recency",
    "Total_Purchases",
    "Total_Spending"
]

profile_summary = df_filtered[profile_vars].describe().T

st.markdown("### Results: Descriptive Statistics")
st.dataframe(profile_summary)

# Key descriptive indicators
age_mean = df_filtered["Age"].mean()
age_sd = df_filtered["Age"].std()

income_mean = df_filtered["Income"].mean()
income_sd = df_filtered["Income"].std()

spending_mean = df_filtered["Total_Spending"].mean()
spending_sd = df_filtered["Total_Spending"].std()

purchases_mean = df_filtered["Total_Purchases"].mean()
purchases_sd = df_filtered["Total_Purchases"].std()

recency_mean = df_filtered["Recency"].mean()
recency_sd = df_filtered["Recency"].std()

st.markdown(f"""
### Interpretation

The descriptive statistics provide an overview of the main characteristics of the customer base.

- The **average age** of customers is **{age_mean:.1f} years**, with a standard deviation of **{age_sd:.1f} years**.
  This means that customer ages are spread around the average by about **{age_sd:.1f} years**.

- The **average income** is **${income_mean:,.0f}**, with a standard deviation of **${income_sd:,.0f}**.
  This suggests that customer income is relatively heterogeneous, which is important for targeting and segmentation.

- The **average total spending** is **${spending_mean:,.0f}**, with a standard deviation of **${spending_sd:,.0f}**.
  A high standard deviation means that some customers spend much more than others, which helps identify high-value customers.

- The **average number of purchases** is **{purchases_mean:.1f}**, with a standard deviation of **{purchases_sd:.1f}**.
  This indicates that customers differ in their purchasing frequency.

- The **average recency** is **{recency_mean:.1f} days**, with a standard deviation of **{recency_sd:.1f} days**.
  Recency measures the number of days since the last purchase. Lower values indicate more recent customers.

Overall, the descriptive results show whether the customer base is homogeneous or heterogeneous.
This is important before moving to segmentation and campaign-response analysis.
""")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df_filtered,
        x="Age",
        nbins=30,
        title="Age Distribution"
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_income = px.histogram(
        df_filtered,
        x="Income",
        nbins=30,
        title="Income Distribution"
    )
    st.plotly_chart(fig_income, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    edu_count = df_filtered["Education"].value_counts().reset_index()
    edu_count.columns = ["Education", "Customers"]

    fig_edu = px.bar(
        edu_count,
        x="Education",
        y="Customers",
        title="Customer Distribution by Education Level"
    )
    st.plotly_chart(fig_edu, use_container_width=True)

with col2:
    marital_count = df_filtered["Marital_Status"].value_counts().reset_index()
    marital_count.columns = ["Marital_Status", "Customers"]

    fig_marital = px.bar(
        marital_count,
        x="Marital_Status",
        y="Customers",
        title="Customer Distribution by Marital Status"
    )
    st.plotly_chart(fig_marital, use_container_width=True)


# --------------------------------------------------
# Question 2
# --------------------------------------------------

st.markdown("---")
st.header("2. What are the main spending patterns?")

st.markdown("""
### Research Question
What are the main spending patterns across product categories and education levels?

### Objective
The objective is to identify the product categories that generate the highest average customer spending
and to examine how spending and income vary across education levels.

### Method
We calculate total spending and average spending by product category. We also compare average income
and spending across education levels to examine how socioeconomic profiles relate to customer value.
""")

spending_summary = df_filtered[spending_cols].mean().reset_index()
spending_summary.columns = ["Product Category", "Average Spending"]
spending_summary = spending_summary.sort_values("Average Spending", ascending=False)

st.markdown("### Results: Average Spending by Product Category")
st.dataframe(spending_summary)

fig_spending = px.bar(
    spending_summary,
    x="Product Category",
    y="Average Spending",
    title="Average Spending by Product Category"
)
st.plotly_chart(fig_spending, use_container_width=True)

education_summary = df_filtered.groupby("Education").agg(
    Average_Income=("Income", "mean"),
    Average_Spending=("Total_Spending", "mean"),
    Median_Spending=("Total_Spending", "median"),
    Average_Purchases=("Total_Purchases", "mean"),
    Customers=("ID", "count")
).reset_index()

st.markdown("### Results: Income and Spending by Education Level")
st.dataframe(education_summary)

col1, col2 = st.columns(2)

with col1:
    fig_income_edu = px.bar(
        education_summary,
        x="Education",
        y="Average_Income",
        title="Average Income by Education Level"
    )
    st.plotly_chart(fig_income_edu, use_container_width=True)

with col2:
    fig_spending_edu = px.bar(
        education_summary,
        x="Education",
        y="Average_Spending",
        title="Average Spending by Education Level"
    )
    st.plotly_chart(fig_spending_edu, use_container_width=True)

fig_box_edu = px.box(
    df_filtered,
    x="Education",
    y="Total_Spending",
    title="Distribution of Total Spending by Education Level"
)
st.plotly_chart(fig_box_edu, use_container_width=True)

top_category = spending_summary.iloc[0]["Product Category"]
top_category_value = spending_summary.iloc[0]["Average Spending"]

top_education_income = education_summary.sort_values("Average_Income", ascending=False).iloc[0]["Education"]
top_education_spending = education_summary.sort_values("Average_Spending", ascending=False).iloc[0]["Education"]

st.markdown(f"""
### Interpretation

The spending analysis highlights the main sources of customer value.

- The product category with the highest average spending is **{top_category}**, with an average of **${top_category_value:,.2f}** per customer.
  This category appears to be the most important contributor to customer spending.

- The education group with the highest average income is **{top_education_income}**.

- The education group with the highest average spending is **{top_education_spending}**.

If the same education group has both high income and high spending, it may represent a particularly attractive segment.
If income is high but spending is not, this may indicate an untapped potential for better targeting.
""")


# --------------------------------------------------
# Question 3
# --------------------------------------------------

st.markdown("---")
st.header("3. Are campaign-responsive customers more valuable?")

st.markdown("""
### Research Question
Do customers who respond to marketing campaigns show different spending and purchasing behavior?

### Objective
The objective is to compare campaign-responsive customers with non-responsive customers.

### Method
We compare average income, total spending, purchases, recency, family structure, and digital engagement
between customers who accepted at least one campaign and customers who did not accept any campaign.

This is a descriptive and statistical comparison, not a causal impact evaluation.
""")

campaign_summary = df_filtered.groupby("Campaign_Response").agg(
    Customers=("ID", "count"),
    Average_Income=("Income", "mean"),
    Average_Spending=("Total_Spending", "mean"),
    Average_Purchases=("Total_Purchases", "mean"),
    Average_Recency=("Recency", "mean"),
    Average_Children=("Children", "mean"),
    Average_Digital_Engagement=("Digital_Engagement", "mean")
).reset_index()

campaign_summary["Group"] = campaign_summary["Campaign_Response"].map({
    0: "Non-responsive customers",
    1: "Campaign-responsive customers"
})

campaign_summary = campaign_summary[
    [
        "Group",
        "Customers",
        "Average_Income",
        "Average_Spending",
        "Average_Purchases",
        "Average_Recency",
        "Average_Children",
        "Average_Digital_Engagement"
    ]
]

st.markdown("### Results: Comparison Between Customer Groups")
st.dataframe(campaign_summary)

fig_campaign_spending = px.bar(
    campaign_summary,
    x="Group",
    y="Average_Spending",
    title="Average Spending by Campaign Response"
)
st.plotly_chart(fig_campaign_spending, use_container_width=True)

fig_campaign_purchases = px.bar(
    campaign_summary,
    x="Group",
    y="Average_Purchases",
    title="Average Purchases by Campaign Response"
)
st.plotly_chart(fig_campaign_purchases, use_container_width=True)

fig_campaign_box = px.box(
    df_filtered,
    x="Campaign_Response",
    y="Total_Spending",
    title="Distribution of Total Spending by Campaign Response",
    labels={"Campaign_Response": "Campaign Response"}
)
st.plotly_chart(fig_campaign_box, use_container_width=True)

st.markdown("### Statistical Test: Difference in Average Spending")

group_nonresponsive = df_filtered[df_filtered["Campaign_Response"] == 0]["Total_Spending"]
group_responsive = df_filtered[df_filtered["Campaign_Response"] == 1]["Total_Spending"]

if len(group_nonresponsive) > 1 and len(group_responsive) > 1:
    t_stat, p_value = stats.ttest_ind(
        group_nonresponsive,
        group_responsive,
        equal_var=False,
        nan_policy="omit"
    )

    test_results = pd.DataFrame({
        "Comparison": ["Responsive vs Non-responsive customers"],
        "Mean Non-responsive": [group_nonresponsive.mean()],
        "Mean Responsive": [group_responsive.mean()],
        "Mean Difference": [group_responsive.mean() - group_nonresponsive.mean()],
        "T-statistic": [t_stat],
        "P-value": [p_value]
    })

    st.dataframe(test_results)

    if p_value < 0.05:
        st.success("The difference in average spending is statistically significant at the 5% level.")
    else:
        st.warning("The difference in average spending is not statistically significant at the 5% level.")

    st.markdown(f"""
    ### Interpretation

    The comparison shows whether campaign-responsive customers are different from non-responsive customers in terms of spending.

    - Non-responsive customers spend on average **${group_nonresponsive.mean():,.2f}**.
    - Campaign-responsive customers spend on average **${group_responsive.mean():,.2f}**.
    - The estimated difference is **${group_responsive.mean() - group_nonresponsive.mean():,.2f}**.
    - The p-value is **{p_value:.4f}**.

    If the p-value is below 0.05, the difference is statistically significant.
    However, the result remains observational. It does not prove that campaign response caused higher spending.
    """)

else:
    st.warning("Not enough observations in both groups to run the t-test.")


# --------------------------------------------------
# Question 4
# --------------------------------------------------

st.markdown("---")
st.header("4. Which factors are associated with customer spending?")

st.markdown("""
### Research Question
Which customer characteristics are associated with total spending?

### Objective
The objective is to identify the variables that are statistically associated with customer spending.

### Method
We estimate an OLS regression model:

Total Spending = f(Income, Age, Campaign Response, Children, Total Purchases, Customer Seniority)
""")

reg_df = df_filtered[
    [
        "Total_Spending",
        "Income",
        "Age",
        "Campaign_Response",
        "Children",
        "Total_Purchases",
        "Customer_Seniority"
    ]
].dropna()

if len(reg_df) < 10:
    st.warning("Not enough observations to estimate the OLS regression with the current filters.")
else:
    X = reg_df[
        [
            "Income",
            "Age",
            "Campaign_Response",
            "Children",
            "Total_Purchases",
            "Customer_Seniority"
        ]
    ]

    X = sm.add_constant(X)
    y = reg_df["Total_Spending"]

    try:
        ols_model = sm.OLS(y, X).fit()

        ols_results = pd.DataFrame({
            "Variable": ols_model.params.index,
            "Coefficient": ols_model.params.values,
            "P-value": ols_model.pvalues.values
        })

        ols_results["Significance"] = np.where(
            ols_results["P-value"] < 0.05,
            "Significant",
            "Not significant"
        )

        st.markdown("### Results: OLS Regression")
        st.dataframe(ols_results)

        st.markdown(f"""
        The R-squared of the model is **{ols_model.rsquared:.3f}**.

        This means that approximately **{ols_model.rsquared * 100:.1f}%** of the variation in total customer spending
        is explained by the variables included in the model.
        """)

        coef_plot = ols_results[ols_results["Variable"] != "const"].copy()

        fig_coef = px.bar(
            coef_plot,
            x="Variable",
            y="Coefficient",
            color="Significance",
            title="OLS Coefficients: Drivers of Customer Spending"
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("### Interpretation")

        important_vars = ["Income", "Campaign_Response", "Children", "Total_Purchases"]

        for var in important_vars:
            if var in ols_results["Variable"].values:
                row = ols_results[ols_results["Variable"] == var].iloc[0]
                coef = row["Coefficient"]
                pval = row["P-value"]

                significance_text = (
                    "statistically significant at the 5% level"
                    if pval < 0.05
                    else "not statistically significant at the 5% level"
                )

                if var == "Income":
                    st.markdown(f"""
- **Income:** The coefficient is **{coef:.4f}** and is **{significance_text}**.
  Holding other variables constant, an increase of **$1,000 in income** is associated with an estimated change of
  **${coef * 1000:,.2f}** in total spending.
""")

                elif var == "Campaign_Response":
                    st.markdown(f"""
- **Campaign Response:** The coefficient is **{coef:.4f}** and is **{significance_text}**.
  This compares customers who accepted at least one campaign with those who did not, holding other variables constant.
  A positive coefficient suggests that campaign-responsive customers tend to spend more on average.
  This is an association, not a causal effect.
""")

                elif var == "Children":
                    st.markdown(f"""
- **Children:** The coefficient is **{coef:.4f}** and is **{significance_text}**.
  Holding other variables constant, one additional child or teenager at home is associated with a change of
  **${coef:,.2f}** in total spending.
""")

                elif var == "Total_Purchases":
                    st.markdown(f"""
- **Total Purchases:** The coefficient is **{coef:.4f}** and is **{significance_text}**.
  Holding other variables constant, one additional purchase is associated with an estimated change of
  **${coef:,.2f}** in total spending.
  This is one of the most important variables because purchasing frequency is directly linked to customer value.
""")

        with st.expander("View full OLS regression output"):
            st.text(ols_model.summary())

    except Exception as e:
        st.warning("The OLS regression could not be estimated with the current filters.")
        st.write(e)


# --------------------------------------------------
# Question 5
# --------------------------------------------------

st.markdown("---")
st.header("5. Which factors are associated with campaign response?")

st.markdown("""
### Research Question
Which customer characteristics are associated with the probability of responding to a campaign?

### Objective
The objective is to identify customer profiles that are more likely to respond to marketing campaigns.

### Method
We estimate a logistic regression model:

Campaign Response = f(Income, Age, Children, Total Purchases, Total Spending, Recency, Digital Engagement)
""")

logit_df = df_filtered[
    [
        "Campaign_Response",
        "Income",
        "Age",
        "Children",
        "Total_Purchases",
        "Total_Spending",
        "Recency",
        "Digital_Engagement"
    ]
].dropna()

if len(logit_df) < 20 or logit_df["Campaign_Response"].nunique() < 2:
    st.warning("Not enough observations or variation to estimate the logistic regression with the current filters.")
else:
    X_logit = logit_df[
        [
            "Income",
            "Age",
            "Children",
            "Total_Purchases",
            "Total_Spending",
            "Recency",
            "Digital_Engagement"
        ]
    ]

    X_logit = sm.add_constant(X_logit)
    y_logit = logit_df["Campaign_Response"]

    try:
        logit_model = sm.Logit(y_logit, X_logit).fit(disp=False)

        logit_results = pd.DataFrame({
            "Variable": logit_model.params.index,
            "Coefficient": logit_model.params.values,
            "P-value": logit_model.pvalues.values
        })

        logit_results["Odds Ratio"] = np.exp(logit_results["Coefficient"])

        logit_results["Significance"] = np.where(
            logit_results["P-value"] < 0.05,
            "Significant",
            "Not significant"
        )

        st.markdown("### Results: Logistic Regression")
        st.dataframe(logit_results)

        odds_plot = logit_results[logit_results["Variable"] != "const"].copy()

        fig_odds = px.bar(
            odds_plot,
            x="Variable",
            y="Odds Ratio",
            color="Significance",
            title="Odds Ratios: Factors Associated with Campaign Response"
        )
        st.plotly_chart(fig_odds, use_container_width=True)

        st.markdown("### Interpretation")

        important_logit_vars = ["Income", "Total_Spending", "Recency", "Digital_Engagement"]

        for var in important_logit_vars:
            if var in logit_results["Variable"].values:
                row = logit_results[logit_results["Variable"] == var].iloc[0]
                odds = row["Odds Ratio"]
                pval = row["P-value"]

                significance_text = (
                    "statistically significant at the 5% level"
                    if pval < 0.05
                    else "not statistically significant at the 5% level"
                )

                if odds > 1:
                    direction = "higher"
                else:
                    direction = "lower"

                if var == "Income":
                    st.markdown(f"""
- **Income:** The odds ratio is **{odds:.3f}** and is **{significance_text}**.
  This indicates whether customers with higher income are more or less likely to respond to campaigns.
  Since income is measured in monetary units, the effect of one dollar is usually very small.
""")

                elif var == "Total_Spending":
                    st.markdown(f"""
- **Total Spending:** The odds ratio is **{odds:.3f}** and is **{significance_text}**.
  This shows whether higher-value customers are associated with a **{direction}** likelihood of campaign response.
""")

                elif var == "Recency":
                    st.markdown(f"""
- **Recency:** The odds ratio is **{odds:.3f}** and is **{significance_text}**.
  Recency measures the number of days since the last purchase.
  If the odds ratio is below 1, more recent customers are more likely to respond.
  If it is above 1, customers with a longer time since their last purchase are more likely to respond.
""")

                elif var == "Digital_Engagement":
                    st.markdown(f"""
- **Digital Engagement:** The odds ratio is **{odds:.3f}** and is **{significance_text}**.
  This indicates whether customers with more web purchases and website visits are associated with a **{direction}**
  likelihood of responding to campaigns.
""")

        with st.expander("View full logistic regression output"):
            st.text(logit_model.summary())

    except Exception as e:
        st.warning("The logistic regression could not be estimated with the current filters.")
        st.write(e)


# --------------------------------------------------
# Question 6
# --------------------------------------------------

st.markdown("---")
st.header("6. Can customers be segmented into meaningful marketing groups?")

st.markdown("""
### Research Question
Can customers be grouped into meaningful segments based on their behavior and value?

### Objective
The objective is to identify customer segments that can guide marketing strategy.

### Method
We use K-means clustering based on income, total spending, total purchases, recency,
campaign response, and digital engagement.
""")

segmentation_vars = [
    "Income",
    "Total_Spending",
    "Total_Purchases",
    "Recency",
    "Total_Campaigns_Accepted",
    "Digital_Engagement"
]

seg_df = df_filtered[segmentation_vars].dropna()

if len(seg_df) < 10:
    st.warning("Not enough observations to perform customer segmentation with the current filters.")
else:
    try:
        scaler = StandardScaler()
        seg_scaled = scaler.fit_transform(seg_df)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(seg_scaled)

        df_seg = df_filtered.loc[seg_df.index].copy()
        df_seg["Segment"] = clusters

        segment_summary = df_seg.groupby("Segment").agg(
            Customers=("ID", "count"),
            Average_Income=("Income", "mean"),
            Average_Spending=("Total_Spending", "mean"),
            Average_Purchases=("Total_Purchases", "mean"),
            Average_Recency=("Recency", "mean"),
            Campaigns_Accepted=("Total_Campaigns_Accepted", "mean"),
            Digital_Engagement=("Digital_Engagement", "mean")
        ).reset_index()

        st.markdown("### Results: Segment Summary")
        st.dataframe(segment_summary)

        fig_segments = px.scatter(
            df_seg,
            x="Income",
            y="Total_Spending",
            color="Segment",
            size="Total_Purchases",
            hover_data=["Age", "Education", "Marital_Status"],
            title="Customer Segments: Income vs Total Spending"
        )
        st.plotly_chart(fig_segments, use_container_width=True)

        segment_long = segment_summary.melt(
            id_vars="Segment",
            value_vars=[
                "Average_Income",
                "Average_Spending",
                "Average_Purchases",
                "Average_Recency",
                "Campaigns_Accepted",
                "Digital_Engagement"
            ],
            var_name="Indicator",
            value_name="Value"
        )

        fig_segment_bar = px.bar(
            segment_long,
            x="Indicator",
            y="Value",
            color="Segment",
            barmode="group",
            title="Segment Profiles Across Key Marketing Indicators"
        )
        st.plotly_chart(fig_segment_bar, use_container_width=True)

        highest_spending_segment = segment_summary.sort_values("Average_Spending", ascending=False).iloc[0]["Segment"]
        highest_campaign_segment = segment_summary.sort_values("Campaigns_Accepted", ascending=False).iloc[0]["Segment"]
        highest_digital_segment = segment_summary.sort_values("Digital_Engagement", ascending=False).iloc[0]["Segment"]

        st.markdown(f"""
        ### Interpretation

        The clustering analysis identifies four customer segments based on income, spending, purchases,
        recency, campaign response, and digital engagement.

        - Segment **{highest_spending_segment}** has the highest average spending and can be interpreted as the strongest
          **high-value customer segment**.

        - Segment **{highest_campaign_segment}** has the highest average campaign acceptance and may represent the most
          **campaign-responsive segment**.

        - Segment **{highest_digital_segment}** has the highest digital engagement and may be particularly relevant
          for online campaigns, web-based promotions, or digital loyalty strategies.

        These segments can support differentiated marketing actions, such as loyalty programs for high-value customers,
        personalized offers for campaign-responsive customers, and digital engagement strategies for online-active customers.
        """)

    except Exception as e:
        st.warning("Customer segmentation could not be performed with the current filters.")
        st.write(e)


# --------------------------------------------------
# Additional Chi-square test
# --------------------------------------------------

st.markdown("---")
st.header("Additional Test: Education and Campaign Response")

st.markdown("""
### Research Question
Is campaign response associated with education level?

### Objective
The objective is to examine whether campaign response differs across education groups.

### Method
We use a Chi-square test of independence between education level and campaign response.
""")

contingency_table = pd.crosstab(df_filtered["Education"], df_filtered["Campaign_Response"])

if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
    st.warning("Not enough variation to run the Chi-square test with the current filters.")
else:
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)

    st.markdown("### Results: Contingency Table")
    st.dataframe(contingency_table)

    chi_results = pd.DataFrame({
        "Chi-square statistic": [chi2],
        "P-value": [chi2_p],
        "Degrees of freedom": [dof]
    })

    st.dataframe(chi_results)

    if chi2_p < 0.05:
        st.success("Education level and campaign response are statistically associated at the 5% level.")
    else:
        st.warning("No statistically significant association is found between education level and campaign response at the 5% level.")

    st.markdown(f"""
    ### Interpretation

    The Chi-square test examines whether education level and campaign response are statistically associated.

    - The p-value is **{chi2_p:.4f}**.

    If the p-value is below 0.05, campaign response differs significantly across education levels.
    If the p-value is above 0.05, there is not enough statistical evidence to conclude that education and campaign response are associated.
    """)


# --------------------------------------------------
# Final recommendations
# --------------------------------------------------

st.markdown("---")
st.header("Managerial Recommendations")

st.markdown("""
Based on the analysis, the company could consider the following actions:

1. **Prioritize high-value customers** with loyalty programs and premium offers.
2. **Target campaign-responsive customers** with personalized promotions.
3. **Use customer segmentation** to design different strategies for different customer profiles.
4. **Focus on high-spending product categories** to maximize marketing returns.
5. **Improve engagement among low-response customers** using more personalized digital campaigns.
6. **Use predictive models** to identify customers more likely to respond to future campaigns.
""")


# --------------------------------------------------
# Conclusion
# --------------------------------------------------

st.markdown("---")
st.header("Conclusion")

st.markdown("""
This marketing study demonstrates how customer-level data can support business and marketing decision-making.

The dashboard combines:

- customer profiling;
- spending analysis;
- campaign response comparison;
- statistical testing;
- OLS regression;
- logistic regression;
- K-means customer segmentation;
- managerial recommendations.

The results should be interpreted as associations rather than causal effects. A future extension could use
a randomized A/B test or a Difference-in-Differences design to estimate the causal impact of a marketing campaign.
""")
