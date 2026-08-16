import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Indian Digital Life Friction Dashboard")

df = pd.read_csv("data/processed/final_dataset.csv")
all_platforms = sorted(df['platform'].unique().tolist())

platform_filter = st.multiselect(
    "Select platforms",
    options=all_platforms,
    default=all_platforms,
    key="platform_filter"
)

if not platform_filter:
    st.warning("Select at least one platform to see results.")
    st.stop()

filtered = df[df['platform'].isin(platform_filter)]
st.caption(f"Showing {len(filtered)} of {len(df)} total reviews")

col1, col2 = st.columns(2)

with col1:
    friction_counts = filtered[filtered['friction_type'] != 'other']['friction_type'].value_counts()
    fig1 = px.bar(
        x=friction_counts.index,
        y=friction_counts.values,
        title="Friction Types",
        labels={"x": "Friction Type", "y": "Count"}
    )
    st.plotly_chart(fig1)

with col2:
    fig2 = px.pie(filtered, names='sentiment', title="Sentiment Split")
    st.plotly_chart(fig2)

st.subheader("Sample complaints")
st.dataframe(filtered[['platform', 'friction_type', 'sentiment', 'text']].head(50))
