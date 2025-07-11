import io
import sys

sys.path.append("../src")

import pandas as pd
import plotly.express as px
import streamlit as st

from helpers.dashboard_utils import (
    column_analysis,
    dataset_unique_value_chart,
    dataset_variance_chart,
)

st.set_page_config(layout="wide")
st.title("Dataset preparation")

l0, r0 = st.columns(2)
dataset_df = pd.DataFrame({})

# Data upload
with l0:
    st.markdown("")
    with st.expander("Upload your dataset"):
        dataset_csv = st.file_uploader("", type="csv")
        try:
            dataset_df = pd.read_csv(dataset_csv)
            st.success("Dataset loaded successfully! 🎉")
        except ValueError:
            st.error("Please upload your dataset. Only .csv files are supported.")

# ==== Overview metrics ====
if len(dataset_df):
    with r0:
        nb_lines, nb_features, size = (
            len(dataset_df),
            len(dataset_df.columns),
            round(dataset_csv.size / (1024**2), 1),
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Rows", nb_lines)
        col2.metric("📐 Columns", nb_features)
        col3.metric("💾 Size (MB)", f"{size} Mo")

    # ===== Raw data ====
    with st.expander("Raw data"):
        st.dataframe(dataset_df)

    # ===== Dataset description =====
    with st.expander("Overview"):
        l1, r1 = st.columns(2)

        # == dataframe statistics
        l1.markdown(" ### Statistics")
        l1.dataframe(dataset_df.describe())

        # == dataframe info
        r1.markdown(" ### Info")
        buffer = io.StringIO()
        dataset_df.info(buf=buffer)  # store it rather than display it in stdout
        lines = buffer.getvalue().splitlines()
        df = (
            pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
            .drop(["Count", "#"], axis=1)
            .rename(columns={"Non-Null": "Non-Null Count"})
        )  # convert to df
        r1.dataframe(df, hide_index=True, height=315)

        # == column distribution
        type_counts = dataset_df.dtypes.value_counts()
        type_fig = px.pie(
            names=type_counts.index.astype(str),
            values=type_counts.values,
            title="Column Types",
        )
        r1.plotly_chart(type_fig)

        # == variance
        var_fig = dataset_variance_chart(dataset_df)
        if var_fig:
            l1.plotly_chart(var_fig)

        # == unique values
        unq_fig = dataset_unique_value_chart(dataset_df)
        st.plotly_chart(unq_fig)

    # ==== Columns analysis ====
    with st.expander("Analysis"):
        selected_col = st.pills("", dataset_df.columns.tolist())
        if selected_col:
            column_analysis(selected_col, dataset_df)
