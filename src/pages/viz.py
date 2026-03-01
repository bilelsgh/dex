import sys

sys.path.append("..")

import pandas as pd
import plotly.express as px
import streamlit as st

from helpers.viz_utils import (
    column_analysis,
    dataset_unique_value_chart,
    dataset_variance_chart,
    df_info_table,
    correlation_matrix,
    plot_correlation,
    visualize_class_apparition
)

from helpers.utils import load_data

# == Page config
st.set_page_config(page_title="Home", page_icon="")

# st.set_page_config(layout="wide")
st.title("DeX - Analysis")

dataset_df = pd.DataFrame({})

# Data upload
st.markdown("")
with st.expander("⬆️ Upload your dataset"):
    dataset_csv = st.file_uploader("Upload", type="csv")
    try:
        dataset_df = load_data(dataset_csv)
        st.success("Dataset loaded successfully! 🎉")
    except ValueError:
        st.error("Please upload your dataset. Only .csv files are supported.")

# ==== Overview metrics ====
if len(dataset_df):
    nb_lines, nb_features, size = (
        len(dataset_df),
        len(dataset_df.columns),
        round(dataset_csv.size / (1024**2), 1),
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 Rows", nb_lines)
    col2.metric("📐 Columns", nb_features)
    col3.metric("💾 Size", f"{size} Mo")

    # ===== Raw data ====
    with st.expander("🥩 Raw data"):
        st.dataframe(dataset_df)

    # ===== Dataset description =====
    with st.expander("📄 Description"):
        l1, r1 = st.columns(2)

        # == dataframe statistics
        l1.markdown(" ### Statistics")
        l1.dataframe(dataset_df.describe())

        # == dataframe info
        r1.markdown(" ### Info")
        info_df, mem_str = df_info_table(
            dataset_df, include_nulls=True, include_unique=False, deep_memory=True
        )
        r1.dataframe(info_df, hide_index=True, height=315)
        r1.caption(f"Memory usage (deep): {mem_str}")

    # ==== Feature overview ===
    with st.expander("🔍 Overview"):

        # == features selection
        removed_features = st.multiselect(
            "I remove..", dataset_df.columns, default=[], placeholder="Chose features"
        )
        type_filter = st.segmented_control(
            "Type filter", dataset_df.dtypes.unique().tolist(), selection_mode="multi"
        )
        df_overview = dataset_df.drop(
            removed_features, axis=1
        )  # dataframe to use for this section
        df_overview = (
            df_overview.select_dtypes(include=type_filter)
            if type_filter
            else df_overview
        )  # filter on type
        l2, r2 = st.columns(2)

        # == column distribution
        type_counts = df_overview.dtypes.value_counts()
        type_fig = px.pie(
            names=type_counts.index.astype(str),
            values=type_counts.values,
            title="Column Types",
        )
        r2.plotly_chart(type_fig)

        # == variance
        with st.spinner("..Computing variances"):
            var_fig = dataset_variance_chart(df_overview)
        if var_fig:
            l2.plotly_chart(var_fig)

        # == unique values
        with st.spinner("..Checking unique values"):
            unq_fig = dataset_unique_value_chart(df_overview)
        st.plotly_chart(unq_fig)

    # # ==== Columns analysis ====
    with st.expander("🔥 Class apparition"):
        option_app = st.selectbox(
            "Class to visualize",
            dataset_df.columns,
        )
        st.plotly_chart(
            visualize_class_apparition(dataset_df, option_app)
        )

    # # ==== Columns analysis ====
    with st.expander("📍 Correlation"):
        st.plotly_chart(
            correlation_matrix(dataset_df)
        )
        st.markdown("---")
        col_corr = st.multiselect( # check correlation between two columns
            "Check correlation between columns",
            dataset_df.columns,
            default=[],
            max_selections=2,
        )
        if len(col_corr) == 2:
            st.plotly_chart(
                plot_correlation(dataset_df, *col_corr)
            )

    # # ==== Columns analysis ====
    with st.expander("💡 Analysis"):
        selected_col = st.selectbox("Columns", dataset_df.columns.tolist())
        st.markdown("---")
        low_rows_bound, high_rows_bound = st.slider(
            "Select the rows you want to analyse", value=(0, len(dataset_df) - 1)
        )
        if selected_col:
            column_analysis(
                selected_col, dataset_df.iloc[low_rows_bound:high_rows_bound, :]
            )
