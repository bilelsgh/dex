"""
Various functions used in the streamlit dashboard.
"""

import sys
from typing import Union

sys.path.append("../src")

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objs import Figure

from helpers.data_preproccesing import run_preprocess


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Load a csv file

    :param path: path to csv file
    :return: dataframe
    """
    return pd.read_csv(path)


def column_analysis(col: str, dataset: pd.DataFrame) -> None:
    st.markdown(f"## 📌 Analysis of {col}")

    l1, r1 = st.columns(2)
    l1.metric("🔢 Unique values", dataset[col].nunique())
    r1.metric("🚫 Null values", dataset[col].isnull().sum())
    st.markdown("---")

    # Numerical column
    if pd.api.types.is_numeric_dtype(dataset[col]):
        st.line_chart(dataset[col].tolist())

        st.markdown("### 📊 Summary stats")
        st.dataframe(dataset[col].describe().to_frame().T, use_container_width=True)

    # Categorical column
    else:
        value_counts = dataset[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]
        fig = px.bar(
            value_counts, x=col, y="count", title=f"Occurrences of values in {col}"
        )
        st.plotly_chart(fig)

        with st.expander("More"):
            st.dataframe(value_counts, use_container_width=True)


@st.cache_data
def dataset_variance_chart(dataset: pd.DataFrame) -> Union[None, Figure]:
    """
    Display a chart showing the variance of each column in a dataset.

    :param dataset: Dataset
    :return: None
    """

    numeric_df = dataset.select_dtypes(include="number")
    if not numeric_df.empty:
        variances = numeric_df.std().sort_values(ascending=False)
        var_df = pd.DataFrame({"Column": variances.index, "Variance": variances.values})
        fig = px.bar(
            var_df,
            x="Column",
            y="Variance",
            orientation="v",
            title="Variance per Numerical Feature",
            height=500,
            labels={"Variance": "Variance", "Column": "Feature"},
        )
        return fig

    return None


@st.cache_data
def dataset_unique_value_chart(dataset: pd.DataFrame) -> Figure:
    """
    Display a chart counting the number of unique values of each column in a dataset.

    :param dataset: Dataset
    :return:
    """

    unique_vals = dataset.nunique().sort_values(ascending=False)
    unique_df = pd.DataFrame(
        {"Column": unique_vals.index, "Unique Values": unique_vals.values}
    )
    return px.bar(
        unique_df,
        x="Column",
        y="Unique Values",
        orientation="v",
        title="Number of Unique Values per Column",
        labels={"Unique Values": "Count", "Column": "Feature"},
    )


@st.dialog("Are you sure?")
def sumup_operations(operations: dict, dataset: pd.DataFrame) -> None:
    st.markdown("### You are about to run")

    txt = "\n".join([f"- {op}" for op, _ in operations.items()])
    st.markdown(txt)

    if st.button("Confirm"):
        st.session_state["enc_dataset"] = run_preprocess(operations, dataset)
        st.rerun()
