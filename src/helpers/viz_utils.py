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
from sklearn.preprocessing import LabelEncoder
from scipy import stats

from helpers.data_preproccesing import run_preprocess



def column_analysis(col: str, dataset: pd.DataFrame) -> None:
    """
    Displays a chart to analyse a column.

    :param col: Column to analyze
    :param dataset: Dataset
    """
    st.markdown(f"## 📌 Analysis of {col}")

    l1, r1 = st.columns(2)
    l1.metric("🔢 Unique values", dataset[col].nunique())
    r1.metric("🚫 Null values", dataset[col].isnull().sum())
    st.markdown("---")
    cat_viz = st.checkbox("Categorical Visualization")  # force the cat. viz

    # Numerical column
    if pd.api.types.is_numeric_dtype(dataset[col]) and not cat_viz:
        st.line_chart(dataset[col].tolist())

        st.markdown("### 📊 Summary stats")
        st.dataframe(dataset[col].describe().to_frame().T, use_container_width=True)

    # Categorical column
    elif not pd.api.types.is_numeric_dtype(dataset[col]) or cat_viz:
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


def df_info_table(
    df: pd.DataFrame,
    include_nulls: bool = True,
    include_unique: bool = False,
    deep_memory: bool = True,
):
    """
    Create a DataFrame similar to the one produced by df.info().

    :param df: Input DataFrame
    :param include_nulls: Whether to include a column for null counts (default: True)
    :param include_unique: Whether to include a column for unique value counts (default: False)
    :param deep_memory: Whether to calculate deep memory usage (default: True)
    """

    data = {
        "Column": df.columns,
        "Non-Null Count": df.notna().sum().to_numpy(),
        "Dtype": df.dtypes.astype(str).to_numpy(),
    }

    if include_nulls:
        data["Null Count"] = df.isna().sum().to_numpy()
    if include_unique:
        # nombre de valeurs distinctes (hors NaN)
        data["Unique"] = df.nunique(dropna=True).to_numpy()

    info_df = pd.DataFrame(data)
    info_df.insert(0, "#", range(len(info_df)))  # index comme dans info()

    mem_bytes = df.memory_usage(deep=deep_memory, index=True).sum()

    def human_bytes(n):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024 or unit == "TB":
                return f"{n:.1f} {unit}"
            n /= 1024.0

    return info_df, human_bytes(mem_bytes)

def correlation_matrix(df: pd.DataFrame) -> None:
    """
    Display a correlation matrix for the numerical features in the dataset.

    :param df: Dataset
    """
    le = LabelEncoder()
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category':
            le = LabelEncoder()
            df_copy[col] = df_copy[col].fillna('__nan__').astype(str)
            df_copy[col] = le.fit_transform(df_copy[col])

    corr = df_copy.corr(numeric_only=True).round(2)

    fig = px.imshow(
        corr,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        labels={'x': 'Feature', 'y': 'Feature', 'color': 'Correlation'},
        title='Correlation Matrix'
    )
    fig.update_layout(width=800, height=800)

    return fig


def plot_correlation(df: pd.DataFrame, col1: str, col2: str) -> any:
    """
    Displays a scatter plot with a trendline to visualize the correlation between two columns.

    :param df: Input DataFrame.
    :param col1: Name of the first column (x-axis).
    :param col2: Name of the second column (y-axis).
    :return: The figure object containing the scatter plot with the trendline and correlation statistics in the title.
    """
    data = df[[col1, col2]].dropna()
    r, p = stats.pearsonr(data[col1], data[col2])

    fig = px.scatter(
        data, x=col1, y=col2,
        trendline="ols",
        title=f"Correlation: {col1} vs {col2} — Pearson r = {r:.3f} (p = {p:.3g})"
    )

    return fig

def visualize_class_apparition(
    train_df: pd.DataFrame, class_label: str
) -> any:
    """
    Visualizes the appearance of classes over time (index).

    :param train_df: DataFrame containing the training data with a class label column.
    :param class_label: Name of the column containing the class labels.
    :return: plotly.graph_objs.Figure
    """

    df = train_df.reset_index()

    fig = px.scatter(
        df,
        x="index",
        y=class_label,
        color=class_label,
        title=f"{class_label} apparition through time",
        labels={"index": "Sample Index", class_label: "Attack Category"},
        opacity=0.6,
        height=600,
        width=900,
    )

    fig.update_traces(marker=dict(size=6), showlegend=False)
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=80, b=40))

    return fig
