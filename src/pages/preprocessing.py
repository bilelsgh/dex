"""
Page to preprocess the dataset
"""

import sys

sys.path.append("../..")

import pandas as pd
import streamlit as st

from helpers.dashboard_utils import load_data, sumup_operations

# == Page config
st.set_page_config(page_title="Preprocessing", page_icon="")

# ==
st.title("DeX - Preprocessing")
dataset_df = pd.DataFrame({})

with st.expander("⬆️ Upload your dataset"):
    dataset_csv = st.file_uploader("Upload", type="csv")

    try:
        dataset_df = load_data(dataset_csv)
        st.success("Dataset loaded successfully! 🎉")
    except ValueError:
        st.error("Please upload your dataset. Only .csv files are supported.")

if len(dataset_df):

    operations = {}
    if st.toggle("Remove invalid value (NaN, inf ...)"):
        operations["remove_inv_val"] = {}

    with st.expander("Encoding"):
        # == ohe
        ohe_col = st.multiselect(
            "Columns to encode with One Hot Encoding", options=dataset_df.columns
        )
        c1, c2 = st.columns(2)

        # == l.encoding
        le_col = [col for col in dataset_df.columns if col not in ohe_col]
        if not c1.checkbox("Use Label Encoder for the remaining columns"):
            le_col = c2.multiselect(
                "Columns to encode with Label Encoder",
                options=[col for col in dataset_df.columns if col not in ohe_col],
            )

        operations["encoding"] = {"ohe": ohe_col, "le": le_col}

    with st.expander("Normalization"):

        normalization_op = st.selectbox(
            "Normalization operation", options=["Standardization", "MinMax"]
        )
        c3, c4 = st.columns(2)

        normalization_col = []
        if not c3.checkbox("Normalize every numeric column"):
            normalization_col = st.multiselect(
                "Columns to normalize", options=dataset_df.columns
            )

        operations[normalization_op.lower()] = {"col": normalization_col}

    # with st.expander("Dimension reduction"):
    #     nb_dim = st.number_input('Number of dimensions to keep')
    #     min_var = st.number_input('Minimum variance')
    #     operations['dimension_reduction'] = [nb_dim, min_var]
    #
    # with st.expander("Undersampling"):
    #     pass

    c5, c6 = st.columns(2)
    if c5.button("Run", type="primary"):
        preprocessed_dataset = sumup_operations(operations, dataset_df)

    if "enc_dataset" in st.session_state:
        c6.download_button(
            "Download encoded dataset",
            (st.session_state["enc_dataset"]).to_csv(index=False),
            file_name="processed_dataset.csv",
            mime="text/csv",
        )
