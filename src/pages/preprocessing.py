"""
Page to preprocess the dataset
"""

import json
import sys

sys.path.append("../..")

import pandas as pd
import streamlit as st

from helpers.dashboard_utils import load_data, sumup_operations
from helpers.data_preproccesing import split_datasets
from helpers.dataset_tools import DatasetDownloader

# == Page config
st.set_page_config(page_title="Preprocessing", page_icon="")

# ==
st.title("DeX - Preprocessing")
dataset_df = pd.DataFrame({})

with st.expander("⬆️ Upload your dataset"):
    datasets_csv = st.file_uploader(
        "Upload", type="csv", accept_multiple_files=True
    )  # multiple files for train, test, val

    # Reset session state
    # if "enc_dataset" in st.session_state:
    #     del st.session_state["enc_dataset"]

    # == Load dataset(s)
    try:
        datasets_name = (
            [d.name for d in datasets_csv]
            if len(datasets_csv) > 1
            else [datasets_csv[0].name]
        )
        datasets_df = [load_data(dataset_csv) for dataset_csv in datasets_csv]
        datasets_idx = [
            len(d) for d in datasets_df
        ]  # keep the size of every dataset for splitting after processing

        dataset_df = pd.concat(datasets_df)
        st.success("Dataset loaded successfully! 🎉")
    except (ValueError, IndexError):
        st.error("Please upload your dataset. Only .csv files are supported.")

if len(dataset_df):

    operations = {}
    view = st.radio(
        "How to preprocess?",
        options=["Step by step", "Import a configuration"],
        horizontal=True,
    )
    st.markdown("---")

    # == Processing is done through the graphic interface
    if view == "Step by step":

        with st.expander("Drop values"):
            if st.toggle("Remove invalid value (NaN, inf ...)"):
                operations["remove_inv_val"] = {}

            col_to_drop = st.multiselect("Drop columns:", dataset_df.columns)
            dataset_df = dataset_df.drop(col_to_drop, axis=1)

        with st.expander("Replace values"):
            if st.toggle("Fill values to replace"):
                placeholder = """Please respect the following format:
    {
        <column_name>: [
            {
                "former_values": ["...", ..],
                "new_value": ".."
            }
    ]
        ...
    }
                """
                mapping_txt = st.text_area("Mapping", placeholder=placeholder)

                try:
                    operations["replace_val"] = {"mappings": json.loads(mapping_txt)}
                except json.decoder.JSONDecodeError:
                    st.warning("Please provide a valid JSON")

        with st.expander("Encoding"):
            # == ohe
            if st.toggle("Encode?"):
                # Initialisation dans session_state
                if "ohe_col" not in st.session_state:
                    st.session_state.ohe_col = []
                if "le_col" not in st.session_state:
                    st.session_state.le_col = []

                all_columns = dataset_df.columns.tolist()

                # Column to OHE
                ohe_col = st.multiselect(
                    "Columns to encode with One Hot Encoding",
                    options=all_columns,
                    default=st.session_state.ohe_col,
                )
                st.session_state.ohe_col = ohe_col

                # Columns that are not OHEncoded
                remaining_columns = [col for col in all_columns if col not in ohe_col]

                c1, c2 = st.columns(2)
                use_label = c1.checkbox(
                    "Use Label Encoder for the remaining columns", value=True
                )

                if use_label:
                    le_col = remaining_columns
                else:
                    le_col = c2.multiselect(
                        "Columns to encode with Label Encoder",
                        options=remaining_columns,
                        default=st.session_state.le_col,
                    )
                    st.session_state.le_col = le_col

                operations["encoding"] = {"ohe": ohe_col, "le": le_col}

        with st.expander("Normalization"):

            if st.toggle("Normalize?"):
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

        # ==== Spit method
        split_options = ["Same as uploaded", "Train/Test (80/20)"]
        split_options = (
            split_options if len(datasets_idx) == 1 else split_options + ["One dataset"]
        )
        split = st.segmented_control(
            "Split method",
            split_options,
            selection_mode="single",
            default="Same as uploaded",
        )

        if split != "Same as uploaded":
            datasets_idx = [sum(datasets_idx)]
            datasets_name = datasets_name[:1]
            if split == "Train/Test (80/20)":
                datasets_idx = [
                    round(datasets_idx[0] * 0.8),
                    round(datasets_idx[0] * 0.2),
                ]
                datasets_name = [
                    f"train_{datasets_name[0]}",
                    f"test_{datasets_name[0]}",
                ]

    # == Import config
    else:
        config = st.text_area("Configuration")
        if config:
            operations = json.loads(config)
    # with st.expander("Dimension reduction"):
    #     nb_dim = st.number_input('Number of dimensions to keep')
    #     min_var = st.number_input('Minimum variance')
    #     operations['dimension_reduction'] = [nb_dim, min_var]
    #
    # with st.expander("Undersampling"):
    #     pass

    c5, c6 = st.columns(2)
    if c5.button("Run", type="primary"):
        preprocessed_dataset = sumup_operations(operations, dataset_df, datasets_idx)

    if "enc_dataset" in st.session_state:
        ready = st.session_state["enc_dataset"]

        # == File to download
        dataset_downloader = DatasetDownloader(datasets_idx, ready, datasets_name)

        c6.download_button(
            "Download preprocessed dataset(s)",
            dataset_downloader.download_file,
            file_name=dataset_downloader.file_name,
            mime=dataset_downloader.mime_type,
        )
