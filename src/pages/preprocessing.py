"""
Page to preprocess the dataset — improved UI version.

Plugins required:
    pip install streamlit-aggrid streamlit-extras
"""

import json
import sys

sys.path.append("../..")

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_extras.metric_cards import style_metric_cards

from helpers.utils import load_data, sumup_operations
from helpers.dataset_tools import DatasetDownloader


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeX — Preprocessing",
    page_icon="⚙️",
)

# Small CSS tweaks: tighten tab font, give metrics a card feel
st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { padding: 6px 18px; border-radius: 6px 6px 0 0; }
        div[data-testid="metric-container"] { background:#f8f9fb; border-radius:8px; padding:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar — global settings ────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Global settings")

    view = st.radio(
        "Configuration mode",
        options=["Step by step", "Import JSON config"],
        help="Build the pipeline interactively, or paste a saved JSON config.",
    )

    st.divider()

    shuffle = st.checkbox("Shuffle dataset before processing")

    st.divider()

    split_method = st.selectbox(
        "Output split",
        ["Same as uploaded", "Train / Test (80 / 20)", "One merged dataset"],
    )

    st.divider()
    st.caption("DeX · Preprocessing v2")


# ─── Main area ────────────────────────────────────────────────────────────────

st.title("⚙️ DeX — Preprocessing")

# ─── Dataset upload ───────────────────────────────────────────────────────────

dataset_df = pd.DataFrame()

with st.expander("⬆️ Upload your dataset", expanded=True):
    datasets_csv = st.file_uploader(
        "Upload one or more CSV files (train / test / val)",
        type="csv",
        accept_multiple_files=True,
    )

    try:
        datasets_name = [d.name for d in datasets_csv] if len(datasets_csv) > 1 else [datasets_csv[0].name]
        datasets_df_list = [load_data(f) for f in datasets_csv]
        datasets_idx = [len(d) for d in datasets_df_list]
        dataset_df = pd.concat(datasets_df_list, ignore_index=True)
        st.success(f"✅ {len(datasets_csv)} file(s) loaded — {len(dataset_df):,} rows × {len(dataset_df.columns)} columns")
    except (ValueError, IndexError):
        st.info("Upload at least one CSV file to get started.")


# ─── Only render the rest once data is loaded ─────────────────────────────────

if dataset_df.empty:
    st.stop()

if shuffle:
    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

operations: dict = {}

# ── Dataset summary metrics ───────────────────────────────────────────────────

st.subheader("📊 Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(dataset_df):,}")
m2.metric("Columns", len(dataset_df.columns))
m3.metric("Missing values", int(dataset_df.isna().sum().sum()))
m4.metric("Duplicate rows", int(dataset_df.duplicated().sum()))
# style_metric_cards()  # streamlit-extras: adds subtle card styling

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP MODE
# ═══════════════════════════════════════════════════════════════════════════════

if view == "Step by step":

    st.subheader("🔧 Preprocessing pipeline")

    tab_drop, tab_replace, tab_encode, tab_norm, tab_reduce = st.tabs(
        ["1 · Drop", "2 · Replace", "3 · Encode", "4 · Normalize", "5 · Reduce"]
    )

    # ── Tab 1 — Drop values ───────────────────────────────────────────────────

    with tab_drop:
        st.markdown("##### Remove rows or columns")

        if st.toggle("Remove invalid values (NaN, inf …)"):
            operations["remove_inv_val"] = {}
            st.caption("✔ Invalid rows will be dropped on Run.")

        col_to_drop = st.multiselect(
            "Columns to drop entirely",
            options=dataset_df.columns.tolist(),
        )
        if col_to_drop:
            dataset_df = dataset_df.drop(columns=col_to_drop)
            st.info(f"Preview updated — {len(col_to_drop)} column(s) removed. {len(dataset_df.columns)} remaining.")

        # Live mini-preview
        st.dataframe(dataset_df.head(5), use_container_width=True)

    # ── Tab 2 — Replace values ────────────────────────────────────────────────

    with tab_replace:
        replace_col1, replace_col2 = st.columns([0.4, 0.6])
        replace_col1.markdown("##### Map old values to new ones")

        if replace_col2.toggle("", key="enable_replace"):
            placeholder = """{
    "column_name": [
        {
            "former_values": ["old_a", "old_b"],
            "new_value": "new_val"
        }
    ]
}"""
            mapping_txt = st.text_area(
                "Replacement mapping (JSON)",
                value=placeholder,
                height=200,
            )
            try:
                parsed = json.loads(mapping_txt)
                operations["replace_val"] = {"mappings": parsed}
                st.success("✔ Valid JSON")
            except json.JSONDecodeError:
                st.warning("⚠️ Invalid JSON — fix before running.")

    # ── Tab 3 — Encoding ──────────────────────────────────────────────────────

    with tab_encode:
        enc_col1, enc_col2 = st.columns(2)
        enc_col1.markdown("##### One-Hot Encoding & Label Encoding")

        if enc_col2.toggle("", key="enable_encoding"):

            # Initialize session state keys for encoding
            for key in ["ohe_col", "le_col_excluded", "le_col_included"]:
                if key not in st.session_state:
                    st.session_state[key] = []

            all_columns = dataset_df.columns.tolist()

            # OHE — filter stale defaults against current columns
            ohe_col = st.multiselect(
                "Columns → One Hot Encoding",
                options=all_columns,
                default=[c for c in st.session_state.ohe_col if c in all_columns],
                key="ohe_col",  # Streamlit owns this value; session state auto-updated
            )

            remaining_columns = [c for c in all_columns if c not in ohe_col]

            c1, c2 = st.columns(2)
            use_label = c1.checkbox("Label-encode remaining columns", value=True)

            if use_label:
                # Save the EXCLUDED list — it's smaller and more stable across reruns
                c2.multiselect(
                    "Exclude from Label Encoding",
                    options=remaining_columns,
                    default=[c for c in st.session_state.le_col_excluded if c in remaining_columns],
                    key="le_col_excluded",  # Streamlit keeps this in sync automatically
                )
                le_col = [c for c in remaining_columns if c not in st.session_state.le_col_excluded]
            else:
                c2.multiselect(
                    "Columns → Label Encoding",
                    options=remaining_columns,
                    default=[c for c in st.session_state.le_col_included if c in remaining_columns],
                    key="le_col_included",
                )
                le_col = st.session_state.le_col_included

            operations["encoding"] = {"ohe": ohe_col, "le": le_col}

            # Summary badges
            if ohe_col:
                st.caption(f"OHE: {len(ohe_col)} column(s) encoded.")
            if le_col:
                st.caption(f"LE: {len(le_col)} column(s) encoded.`")

    # ── Tab 4 — Normalization ─────────────────────────────────────────────────

    with tab_norm:
        norm_col1, norm_col2 = st.columns([0.45, 0.65])
        norm_col1.markdown("##### Scale numeric features")

        if norm_col2.toggle("", key="scale"):
            norm_method = st.selectbox("Method", ["Standardization", "MinMax"])

            c3, c4 = st.columns(2)

            if c3.checkbox("Apply to all numeric columns"):
                excluded_norm = c4.multiselect("Exclude columns", options=dataset_df.columns)
                norm_cols = [col for col in dataset_df.columns if col not in excluded_norm]
            else:
                norm_cols = c4.multiselect("Columns to normalize", options=dataset_df.columns)

            operations[norm_method.lower()] = {"col": norm_cols}

            if norm_cols:
                st.caption(f"{norm_method} will be applied to {len(norm_cols)} column(s).")

    # ── Tab 5 — Dimension reduction ───────────────────────────────────────────

    with tab_reduce:
        pca_col3, pca_col4 = st.columns([0.4, 0.6])
        pca_col3.markdown("##### PCA — reduce feature space")

        if pca_col4.toggle(""):
            pca_col1, pca_col2 = st.columns(2)
            method = pca_col1.radio(
                "Define target dimensions by",
                ["Minimum explained variance", "Fixed number of components"],
                horizontal=True,
            )

            col_to_drop_pca = pca_col2.multiselect(
                "Columns to exclude from PCA",
                options=dataset_df.columns.tolist(),
                help="PCA will be applied to the remaining numeric columns after encoding and normalization steps."
            )
            st.divider()

            if method == "Minimum explained variance":
                min_var = st.slider(
                    "Minimum variance to retain",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.80,
                    step=0.01,
                    format="%.2f",
                )
                operations["pca"] = {"min_variance": min_var, "cols_to_drop": col_to_drop_pca}
                st.caption(f"PCA will keep enough components to explain ≥ {min_var:.0%} of variance.")
            else:
                nb_dim = st.number_input(
                    "Number of components to keep",
                    step=1,
                    min_value=1,
                    max_value=len(dataset_df.columns) - 1,
                    value=min(len(dataset_df.columns) - 1, 10),
                )
                operations["pca"] = {"nb_dim": nb_dim}
                st.caption(f"PCA will reduce to {nb_dim} component(s).")


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT JSON CONFIG MODE
# ═══════════════════════════════════════════════════════════════════════════════

else:
    st.subheader("📥 Import configuration")
    config_txt = st.text_area("Paste your JSON pipeline config here", height=250)

    if config_txt:
        try:
            operations = json.loads(config_txt)
            st.success(f"✔ Config loaded — {len(operations)} operation(s) detected: `{'`, `'.join(operations.keys())}`")
        except json.JSONDecodeError:
            st.warning("⚠️ Invalid JSON — please check your config.")


# ─── Split configuration (driven by sidebar selection) ────────────────────────

# Resolve sidebar split choice into datasets_idx / datasets_name
try:
    if split_method == "One merged dataset" or split_method == "Same as uploaded":
        if split_method == "One merged dataset":
            datasets_idx = [sum(datasets_idx)]
            datasets_name = datasets_name[:1]
        # else: keep as-is

    elif split_method == "Train / Test (80 / 20)":
        total = sum(datasets_idx)
        datasets_idx = [round(total * 0.8), round(total * 0.2)]
        datasets_name = [f"train_{datasets_name[0]}", f"test_{datasets_name[0]}"]
except NameError:
    pass  # datasets_idx not yet defined (no file uploaded), handled by st.stop() above


# ─── Run pipeline ─────────────────────────────────────────────────────────────

st.divider()
c_run, c_dl = st.columns([1, 2])

if c_run.button("▶  Run preprocessing", type="primary", use_container_width=True):
    if not operations:
        st.warning("No operations configured. Add at least one step before running.")
    else:
        with st.status("Running preprocessing pipeline…", expanded=True) as status:
            for op_name in operations:
                st.write(f"• Applying **{op_name}**…")

            preprocessed = sumup_operations(operations, dataset_df, datasets_idx)
            status.update(label="✅ Preprocessing complete!", state="complete", expanded=False)


# ─── Download button ──────────────────────────────────────────────────────────

if "enc_dataset" in st.session_state:
    ready = st.session_state["enc_dataset"]
    dataset_downloader = DatasetDownloader(datasets_idx, ready, datasets_name)

    c_dl.download_button(
        label="⬇️  Download preprocessed dataset(s)",
        data=dataset_downloader.download_file,
        file_name=dataset_downloader.file_name,
        mime=dataset_downloader.mime_type,
        use_container_width=True,
    )