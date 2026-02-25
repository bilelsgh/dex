import streamlit as st

# pages
page_1 = st.Page("pages/viz.py", title="Analysis", icon="📊")
page_2 = st.Page("pages/preprocessing.py", title="Preprocessing", icon="⚙️")

pg = st.navigation([page_1, page_2])
st.set_page_config(page_title="DeX", page_icon="🚀")

pg.run()