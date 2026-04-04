"""
DIAGNOSTIC TEST - Check if HTML renders properly
"""
import streamlit as st

st.set_page_config(page_title="Test", layout="wide")

st.markdown("## Test 1: Direct HTML (should NOT render)")
test1 = """<div style="background:red;color:white;padding:10px;">BROKEN - Raw HTML</div>"""
st.write(test1)

st.markdown("## Test 2: HTML with unsafe_allow_html (should render)")
st.markdown("""<div style="background:green;color:white;padding:10px;">FIXED - Rendered HTML</div>""", unsafe_allow_html=True)

st.markdown("## Test 3: Pipeline Strip Test")
items = ""
for i in range(3):
    items += f"<div>Item {i}</div>"
pipeline_html = f"<div>{items}</div>"
st.markdown(pipeline_html, unsafe_allow_html=True)
