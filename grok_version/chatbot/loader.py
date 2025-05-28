import streamlit as st
import time
from typing import Any, Dict

with st.spinner("Loading..."):
    time.sleep(7)  # Simulate loading time
    # Simulate a long-running process
    st.write("result")