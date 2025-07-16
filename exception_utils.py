# exception_utils.py

import streamlit as st
import traceback
from logging import Logger

def handle_exception(e: Exception, logger: Logger = None):
    error_message = f"❌ Exception: {str(e)}"
    st.error(error_message)
    st.expander("See details").write(traceback.format_exc())
    
    if logger:
        logger.error(error_message)
        logger.error(traceback.format_exc())
