import os
import streamlit as st

class Config:
    """
    Central configuration file for Gemini models.
    No API keys should be hardcoded here.
    Keys are safely loaded from Streamlit Secrets OR Environment Variables.
    """

    # --- Model Names ---
    PRIMARY_MODEL_NAME = "gemini-2.5-flash"
    FALLBACK_MODEL_NAME = "gemini-2.5-pro"

    # --- Load API key safely ---
    @property
    def GEMINI_API_KEY(self):
        # 1. Try Streamlit Secrets
        try:
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

        # 2. Try environment variable
        return os.getenv("GEMINI_API_KEY", None)


config = Config()


def configure_genai_if_available(genai_module, cfg: Config) -> bool:
    """
    Attempt to configure google.generativeai with the available API key.
    Returns:
        True  → configuration successful
        False → API key missing or failed
    """
    api_key = cfg.GEMINI_API_KEY

    if not api_key:
        return False

    try:
        genai_module.configure(api_key=api_key)
        return True
    except Exception:
        return False
