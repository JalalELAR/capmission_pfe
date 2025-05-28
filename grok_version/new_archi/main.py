# chatbot_project/main.py

import os
import logging
import streamlit.web.cli as stcli
import sys
import subprocess

def launch_streamlit_ui():
    """Lance l'interface Streamlit du chatbot éducatif"""
    script_path = os.path.join("frontend", "ui_streamlit.py")
    try:
        print("🚀 Lancement de l'interface Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement de Streamlit : {e}")

if __name__ == "__main__":
    launch_streamlit_ui()
