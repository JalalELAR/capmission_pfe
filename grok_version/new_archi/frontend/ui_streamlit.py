# chatbot_project/frontend/ui_streamlit.py
import streamlit as st
from core.session_manager import load_user_state, save_user_state
from core.message_handler import handle_user_message
from pathlib import Path
import os


st.set_page_config(page_title="Chatbot éducatif", page_icon="📚", layout="centered")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
# Injecter le style CSS externe
STYLE_PATH = Path("frontend/style.css")
if STYLE_PATH.exists():
    with open(STYLE_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#st.title("🤖 Chatbot de soutien scolaire")

logo_path = os.path.join(grandparent_dir, "images", "logo.png")
try:    
    st.image(logo_path)
except FileNotFoundError:
    st.warning("Logo non trouvé. Veuillez placer 'logo.png' dans le répertoire du script.")
#st.title("Chatbot de Recommandation de Groupes")
profile_path = os.path.join(grandparent_dir, "images", "profile1.png")

with st.sidebar:
    st.image(profile_path, width=280, use_container_width=False, output_format="auto")
    st.markdown("<div class='profile-name'>ELARACHE Jalal</div>", unsafe_allow_html=True)
    st.header("Options")
    st.write("Bienvenue dans le Chatbot de Recommandation !")
    if st.button("Réinitialiser la conversation"):
        print("Réinitialiser la conversation")
    if st.button("Nouvelle discussion"):
        print("Nouvelle discussion")
    else:
        st.write("Aucun historique disponible.")
    st.write("---")
    st.write("**À propos**")
    st.write("Développé par IA pour optimiser la recherche de groupes éducatifs.")


# Initialiser session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Affichage des messages
st.markdown("<div class='messages-container'>", unsafe_allow_html=True)
for msg, sender in st.session_state.messages:
    class_name = "bot-message" if sender == "bot" else "user-message"
    st.markdown(f"<div class='{class_name}'>{msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Zone de saisie
user_input = st.chat_input("Votre message...")
if user_input:
    st.session_state.messages.append((user_input, "user"))
    user_state = load_user_state()
    bot_response, updated_state = handle_user_message(user_input, user_state)
    st.session_state.messages.append((bot_response, "bot"))
    save_user_state(updated_state)
    st.rerun()