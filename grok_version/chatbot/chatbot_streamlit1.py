import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process

# Initialiser ChromaDB
client = chromadb.PersistentClient(path="../chroma_db2")
collection_name = "groupes_vectorises2"
collection = client.get_collection(name=collection_name)

# Initialiser le modèle pour les embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Récupérer toutes les écoles, niveaux et matières uniques depuis ChromaDB
all_groups = collection.get(include=["metadatas", "documents"])
schools = set()
levels = set()
subjects = set()
for metadata, description in zip(all_groups['metadatas'], all_groups['documents']):
    try:
        school_section = description.split("Écoles: ")[1]
        schools.update(school_section.split(", "))
        level = description.split("Niveau: ")[1].split(", ")[0]
        levels.add(level)
        subject = description.split("Matière: ")[1].split(", ")[0]
        subjects.add(subject)
    except IndexError:
        continue

schools_list = sorted(list(schools))
levels_list = sorted(list(levels))
subjects_list = sorted(list(subjects))

# Fonction pour adapter les entrées utilisateur avec fuzzy matching
def match_value(user_input, valid_values):
    if not user_input:
        return None
    best_match, score = process.extractOne(user_input, valid_values)
    return best_match if score > 80 else user_input

# Fonction principale pour obtenir les recommandations par matière
def get_recommendations(student_name, user_level, user_subjects, user_school):
    output = []
    matched_level = match_value(user_level, levels)
    matched_subjects = [match_value(subj.strip(), subjects) for subj in user_subjects.split(",")]
    matched_school = match_value(user_school, schools)
    
    if matched_level:
        output.append(f"<b>Niveau adapté</b>: {matched_level}")
    else:
        output.append(f"Niveau '{user_level}' non reconnu, je vais chercher sans filtre strict.")
        matched_level = user_level
    
    output.append("<b>Matières adaptées</b>:")
    for subj, matched_subj in zip(user_subjects.split(","), matched_subjects):
        if matched_subj:
            output.append(f"- {subj.strip()} -> {matched_subj}")
        else:
            output.append(f"- {subj.strip()} -> non reconnu, je vais chercher sans filtre strict.")
            matched_subjects[matched_subjects.index(None)] = subj.strip()
    
    if matched_school:
        output.append(f"<b>École adaptée</b> : {matched_school}")
    else:
        output.append(f"École '{user_school}' non reconnue, je vais chercher sans filtre strict.")
        matched_school = user_school

    all_recommendations = {}
    for matched_subject in matched_subjects:
        query_description = f"<b>Niveau:</b> {matched_level}, <b>Matière:</b> {matched_subject}, <b>École:</b> {matched_school}"
        query_embedding = model.encode([query_description])[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["metadatas", "documents", "distances"]
        )

        matching_groups_dict = {}
        if results and 'metadatas' in results and results['metadatas']:
            for metadata, description, distance in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
                group_level = description.split("Niveau: ")[1].split(", ")[0]
                group_subject = description.split("Matière: ")[1].split(", ")[0]
                group_schools = description.split("Écoles: ")[1].split(", ")
                
                if matched_level in group_level and group_subject == matched_subject:
                    school_match = matched_school in group_schools if matched_school else False
                    group_id = metadata['id_cours']
                    if group_id not in matching_groups_dict or distance < matching_groups_dict[group_id]['distance']:
                        matching_groups_dict[group_id] = {
                            "id_cours": group_id,
                            "name_cours": metadata['name_cours'],
                            "num_students": metadata['num_students'],
                            "description": description,
                            "distance": distance,
                            "school_match": school_match,
                            "num_schools": len(group_schools)
                        }

        matching_groups = list(matching_groups_dict.values())

        if matching_groups:
            matching_groups.sort(key=lambda x: (not x['school_match'], x['distance'], -x['num_schools']))
            top_groups = matching_groups[:3]
        else:
            output.append(f"Aucun groupe trouvé pour {matched_subject} dans la requête initiale, recherche dans toute la base.")
            top_groups = []

        if len(top_groups) < 3:
            all_groups = collection.get(include=["metadatas", "documents"])
            if all_groups['metadatas']:
                additional_groups = {}
                for metadata, description in zip(all_groups['metadatas'], all_groups['documents']):
                    group_level = description.split("Niveau: ")[1].split(", ")[0]
                    group_subject = description.split("Matière: ")[1].split(", ")[0]
                    group_schools = description.split("Écoles: ")[1].split(", ")
                    group_id = metadata['id_cours']
                    
                    if (matched_level in group_level and 
                        group_subject == matched_subject and 
                        group_id not in [g['id_cours'] for g in top_groups]):
                        school_match = matched_school in group_schools if matched_school else False
                        additional_groups[group_id] = {
                            "id_cours": group_id,
                            "name_cours": metadata['name_cours'],
                            "num_students": metadata['num_students'],
                            "description": description,
                            "distance": float('inf'),
                            "school_match": school_match,
                            "num_schools": len(group_schools)
                        }
                additional_groups_list = list(additional_groups.values())
                additional_groups_list.sort(key=lambda x: (not x['school_match'], -x['num_schools']))
                top_groups.extend(additional_groups_list[:3 - len(top_groups)])
            else:
                output.append(f"La collection ChromaDB semble vide pour {matched_subject}.")

        output.append(f"<b>Nombre total de groupes sélectionnés pour {matched_subject}</b> : {len(top_groups)}")

        recommendations = []
        for i, group in enumerate(top_groups[:3], 1):
            description_parts = group['description'].split(", ")
            formatted_description = "<br>".join(description_parts)
            recommendation = (
                f"<h4>Groupe {i} ({matched_subject})</h4>"
                f"<b>ID:</b> {group['id_cours']}<br>"
                f"<b>Nom:</b> {group['name_cours']}<br>"
                f"<b>Nombre d'étudiants:</b> {group['num_students']}<br>"
                f"<b>Détails:</b><br>{formatted_description}"
            )
            recommendations.append(recommendation)
        
        all_recommendations[matched_subject] = recommendations

    output.append(f"<b>Les groupes recommandés pour l'étudiant</b> {student_name} :")
    return output, all_recommendations

# Styles CSS pour différencier les messages
st.markdown("""
    <style>
    .bot-message {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        width: 70%;
        float: left;
        color: #00695c;
        font-family: Arial, sans-serif;
    }
    .user-message {
        background-color: #c8e6c9;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        width: 70%;
        float: right;
        color: #2e7d32;
        font-family: Arial, sans-serif;
        text-align: right;
    }
    .container {
        overflow: hidden;
    }
    h4 {
        color: #00796b;
        margin-bottom: 5px;
    }
    b {
        color: #004d40;
    }
    </style>
""", unsafe_allow_html=True)

# Interface Streamlit en mode conversationnel
st.title("Chatbot de Recommandation de Groupes")
st.write("Je vais vous poser des questions une par une. Répondez dans le champ ci-dessous.")

# Initialiser l'état de la session
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.messages = [("<div class='bot-message'>Bonjour ! Je vais vous aider à trouver des groupes recommandés.</div>", True)]
    st.session_state.responses = {}

# Afficher les messages précédents
st.markdown("<div class='container'>", unsafe_allow_html=True)
for message, is_bot in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Logique conversationnelle avec clés uniques pour chaque étape
if st.session_state.step == 0:
    st.session_state.messages.append(("<div class='bot-message'>Quel est le nom de l'étudiant ?</div>", True))
    st.session_state.step = 1
    st.rerun()

elif st.session_state.step == 1:
    response = st.text_input("Votre réponse :", key="input_step1", placeholder="Ex: Ahmed")
    if response:
        st.session_state.responses['student_name'] = response
        st.session_state.messages.append((f"<div class='user-message'> {response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quel est le niveau de l'étudiant (ex. Terminale, Première) ?</div>", True))
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    response = st.text_input("Votre réponse :", key="input_step2", placeholder="Ex: BL - 1bac sc maths")
    if response:
        st.session_state.responses['user_level'] = response
        st.session_state.messages.append((f"<div class='user-message'> {response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quelles sont les matières qui intéressent l'étudiant (séparez par des virgules, ex. Français, Maths) ?</div>", True))
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    response = st.text_input("Votre réponse :", key="input_step3", placeholder="Ex: Français, Maths")
    if response:
        st.session_state.responses['user_subjects'] = response
        st.session_state.messages.append((f"<div class='user-message'> {response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quelle est l'école de l'étudiant (ex. Al Khawarezmi) ?</div>", True))
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    response = st.text_input("Votre réponse :", key="input_step4", placeholder="Ex: Charles Péguy")
    if response:
        st.session_state.responses['user_school'] = response
        st.session_state.messages.append((f"<div class='user-message'> {response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Merci ! Je recherche les groupes recommandés...</div>", True))
        with st.spinner("Recherche en cours..."):
            output, all_recommendations = get_recommendations(
                st.session_state.responses['student_name'],
                st.session_state.responses['user_level'],
                st.session_state.responses['user_subjects'],
                st.session_state.responses['user_school']
            )
        for line in output:
            st.session_state.messages.append((f"<div class='bot-message'>{line}</div>", True))
        if all_recommendations:
            for subject, recommendations in all_recommendations.items():
                st.session_state.messages.append((f"<div class='bot-message'><h3>Recommandations pour {subject}</h3></div>", True))
                for rec in recommendations:
                    st.session_state.messages.append((f"<div class='bot-message'>{rec}</div>", True))
        else:
            st.session_state.messages.append(("<div class='bot-message'>Désolé, aucune recommandation disponible.</div>", True))
        st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
        st.session_state.step = 5
        st.rerun()

elif st.session_state.step == 5:
    choice = st.radio("", ["Oui", "Non"], index=None, key="restart_choice")
    if choice == "Oui":
        st.session_state.clear()
        st.session_state.step = 0
        st.rerun()
    elif choice == "Non":
        st.session_state.step = 6  # Passer à une étape finale pour éviter de relancer la question

# À l'étape 6, les résultats restent affichés sans action supplémentaire