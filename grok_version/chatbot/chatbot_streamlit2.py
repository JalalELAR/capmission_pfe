import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import random

# Initialiser ChromaDB
client = chromadb.PersistentClient(path="../chroma_db3")
collection_name = "groupes_vectorises3"
collection = client.get_collection(name=collection_name)

# Vérifier si la collection est vide
if collection.count() == 0:
    st.error("Erreur : La collection ChromaDB est vide. Veuillez exécuter la vectorisation d'abord.")
    st.stop()

# Initialiser le modèle pour les embeddings (si nécessaire)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Récupérer toutes les valeurs uniques depuis ChromaDB
all_groups = collection.get(include=["metadatas", "documents"])
schools = set()
levels = set()
subjects = set()
centers = set()
teachers = set()
for metadata in all_groups['metadatas']:
    try:
        schools.add(metadata['ecole'].split(", ")[0])
        levels.add(metadata['niveau'])
        subjects.add(metadata['matiere'])
        centers.add(metadata['centre'])
        teachers.add(metadata['teacher'])
    except (KeyError, IndexError):
        continue

schools_list = sorted(list(schools))
levels_list = sorted(list(levels))
subjects_list = sorted(list(subjects))
centers_list = sorted(list(centers))
teachers_list = sorted(list(teachers))

# Fonction pour adapter les entrées utilisateur avec fuzzy matching
def match_value(user_input, valid_values):
    if not user_input or not valid_values:
        return user_input
    result = process.extractOne(user_input, valid_values)
    if result is None:
        return user_input
    best_match, score = result
    return best_match if score > 80 else user_input

# Fonction pour compter le nombre d'étudiants de la même école dans un groupe
def count_school_students(group_schools, user_school):
    return sum(1 for school in group_schools if school == user_school)

# Fonction principale pour obtenir les recommandations par matière
def get_recommendations(student_name, user_level, user_subjects, user_teachers, user_school, user_center):
    output = []
    matched_level = match_value(user_level, levels)
    matched_subjects = [match_value(subj.strip(), subjects) for subj in user_subjects.split(",")]
    matched_teachers = [match_value(teacher.strip(), teachers) for teacher in user_teachers.split(",")] if user_teachers else [None] * len(matched_subjects)
    matched_school = match_value(user_school, schools)
    matched_center = match_value(user_center, centers) if user_center else None
    
    if len(matched_teachers) != len(matched_subjects):
        output.append(f"<b>Attention</b> : Le nombre de professeurs ({len(matched_teachers)}) ne correspond pas au nombre de matières ({len(matched_subjects)}). Je vais associer les professeurs disponibles dans l'ordre.")
        matched_teachers.extend([None] * (len(matched_subjects) - len(matched_teachers)))

    output.append(f"<b>Niveau adapté</b>: {matched_level}")
    output.append("<b>Matières adaptées</b>:")
    for subj, matched_subj in zip(user_subjects.split(","), matched_subjects):
        output.append(f"- {subj.strip()} -> {matched_subj}")
    
    output.append("<b>Professeurs actuels</b>:")
    if user_teachers:
        for teacher, matched_teacher in zip(user_teachers.split(","), matched_teachers):
            output.append(f"- {teacher.strip()} -> {matched_teacher if matched_teacher else 'Non spécifié'}")
    else:
        output.append("- Aucun professeur spécifié")
    
    output.append(f"<b>École adaptée</b> : {matched_school}")
    if user_center:
        output.append(f"<b>Centre souhaité</b> : {matched_center}")
    else:
        output.append("<b>Centre souhaité</b> : Non spécifié")

    all_recommendations = {}
    for matched_subject, matched_teacher in zip(matched_subjects, matched_teachers):
        # Récupérer tous les groupes correspondant au niveau et à la matière
        all_groups_data = collection.get(include=["metadatas", "documents"])
        groups = []
        for metadata, document in zip(all_groups_data['metadatas'], all_groups_data['documents']):
            if metadata['niveau'] == matched_level and metadata['matiere'] == matched_subject:
                group_schools = metadata['ecole'].split(", ")
                groups.append({
                    "id_cours": metadata['id_cours'],
                    "name_cours": metadata['name_cours'],
                    "num_students": int(metadata['num_students']),
                    "description": document,
                    "centre": metadata['centre'],
                    "teacher": metadata['teacher'],
                    "schools": group_schools,
                    "date_debut": metadata['date_debut'],
                    "date_fin": metadata['date_fin'],
                    "heure_debut": metadata['heure_debut'],
                    "heure_fin": metadata['heure_fin'],
                    "jour": metadata['jour'],
                    "niveau": metadata['niveau'],
                    "matiere": metadata['matiere']
                })

        if not groups:
            output.append(f"Aucun groupe trouvé pour {matched_subject} au niveau {matched_level}.")
            continue

        # Regrouper les groupes identiques (même id_cours)
        grouped_groups = {}
        for group in groups:
            key = group['id_cours']
            if key not in grouped_groups:
                grouped_groups[key] = {
                    "group": group,
                    "centres": [group['centre']],
                    "total_students": group['num_students'],
                    "by_centre": {group['centre']: {
                        "num_students": group['num_students'],
                        "description": group['description'],
                        "schools": group['schools']
                    }}
                }
            else:
                grouped_groups[key]['centres'].append(group['centre'])
                grouped_groups[key]['total_students'] += group['num_students']
                grouped_groups[key]['by_centre'][group['centre']] = {
                    "num_students": group['num_students'],
                    "description": group['description'],
                    "schools": group['schools']
                }

        # Convertir les groupes regroupés en une liste
        consolidated_groups = []
        for key, data in grouped_groups.items():
            group = data['group'].copy()
            # Toujours utiliser le total des étudiants
            group['num_students'] = data['total_students']
            # Stocker tous les centres pour les priorités
            group['centres'] = list(set(data['centres']))
            # Vérifier si le groupe est partagé entre plusieurs centres
            is_shared = len(group['centres']) > 1
            # Si le groupe est partagé, toujours utiliser les données combinées
            if is_shared:
                group['display_centres'] = group['centres']
                combined_description = ", ".join(
                    centre_data['description'] for centre_data in data['by_centre'].values()
                )
                combined_schools = []
                for centre_data in data['by_centre'].values():
                    combined_schools.extend(centre_data['schools'])
                group['description'] = combined_description
                group['schools'] = combined_schools
            else:
                # Si le groupe n'est pas partagé, appliquer la logique du centre spécifié
                if matched_center and matched_center in data['by_centre']:
                    group['display_centres'] = [matched_center]
                    group['description'] = data['by_centre'][matched_center]['description']
                    group['schools'] = data['by_centre'][matched_center]['schools']
                else:
                    # Si aucun centre n'est spécifié, utiliser les données combinées (ici un seul centre)
                    group['display_centres'] = group['centres']
                    combined_description = ", ".join(
                        centre_data['description'] for centre_data in data['by_centre'].values()
                    )
                    combined_schools = []
                    for centre_data in data['by_centre'].values():
                        combined_schools.extend(centre_data['schools'])
                    group['description'] = combined_description
                    group['schools'] = combined_schools
            consolidated_groups.append(group)

        # Définir les priorités
        selected_groups = []
        selected_ids = set()

        def add_groups(new_groups):
            for group in new_groups:
                if group['id_cours'] not in selected_ids and len(selected_groups) < 3:
                    selected_groups.append(group)
                    selected_ids.add(group['id_cours'])

        # Priorité 1 : Même professeur, même centre, même école
        if matched_teacher and matched_center:
            priority_1 = [g for g in consolidated_groups if matched_teacher == g['teacher'] and matched_center in g['centres'] and matched_school in g['schools']]
            # Trier par nombre d'étudiants de la même école (décroissant)
            priority_1.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_1)

        # Priorité 2 : Même professeur, même centre (ignorer école)
        if matched_teacher and matched_center and len(selected_groups) < 3:
            priority_2 = [g for g in consolidated_groups if matched_teacher == g['teacher'] and matched_center in g['centres']]
            add_groups(priority_2)

        # Priorité 3 : Même centre, même école (ignorer professeur)
        if matched_center and len(selected_groups) < 3:
            priority_3 = [g for g in consolidated_groups if matched_center in g['centres'] and matched_school in g['schools']]
            # Trier par nombre d'étudiants de la même école (décroissant)
            priority_3.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_3)

        # Priorité 4 : Même professeur, même école (ignorer centre)
        if matched_teacher and len(selected_groups) < 3:
            priority_4 = [g for g in consolidated_groups if matched_teacher == g['teacher'] and matched_school in g['schools']]
            # Trier par nombre d'étudiants de la même école (décroissant)
            priority_4.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_4)

        # Priorité 5 : Même école (ignorer professeur et centre)
        if len(selected_groups) < 3:
            priority_5 = [g for g in consolidated_groups if matched_school in g['schools']]
            # Trier par nombre d'étudiants de la même école (décroissant)
            priority_5.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_5)

        # Priorité 6 : Même centre (ignorer professeur et école)
        if matched_center and len(selected_groups) < 3:
            priority_6 = [g for g in consolidated_groups if matched_center in g['centres']]
            add_groups(priority_6)

        # Priorité 7 : Groupes aléatoires
        if len(selected_groups) < 3:
            remaining_groups = [g for g in consolidated_groups if g['id_cours'] not in selected_ids]
            random.shuffle(remaining_groups)
            add_groups(remaining_groups)

        if len(selected_groups) < 3:
            output.append(f"Attention : Seulement {len(selected_groups)} groupe(s) trouvé(s) pour {matched_subject}.")

        # Préparer les recommandations
        recommendations = []
        for i, group in enumerate(selected_groups[:3], 1):
            description_parts = group['description'].split(", ")
            formatted_description = "<br>".join(description_parts)
            centres_display = " et ".join(group['display_centres'])
            recommendation = (
                f"<h4>Groupe {i} ({matched_subject})</h4>"
                f"<b>ID:</b> {group['id_cours']}<br>"
                f"<b>Nom:</b> {group['name_cours']}<br>"
                f"<b>Nombre d'étudiants:</b> {group['num_students']}<br>"
                f"<b>Professeur:</b> {group['teacher']}<br>"
                f"<b>Centre(s):</b> {centres_display}<br>"
                f"<b>Date de début:</b> {group['date_debut']}<br>"
                f"<b>Date de fin:</b> {group['date_fin']}<br>"
                f"<b>Heure de début:</b> {group['heure_debut']}<br>"
                f"<b>Heure de fin:</b> {group['heure_fin']}<br>"
                f"<b>Jour:</b> {group['jour']}<br>"
                f"<b>Détails:</b><br>{formatted_description}"
            )
            recommendations.append(recommendation)
        
        all_recommendations[matched_subject] = recommendations

    output.append(f"<b>Les groupes recommandés pour l'étudiant</b> {student_name} :")
    return output, all_recommendations

# Styles CSS pour l'affichage
st.markdown("""
    <style>
    .bot-message { 
        background-color: #e0f7fa; 
        border-radius: 10px; 
        padding: 10px; 
        margin: 10px 0; 
        width: 70%; 
        float: left; 
        color: #00695c; 
        font-family: Arial, sans-serif; 
    }
    .user-message { 
        background-color: #c8e6c9; 
        border-radius: 10px; 
        padding: 10px; 
        margin: 10px 0; 
        width: 70%; 
        float: right; 
        color: #2e7d32; 
        font-family: Arial, sans-serif; 
        text-align: right; 
    }
    .container { 
        overflow: hidden; 
    }
    h4 { color: #00796b; margin-bottom: 5px; }
    b { color: #004d40; }
    div[data-testid="stTextInput"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stTextInput"] > div {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Interface Streamlit
st.title("Chatbot de Recommandation de Groupes")
st.write("Je vais vous poser des questions une par une. Répondez dans le champ ci-dessous et appuyez sur Entrée pour valider.")

# Initialiser l'état de la session
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.messages = [("<div class='bot-message'>Bonjour ! Je vais vous aider à trouver des groupes recommandés.</div>", True)]
    st.session_state.responses = {}
    st.session_state.current_input = ""
    st.session_state.submitted = False
    st.session_state.input_counter = 0

# Afficher les messages précédents
st.markdown("<div class='container'>", unsafe_allow_html=True)
for message, is_bot in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Liste des questions et placeholders
questions = [
    "Quel est le nom de l'étudiant ?",
    "Quel est le niveau de l'étudiant (ex. Terminale, Première) ?",
    "Quelles sont les matières qui intéressent l'étudiant (séparez par des virgules, ex. Français, Maths) ?",
    "Quels sont les professeurs actuels de l'étudiant pour ces matières (séparez par des virgules dans le même ordre, ex. Ahmed Belkadi, Sara Lahlou) ? Laissez vide si inconnu.",
    "Quelle est l'école de l'étudiant (obligatoire, ex. Al Khawarezmi) ?",
    "Quel est le centre souhaité par l'étudiant (ex. Centre A) ? Laissez vide si pas de préférence."
]

placeholders = [
    "Ex: Ahmed",
    "Ex: BL - 1bac sc maths",
    "Ex: Français, Maths",
    "Ex: Ahmed Belkadi, Sara Lahlou (ou laissez vide)",
    "Ex: Massignon Bouskoura",
    "Ex: Centre A (ou laissez vide)"
]

# Fonction pour gérer la soumission par Entrée
def handle_input_submission(step, response):
    if step == 1:  # Nom de l'étudiant (obligatoire)
        if response.strip():  # Vérifier que la réponse n'est pas vide après suppression des espaces
            st.session_state.responses['student_name'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quel est le niveau de l'étudiant (ex. Terminale, Première) ?</div>", True))
            st.session_state.step = 2
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 2:  # Niveau (obligatoire)
        if response.strip():
            st.session_state.responses['user_level'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quelles sont les matières qui intéressent l'étudiant (séparez par des virgules, ex. Français, Maths) ?</div>", True))
            st.session_state.step = 3
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 3:  # Matières (obligatoire)
        if response.strip():
            st.session_state.responses['user_subjects'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quels sont les professeurs actuels de l'étudiant pour ces matières (séparez par des virgules dans le même ordre, ex. Ahmed Belkadi, Sara Lahlou) ? Laissez vide si inconnu.</div>", True))
            st.session_state.step = 4
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 4:  # Professeurs (facultatif)
        # Accepter une entrée vide
        st.session_state.responses['user_teachers'] = response.strip() if response.strip() else None
        st.session_state.messages.append((f"<div class='user-message'>{response if response.strip() else 'Aucun spécifié'}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quelle est l'école de l'étudiant (obligatoire, ex. Al Khawarezmi) ?</div>", True))
        st.session_state.step = 5
    elif step == 5:  # École (obligatoire)
        if response.strip():
            st.session_state.responses['user_school'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quel est le centre souhaité par l'étudiant (ex. Centre A) ? Laissez vide si pas de préférence.</div>", True))
            st.session_state.step = 6
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 6:  # Centre (facultatif)
        # Accepter une entrée vide
        st.session_state.responses['user_center'] = response.strip() if response.strip() else None
        st.session_state.messages.append((f"<div class='user-message'>{response if response.strip() else 'Aucun spécifié'}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Merci ! Je recherche les groupes recommandés...</div>", True))
        with st.spinner("Recherche en cours..."):
            output, all_recommendations = get_recommendations(
                st.session_state.responses['student_name'],
                st.session_state.responses['user_level'],
                st.session_state.responses['user_subjects'],
                st.session_state.responses['user_teachers'],
                st.session_state.responses['user_school'],
                st.session_state.responses['user_center']
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
        st.session_state.step = 7

# Logique conversationnelle sans st.form
if st.session_state.step == 0:
    st.session_state.messages.append(("<div class='bot-message'>Quel est le nom de l'étudiant ?</div>", True))
    st.session_state.step = 1
    st.session_state.current_input = ""
    st.session_state.submitted = False
    st.session_state.input_counter = 0
    st.rerun()

elif st.session_state.step in [1, 2, 3, 4, 5, 6]:
    input_key = f"input_step_{st.session_state.step}_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder=placeholders[st.session_state.step - 1])
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        # Pour les champs obligatoires (étapes 1, 2, 3, 5), vérifier que la réponse n'est pas vide
        if st.session_state.step in [1, 2, 3, 5]:
            if st.session_state[input_key].strip():  # Vérifier que la réponse n'est pas vide
                handle_input_submission(st.session_state.step, st.session_state[input_key])
                st.session_state.input_counter += 1
                st.session_state.current_input = ""
                st.session_state.submitted = False
                st.rerun()
            else:
                st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
                st.session_state.input_counter += 1
                st.session_state.submitted = False
                st.rerun()
        else:  # Pour les champs facultatifs (étapes 4, 6), accepter une entrée vide
            handle_input_submission(st.session_state.step, st.session_state[input_key])
            st.session_state.input_counter += 1
            st.session_state.current_input = ""
            st.session_state.submitted = False
            st.rerun()

elif st.session_state.step == 7:
    choice = st.radio("", ["Oui", "Non"], index=None, key="restart_choice")
    if choice == "Oui":
        st.session_state.clear()
        st.session_state.step = 0
        st.rerun()
    elif choice == "Non":
        st.session_state.step = 8