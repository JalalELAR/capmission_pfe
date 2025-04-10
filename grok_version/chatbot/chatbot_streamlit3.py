import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import random

# Initialiser ChromaDB
client = chromadb.PersistentClient(path="../chroma_db5")
collection_name = "groupes_vectorises5"
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

# Fonction pour compter le nombre d'occurrences de l'école de l'utilisateur dans un groupe (sans normalisation)
def count_school_students(group_schools, user_school):
    return sum(1 for school in group_schools if school == user_school)

# Fonction principale pour obtenir les recommandations par matière
def get_recommendations(student_name, user_level, user_subjects, user_teachers, user_school, user_center):
    output = []
    matched_level = match_value(user_level, levels)
    matched_subjects = [match_value(subj.strip(), subjects) for subj in user_subjects.split(",")]
    matched_teachers = [match_value(teacher.strip(), teachers) for teacher in user_teachers.split(",")] if user_teachers else [None] * len(matched_subjects)
    matched_school = match_value(user_school, schools)  # Pas de normalisation
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
    all_groups_for_selection = {}  # Pour stocker les groupes sous forme de dictionnaires pour la sélection
    for matched_subject, matched_teacher in zip(matched_subjects, matched_teachers):
        # Récupérer tous les groupes correspondant au niveau et à la matière
        all_groups_data = collection.get(include=["metadatas", "documents"])
        groups = []
        for metadata, document in zip(all_groups_data['metadatas'], all_groups_data['documents']):
            if metadata['niveau'] == matched_level and metadata['matiere'] == matched_subject:
                group_schools = [school for school in metadata['ecole'].split(", ")]  # Pas de normalisation
                groups.append({
                    "id_cours": metadata['id_cours'],
                    "name_cours": metadata['name_cours'],
                    "num_students": int(metadata['num_students']),
                    "total_students": int(metadata['total_students']),
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

        # Ajouter la description formatée (liste des écoles)
        for group in groups:
            formatted_schools = "<br>".join(sorted(group['schools']))
            group['description'] = formatted_schools

        # Définir les priorités
        selected_groups = []
        selected_ids = set()

        def add_groups(new_groups):
            for group in new_groups:
                if group['id_cours'] not in selected_ids and len(selected_groups) < 3:
                    selected_groups.append(group)
                    selected_ids.add(group['id_cours'])

        # Si le centre est spécifié, utiliser la logique
        if matched_center:
            # Priorité 1 : Même professeur, même centre, même école
            priority_1_groups = []
            if matched_teacher:
                priority_1_groups = [g for g in groups if matched_teacher == g['teacher'] and matched_center == g['centre'] and matched_school in g['schools']]
                for g in priority_1_groups:
                    schools_list = g['schools']
                    count = count_school_students(g['schools'], matched_school)
                if len(priority_1_groups) >= 3:
                    priority_1_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                    for g in priority_1_groups:
                        count = count_school_students(g['schools'], matched_school)
                    add_groups(priority_1_groups[:3])
                else:
                    add_groups(priority_1_groups)

            # Priorité 2 : Même professeur, même centre (ignorer l'école)
            if matched_teacher and len(selected_groups) < 3:
                priority_2_groups = [g for g in groups if matched_teacher == g['teacher'] and matched_center == g['centre'] and g['id_cours'] not in selected_ids]
                add_groups(priority_2_groups)

            # Priorité 3 : Même centre, même école (ignorer le professeur)
            if len(selected_groups) < 3:
                priority_3_groups = [g for g in groups if matched_center == g['centre'] and matched_school in g['schools'] and g['id_cours'] not in selected_ids]
                priority_3_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(priority_3_groups)

            # Priorité 4 : Même centre (ignorer le professeur et l'école)
            if len(selected_groups) < 3:
                priority_4_groups = [g for g in groups if matched_center == g['centre'] and g['id_cours'] not in selected_ids]
                add_groups(priority_4_groups)

            # Priorité 5 : Même professeur (ignorer le centre)
            if matched_teacher and len(selected_groups) < 3:
                priority_5_groups = [g for g in groups if matched_teacher == g['teacher'] and g['id_cours'] not in selected_ids]
                priority_5_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(priority_5_groups)

            # Priorité 6 : Même école (ignorer le professeur et le centre)
            if len(selected_groups) < 3:
                priority_6_groups = [g for g in groups if matched_school in g['schools'] and g['id_cours'] not in selected_ids]
                priority_6_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(priority_6_groups)

            # Priorité 7 : Groupes aléatoires
            if len(selected_groups) < 3:
                remaining_groups = [g for g in groups if g['id_cours'] not in selected_ids]
                random.shuffle(remaining_groups)
                add_groups(remaining_groups)

        else:
            # Étape 1 : Vérifier l'existence du même professeur
            teacher_groups = []
            if matched_teacher:
                teacher_groups = [g for g in groups if matched_teacher == g['teacher']]

            # Étape 2 : Si on a 3 groupes ou plus avec le même professeur
            if len(teacher_groups) >= 3:
                teacher_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(teacher_groups[:3])

            else:
                has_school = any(matched_school in g['schools'] for g in teacher_groups)
                if has_school:
                    teacher_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(teacher_groups)

                if len(selected_groups) < 3:
                    school_groups = [g for g in groups if matched_school in g['schools'] and g['id_cours'] not in selected_ids]
                    school_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                    add_groups(school_groups)

                if len(selected_groups) < 3:
                    remaining_groups = [g for g in groups if g['id_cours'] not in selected_ids]
                    random.shuffle(remaining_groups)
                    add_groups(remaining_groups)

        if len(selected_groups) < 3:
            output.append(f"Attention : Seulement {len(selected_groups)} groupe(s) trouvé(s) pour {matched_subject}.")

        # Préparer les recommandations (format HTML pour l'affichage)
        recommendations = []
        # Préparer les groupes pour la sélection (format simple)
        groups_for_selection = []
        for i, group in enumerate(selected_groups[:3], 1):
            recommendation = (
                f"<h4>Groupe {i} ({matched_subject})</h4>"
                f"<b>ID:</b> {group['id_cours']}<br>"
                f"<b>Nom:</b> {group['name_cours']}<br>"
                f"<b>Nombre d'étudiants :</b> {group['num_students']}<br>"
                f"<b>Professeur:</b> {group['teacher']}<br>"
                f"<b>Centre:</b> {group['centre']}<br>"
                f"<b>Date de début:</b> {group['date_debut']}<br>"
                f"<b>Date de fin:</b> {group['date_fin']}<br>"
                f"<b>Heure de début:</b> {group['heure_debut']}<br>"
                f"<b>Heure de fin:</b> {group['heure_fin']}<br>"
                f"<b>Jour:</b> {group['jour']}<br>"
                f"<b>Écoles:</b><br>{group['description']}"
            )
            recommendations.append(recommendation)
            # Ajouter le groupe pour la sélection
            groups_for_selection.append({
                "id_cours": group['id_cours'],
                "name_cours": group['name_cours'],
                "display": f"Groupe {i} : {group['id_cours']} - {group['name_cours']}"
            })
        
        all_recommendations[matched_subject] = recommendations
        all_groups_for_selection[matched_subject] = groups_for_selection

    output.append(f"<b>Les groupes recommandés pour l'étudiant</b> {student_name} :")
    return output, all_recommendations, all_groups_for_selection, matched_subjects

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
    st.session_state.all_recommendations = {}
    st.session_state.all_groups_for_selection = {}
    st.session_state.matched_subjects = []
    st.session_state.selected_groups = {}
    st.session_state.current_subject_index = 0

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

# Fonction pour gérer la soumission par Entrée (étapes 1 à 6)
def handle_input_submission(step, response):
    if step == 1:  # Nom de l'étudiant (obligatoire)
        if response.strip():
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
        st.session_state.responses['user_center'] = response.strip() if response.strip() else None
        st.session_state.messages.append((f"<div class='user-message'>{response if response.strip() else 'Aucun spécifié'}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Merci ! Je recherche les groupes recommandés...</div>", True))
        with st.spinner("Recherche en cours..."):
            output, all_recommendations, all_groups_for_selection, matched_subjects = get_recommendations(
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
            st.session_state.step = 9  # Passer directement à l'étape finale
            st.rerun()
            return
        # Stocker les données pour la sélection
        st.session_state.all_recommendations = all_recommendations
        st.session_state.all_groups_for_selection = all_groups_for_selection
        st.session_state.matched_subjects = matched_subjects
        st.session_state.current_subject_index = 0
        # Vérifier s'il y a des groupes à sélectionner
        if not any(st.session_state.all_groups_for_selection.values()):
            st.session_state.messages.append(("<div class='bot-message'>Aucun groupe disponible pour sélection.</div>", True))
            st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
            st.session_state.step = 9
            st.rerun()
            return
        # Passer à l'étape de sélection des groupes
        current_subject = st.session_state.matched_subjects[st.session_state.current_subject_index]
        groups = st.session_state.all_groups_for_selection.get(current_subject, [])
        if not groups:
            # Si aucune recommandation pour cette matière, passer à la suivante
            st.session_state.current_subject_index += 1
            if st.session_state.current_subject_index < len(st.session_state.matched_subjects):
                current_subject = st.session_state.matched_subjects[st.session_state.current_subject_index]
                groups = st.session_state.all_groups_for_selection.get(current_subject, [])
                if not groups:
                    st.session_state.messages.append(("<div class='bot-message'>Aucun groupe disponible pour sélection. Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                    st.session_state.step = 9
                    st.rerun()
                    return
            else:
                st.session_state.messages.append(("<div class='bot-message'>Aucun groupe disponible pour sélection. Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                st.session_state.step = 9
                st.rerun()
                return
        # Afficher les groupes pour la matière actuelle
        groups_message = f"<div class='bot-message'>Voici les groupes recommandés pour {current_subject} :<br>"
        for i, group in enumerate(groups, 1):
            groups_message += f"{group['display']}<br>"
        groups_message += f"Veuillez entrer le numéro du groupe que vous souhaitez choisir (par exemple, 1 pour {groups[0]['display']}) :</div>"
        st.session_state.messages.append((groups_message, True))
        st.session_state.step = 8

# Fonction pour gérer la sélection des groupes (étape 8)
def handle_group_selection(response):
    current_subject = st.session_state.matched_subjects[st.session_state.current_subject_index]
    groups = st.session_state.all_groups_for_selection.get(current_subject, [])
    
    # Vérifier si la réponse est un numéro valide
    try:
        group_number = int(response.strip())
        if 1 <= group_number <= len(groups):
            selected_group = groups[group_number - 1]
            st.session_state.selected_groups[current_subject] = selected_group
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append((f"<div class='bot-message'>Vous avez choisi {selected_group['display']} pour {current_subject}.</div>", True))
            # Passer à la matière suivante
            st.session_state.current_subject_index += 1
            if st.session_state.current_subject_index < len(st.session_state.matched_subjects):
                # Demander pour la matière suivante
                current_subject = st.session_state.matched_subjects[st.session_state.current_subject_index]
                groups = st.session_state.all_groups_for_selection.get(current_subject, [])
                if not groups:
                    # Si aucune recommandation pour cette matière, passer à la suivante
                    st.session_state.current_subject_index += 1
                    if st.session_state.current_subject_index < len(st.session_state.matched_subjects):
                        current_subject = st.session_state.matched_subjects[st.session_state.current_subject_index]
                        groups = st.session_state.all_groups_for_selection.get(current_subject, [])
                        if not groups:
                            # Afficher le récapitulatif si plus de matières
                            recap_message = "<div class='bot-message'><b>Récapitulatif de vos choix :</b><br>"
                            for subject, group in st.session_state.selected_groups.items():
                                recap_message += f"- {subject} : {group['display']}<br>"
                            st.session_state.messages.append((recap_message, True))
                            st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
                            st.session_state.step = 9
                            st.rerun()
                            return
                    else:
                        # Afficher le récapitulatif si plus de matières
                        recap_message = "<div class='bot-message'><b>Récapitulatif de vos choix :</b><br>"
                        for subject, group in st.session_state.selected_groups.items():
                            recap_message += f"- {subject} : {group['display']}<br>"
                        st.session_state.messages.append((recap_message, True))
                        st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
                        st.session_state.step = 9
                        st.rerun()
                        return
                groups_message = f"<div class='bot-message'>Voici les groupes recommandés pour {current_subject} :<br>"
                for i, group in enumerate(groups, 1):
                    groups_message += f"{group['display']}<br>"
                groups_message += f"Veuillez entrer le numéro du groupe que vous souhaitez choisir (par exemple, 1 pour {groups[0]['display']}) :</div>"
                st.session_state.messages.append((groups_message, True))
            else:
                # Afficher le récapitulatif
                recap_message = "<div class='bot-message'><b>Récapitulatif de vos choix :</b><br>"
                for subject, group in st.session_state.selected_groups.items():
                    recap_message += f"- {subject} : {group['display']}<br>"
                st.session_state.messages.append((recap_message, True))
                st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
                st.session_state.step = 9
        else:
            st.session_state.messages.append(("<div class='bot-message'>Veuillez entrer un numéro valide (entre 1 et " + str(len(groups)) + ").</div>", True))
            groups_message = f"<div class='bot-message'>Voici les groupes recommandés pour {current_subject} :<br>"
            for i, group in enumerate(groups, 1):
                groups_message += f"{group['display']}<br>"
            groups_message += f"Veuillez entrer le numéro du groupe que vous souhaitez choisir (par exemple, 1 pour {groups[0]['display']}) :</div>"
            st.session_state.messages.append((groups_message, True))
    except ValueError:
        st.session_state.messages.append(("<div class='bot-message'>Veuillez entrer un numéro valide (par exemple, 1).</div>", True))
        groups_message = f"<div class='bot-message'>Voici les groupes recommandés pour {current_subject} :<br>"
        for i, group in enumerate(groups, 1):
            groups_message += f"{group['display']}<br>"
        groups_message += f"Veuillez entrer le numéro du groupe que vous souhaitez choisir (par exemple, 1 pour {groups[0]['display']}) :</div>"
        st.session_state.messages.append((groups_message, True))

# Logique conversationnelle
if st.session_state.step == 0:
    st.session_state.messages = [("<div class='bot-message'>Quel est le nom de l'étudiant ?</div>", True)]
    st.session_state.step = 1
    st.session_state.current_input = ""
    st.session_state.submitted = False
    st.session_state.input_counter = 0
    st.session_state.responses = {}
    st.session_state.all_recommendations = {}
    st.session_state.all_groups_for_selection = {}
    st.session_state.matched_subjects = []
    st.session_state.selected_groups = {}
    st.session_state.current_subject_index = 0
    st.rerun()

elif st.session_state.step in [1, 2, 3, 4, 5, 6]:
    input_key = f"input_step_{st.session_state.step}_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder=placeholders[st.session_state.step - 1])
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        if st.session_state.step in [1, 2, 3, 5]:
            if st.session_state[input_key].strip():
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
        else:
            handle_input_submission(st.session_state.step, st.session_state[input_key])
            st.session_state.input_counter += 1
            st.session_state.current_input = ""
            st.session_state.submitted = False
            st.rerun()

elif st.session_state.step == 8:  # Étape de sélection des groupes
    input_key = f"input_step_8_{st.session_state.current_subject_index}_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder="Entrez le numéro du groupe (ex: 1)")
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        if st.session_state[input_key].strip():
            handle_group_selection(st.session_state[input_key])
            st.session_state.input_counter += 1
            st.session_state.current_input = ""
            st.session_state.submitted = False
            st.rerun()
        else:
            st.session_state.messages.append(("<div class='bot-message'>Veuillez entrer un numéro valide (par exemple, 1).</div>", True))
            current_subject = st.session_state.matched_subjects[st.session_state.current_subject_index]
            groups = st.session_state.all_groups_for_selection.get(current_subject, [])
            groups_message = f"<div class='bot-message'>Voici les groupes recommandés pour {current_subject} :<br>"
            for i, group in enumerate(groups, 1):
                groups_message += f"{group['display']}<br>"
            groups_message += f"Veuillez entrer le numéro du groupe que vous souhaitez choisir (par exemple, 1 pour {groups[0]['display']}) :</div>"
            st.session_state.messages.append((groups_message, True))
            st.session_state.input_counter += 1
            st.session_state.submitted = False
            st.rerun()

elif st.session_state.step == 9:  # Étape finale : recommencer ou terminer
    input_key = f"input_step_9_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder="Oui ou Non")
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        choice = st.session_state[input_key].strip().lower()
        if choice in ["oui", "yes"]:
            # Réinitialiser tous les attributs nécessaires
            st.session_state.clear()
            st.session_state.step = 0
            st.session_state.messages = [("<div class='bot-message'>Bonjour ! Je vais vous aider à trouver des groupes recommandés.</div>", True)]
            st.session_state.responses = {}
            st.session_state.current_input = ""
            st.session_state.submitted = False
            st.session_state.input_counter = 0
            st.session_state.all_recommendations = {}
            st.session_state.all_groups_for_selection = {}
            st.session_state.matched_subjects = []
            st.session_state.selected_groups = {}
            st.session_state.current_subject_index = 0
            st.rerun()
        elif choice in ["non", "no"]:
            st.session_state.step = 10
            st.session_state.messages.append(("<div class='bot-message'>Merci d'avoir utilisé le chatbot ! Au revoir.</div>", True))
            st.session_state.input_counter += 1
            st.session_state.submitted = False
            st.rerun()
        else:
            st.session_state.messages.append(("<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'.</div>", True))
            st.session_state.input_counter += 1
            st.session_state.submitted = False
            st.rerun()