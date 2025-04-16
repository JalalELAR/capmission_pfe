import streamlit as st
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import random
import chromadb
from datetime import datetime, timedelta
import os 

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Build path to chroma_db relative to the current file
chroma_path = os.path.join(parent_dir, "chroma_db5")
# Initialize ChromaDB
client = chromadb.PersistentClient(path=chroma_path)
collection_groupes = client.get_collection(name="groupes_vectorises8")
collection_seances = client.get_or_create_collection(name="seances_vectorises")
collection_combinaisons = client.get_or_create_collection(name="combinaisons_vectorises")
collection_students = client.get_or_create_collection(name="students_vectorises")

# Vérifier si les collections sont vides
if collection_groupes.count() == 0:
    st.error("Erreur : La collection ChromaDB des groupes est vide. Veuillez exécuter la vectorisation d'abord.")
    st.stop()
if collection_combinaisons.count() == 0:
    st.error("Erreur : La collection ChromaDB des combinaisons est vide. Veuillez exécuter la vectorisation des combinaisons d'abord.")
    st.stop()

# Fonction pour charger le modèle SentenceTransformer à la demande
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

# Récupérer toutes les valeurs uniques depuis ChromaDB (groupes)
all_groups = collection_groupes.get(include=["metadatas", "documents"])
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

# Fonction pour extraire les types de durée valides depuis duree_tarifs
def get_valid_duree_types():
    duree_types = set()
    all_groups = collection_groupes.get(include=["metadatas"])
    for metadata in all_groups['metadatas']:
        if 'duree_tarifs' in metadata:
            duree_tarifs = metadata['duree_tarifs'].split(';')
            for dt in duree_tarifs:
                if dt.strip():
                    parts = dt.split(':')
                    if len(parts) == 3:
                        name = parts[0]
                        duree_types.add(name)
    return sorted(list(duree_types))

duree_types_list = get_valid_duree_types()

# Fonctions utilitaires
def match_value(user_input, valid_values):
    if not user_input or not valid_values:
        return user_input, False
    result = process.extractOne(user_input, valid_values)
    if result is None:
        return user_input, False
    best_match, score = result
    return best_match if score > 80 else user_input, score > 80

def count_school_students(group_schools, user_school):
    return sum(1 for school in group_schools if school == user_school)

def parse_time(time_str):
    return datetime.strptime(time_str, "%H:%M")

def has_overlap(group1, group2):
    if group1['jour'] != group2['jour']:
        return False
    start1 = parse_time(group1['heure_debut'])
    end1 = parse_time(group1['heure_fin'])
    start2 = parse_time(group2['heure_debut'])
    end2 = parse_time(group2['heure_fin'])
    if group1['centre'] == group2['centre']:
        return start1 < end2 and start2 < end1 and (end1 != start2 or start1 != end2)
    else:
        margin = timedelta(minutes=15)
        return (start1 < end2 + margin and start2 < end1 + margin)

def check_overlaps(selected_groups):
    overlaps = []
    group_list = list(selected_groups.values())
    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            if has_overlap(group_list[i], group_list[j]):
                overlaps.append((group_list[i], group_list[j]))
    return overlaps

def get_remaining_sessions(id_cours):
    reference_date_fixed = datetime.strptime("2025/03/06", "%Y/%m/%d")
    seances_data = collection_seances.get(include=["metadatas"])
    id_cours_str = str(id_cours)
    relevant_seances = [metadata for metadata in seances_data['metadatas'] if metadata['id_cours'] == id_cours_str]
    remaining = sum(1 for metadata in relevant_seances
                    if datetime.strptime(metadata['date_seance'], "%Y/%m/%d") > reference_date_fixed)
    return remaining

def calculate_tariffs(selected_groups, user_duree_type):
    tariffs_by_group = {}
    total_tariff_base = 0
    reduction_applied = 0
    reduction_description = ""

    selected_forfait_ids = []
    for subject, group in selected_groups.items():
        groupe_data = collection_groupes.get(ids=[group['id_cours']], include=["metadatas"])
        if not groupe_data['metadatas'] or 'duree_tarifs' not in groupe_data['metadatas'][0]:
            return None, f"Erreur : Données de tarification non trouvées pour le cours {group['id_cours']}.", None
        
        duree_tarifs = groupe_data['metadatas'][0]['duree_tarifs'].split(';')
        tarif_unitaire = None
        id_forfait = groupe_data['metadatas'][0].get('id_forfait', None)
        if id_forfait is None:
            return None, f"Erreur : Aucun id_forfait trouvé dans les métadonnées pour le cours {group['id_cours']}.", None
        
        for dt in duree_tarifs:
            if not dt.strip():
                continue
            parts = dt.split(':')
            if len(parts) != 3:
                continue
            name, forfait_id, tarif = parts[0], parts[1], parts[2]
            try:
                if name == user_duree_type and forfait_id == id_forfait:
                    tarif_unitaire = float(tarif)
                    break
            except ValueError:
                continue
        
        if tarif_unitaire is None:
            return None, f"Erreur : Type de durée '{user_duree_type}' avec id_forfait '{id_forfait}' non disponible pour le cours {group['id_cours']}.", None
        
        selected_forfait_ids.append(id_forfait)
        remaining_sessions = get_remaining_sessions(group['id_cours'])
        tarif_total = remaining_sessions * tarif_unitaire
        
        tariffs_by_group[subject] = {
            "id_cours": group['id_cours'],
            "id_forfait": id_forfait,
            "remaining_sessions": remaining_sessions,
            "tarif_unitaire": tarif_unitaire,
            "tarif_total": tarif_total
        }
        total_tariff_base += tarif_total

    if len(selected_forfait_ids) > 1:
        combinaisons_data = collection_combinaisons.get(include=["metadatas"])
        combinaisons_dict = {}
        for metadata in combinaisons_data['metadatas']:
            id_combinaison = metadata['id_combinaison']
            id_forfait = str(metadata['id_forfait'])
            reduction = float(metadata['reduction'])
            if id_combinaison not in combinaisons_dict:
                combinaisons_dict[id_combinaison] = []
            combinaisons_dict[id_combinaison].append((id_forfait, reduction))
        
        for id_combinaison, pairs in combinaisons_dict.items():
            forfait_ids = [pair[0] for pair in pairs]
            reduction_percentage = pairs[0][1]
            if all(id in selected_forfait_ids for id in forfait_ids):
                reduction_amount = total_tariff_base * (reduction_percentage / 100)
                total_tariff_base -= reduction_amount
                reduction_applied = reduction_amount
                reduction_description = f"Réduction pour combinaison ({id_combinaison}) : -{reduction_amount:.2f} DH ({reduction_percentage:.2f}%)"
                break

    tariff_message = f"<b>Détails des tarifs (Type de durée : {user_duree_type}) :</b><br>"
    for subject, info in tariffs_by_group.items():
        tariff_message += f"- {subject} : {info['remaining_sessions']} séances restantes, tarif unitaire {info['tarif_unitaire']} DH, tarif total {info['tarif_total']:.2f} DH<br>"
    tariff_message += f"<b>Total de base :</b> {sum(info['tarif_total'] for info in tariffs_by_group.values()):.2f} DH<br>"
    if reduction_applied > 0:
        tariff_message += f"{reduction_description}<br>"
        tariff_message += f"<b>Total après réduction :</b> {total_tariff_base:.2f} DH"
    else:
        tariff_message += f"<b>Total :</b> {total_tariff_base:.2f} DH"
    
    return tariffs_by_group, tariff_message, total_tariff_base

def get_recommendations(student_name, user_level, user_subjects, user_teachers, user_school, user_center):
    output = []
    matched_level = match_value(user_level, levels)[0]
    matched_subjects = [match_value(subj.strip(), subjects)[0] for subj in user_subjects.split(",")]
    matched_teachers = [teacher.strip() for teacher in user_teachers.split(",")] if user_teachers else [None] * len(matched_subjects)
    matched_school = match_value(user_school, schools)[0]
    matched_center = match_value(user_center, centers)[0] if user_center else None

    all_recommendations = {}
    all_groups_for_selection = {}

    for matched_subject, matched_teacher in zip(matched_subjects, matched_teachers):
        all_groups_data = collection_groupes.get(include=["metadatas", "documents"])
        groups = {}
        for metadata, document in zip(all_groups_data['metadatas'], all_groups_data['documents']):
            if metadata['niveau'].strip().lower() == matched_level.strip().lower() and metadata['matiere'].strip().lower() == matched_subject.strip().lower():
                group_schools = [school.strip() for school in metadata['ecole'].split(", ")]
                group_students = [student.strip() for student in metadata.get('student', '').split(", ")] or [f"Étudiant_{i}" for i in range(int(metadata['num_students']))]

                num_students = int(metadata['num_students'])
                unique_schools = list(dict.fromkeys(group_schools))
                if len(unique_schools) < num_students:
                    unique_schools.extend(["École inconnue"] * (num_students - len(unique_schools)))
                unique_schools = unique_schools[:num_students]

                if len(group_students) < num_students:
                    group_students.extend([f"Étudiant_{i}" for i in range(len(group_students), num_students)])
                group_students = group_students[:num_students]

                school_student_pairs = dict(zip(group_students, unique_schools))

                groups[metadata['id_cours']] = {
                    "id_cours": metadata['id_cours'],
                    "name_cours": metadata['name_cours'],
                    "num_students": num_students,
                    "total_students": int(metadata['total_students']),
                    "description": document,
                    "centre": metadata['centre'],
                    "teacher": metadata['teacher'],
                    "schools": list(school_student_pairs.values()),
                    "students": list(school_student_pairs.keys()),
                    "date_debut": metadata['date_debut'],
                    "date_fin": metadata['date_fin'],
                    "heure_debut": metadata['heure_debut'],
                    "heure_fin": metadata['heure_fin'],
                    "jour": metadata['jour'] if metadata.get('jour') else "None",
                    "niveau": metadata['niveau'],
                    "matiere": metadata['matiere']
                }

        if not groups:
            output.append(f"Aucun groupe trouvé pour {matched_subject} au niveau {matched_level}.")
            continue

        group_list = list(groups.values())
        for group in group_list:
            formatted_schools = "<br>".join(sorted(set(group['schools'])))
            group['description'] = formatted_schools

        selected_groups = []
        selected_ids = set()

        def add_groups(new_groups, criteria="Non spécifié"):
            for group in new_groups:
                if group['id_cours'] not in selected_ids and len(selected_groups) < 3:
                    group['criteria'] = criteria
                    selected_groups.append(group)
                    selected_ids.add(group['id_cours'])

        if matched_center:
            priority_1_groups = []
            if matched_teacher:
                priority_1_groups = [g for g in group_list if matched_teacher == g['teacher'] and matched_center == g['centre'] and matched_school in g['schools']]
                if len(priority_1_groups) >= 3:
                    priority_1_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                    add_groups(priority_1_groups[:3], "Professeur, Centre, École")
                else:
                    add_groups(priority_1_groups, "Professeur, Centre, École")

            if matched_teacher and len(selected_groups) < 3:
                priority_2_groups = [g for g in group_list if matched_teacher == g['teacher'] and matched_center == g['centre'] and g['id_cours'] not in selected_ids]
                add_groups(priority_2_groups, "Professeur, Centre")

            if len(selected_groups) < 3:
                priority_3_groups = [g for g in group_list if matched_center == g['centre'] and matched_school in g['schools'] and g['id_cours'] not in selected_ids]
                priority_3_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(priority_3_groups, "Centre, École")

            if len(selected_groups) < 3:
                priority_4_groups = [g for g in group_list if matched_center == g['centre'] and g['id_cours'] not in selected_ids]
                add_groups(priority_4_groups, "Centre")

            if matched_teacher and len(selected_groups) < 3:
                priority_5_groups = [g for g in group_list if matched_teacher == g['teacher'] and g['id_cours'] not in selected_ids]
                priority_5_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(priority_5_groups, "Professeur")

            if len(selected_groups) < 3:
                priority_6_groups = [g for g in group_list if matched_school in g['schools'] and g['id_cours'] not in selected_ids]
                priority_6_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(priority_6_groups, "École")

            if len(selected_groups) < 3:
                remaining_groups = [g for g in group_list if g['id_cours'] not in selected_ids]
                random.shuffle(remaining_groups)
                add_groups(remaining_groups, "Aucun critère spécifique (aléatoire)")
        else:
            teacher_groups = []
            if matched_teacher:
                teacher_groups = [g for g in group_list if matched_teacher == g['teacher']]

            if len(teacher_groups) >= 3:
                teacher_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(teacher_groups[:3], "Professeur")
            else:
                has_school = any(matched_school in g['schools'] for g in teacher_groups)
                if has_school:
                    teacher_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                add_groups(teacher_groups, "Professeur")

                if len(selected_groups) < 3:
                    school_groups = [g for g in group_list if matched_school in g['schools'] and g['id_cours'] not in selected_ids]
                    school_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
                    add_groups(school_groups, "École")

                if len(selected_groups) < 3:
                    remaining_groups = [g for g in group_list if g['id_cours'] not in selected_ids]
                    random.shuffle(remaining_groups)
                    add_groups(remaining_groups, "Aucun critère spécifique (aléatoire)")

        if len(selected_groups) < 3:
            output.append(f"Attention : Seulement {len(selected_groups)} groupe(s) trouvé(s) pour {matched_subject}.")

        recommendations = []
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
                f"<b>Écoles:</b><br>{group['description']}<br>"
                f"<b>Critères de sélection :</b> {group['criteria']}"
            )
            recommendations.append(recommendation)
            groups_for_selection.append({
                "id_cours": group['id_cours'],
                "name_cours": group['name_cours'],
                "display": f"Groupe {i} : {group['id_cours']} - {group['name_cours']}",
                "centre": group['centre'],
                "heure_debut": group['heure_debut'],
                "heure_fin": group['heure_fin'],
                "jour": group['jour'],
                "matiere": group['matiere'],
                "criteria": group['criteria']
            })
        
        all_recommendations[matched_subject] = recommendations
        all_groups_for_selection[matched_subject] = groups_for_selection

    output.append(f"<b>Les groupes recommandés pour l'étudiant</b> {student_name} :")
    return output, all_recommendations, all_groups_for_selection, matched_subjects

# Styles CSS pour l'affichage
st.markdown("""
    <style>
    .stApp {
        background-color: #808080;
    }
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
    .profile-name {
        text-align: center;
        font-size: 25px;
        color: white;
        margin-top: 10px;
        margin-left: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Interface Streamlit
logo_path = os.path.join(parent_dir, "images", "logo.png")
try:
    st.image(logo_path)
except FileNotFoundError:
    st.warning("Logo non trouvé. Veuillez placer 'logo.png' dans le répertoire du script.")
st.title("Chatbot de Recommandation de Groupes")
profile_path = os.path.join(parent_dir, "images", "profile1.png")

with st.sidebar:
    st.image(profile_path, width=280, use_container_width=False, output_format="auto")
    st.markdown("<div class='profile-name'>ELARACHE Jalal</div>", unsafe_allow_html=True)
    st.header("Options")
    st.write("Bienvenue dans le Chatbot de Recommandation !")
    st.write("Utilisez ce chatbot pour trouver des groupes adaptés à vos besoins.")
    if st.button("Réinitialiser la conversation"):
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
        st.session_state.subject_grades = {}
        st.session_state.course_choices = {}
        st.session_state.tariffs_by_group = {}
        st.session_state.total_tariff_base = 0
        st.rerun()
    st.write("---")
    st.write("**À propos**")
    st.write("Développé par IA pour optimiser la recherche de groupes éducatifs.")
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
    st.session_state.subject_grades = {}
    st.session_state.course_choices = {}
    st.session_state.tariffs_by_group = {}
    st.session_state.total_tariff_base = 0

# Afficher les messages précédents
st.markdown("<div class='container'>", unsafe_allow_html=True)
for message, is_bot in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Liste des questions et placeholders
questions = [
    "Quel est le nom de l'étudiant ?",
    "Quel est le niveau de l'étudiant (ex. Terminale, Première) ?",
    "Quelles sont les matières qui intéressent l'étudiant ?",
    "Quelles sont les notes de l'étudiant pour ces matières ? (facultatif)",
    "Veuillez choisir le type de cours pour chaque matière (indiv/groupe)",
    "Quels sont les professeurs actuels de l'étudiant pour ces matières ?",
    "Quelle est l'école de l'étudiant (obligatoire) ?",
    "Quel est le centre souhaité par l'étudiant ?",
    "Quel est le type de durée souhaité ?",
    "Veuillez sélectionner un groupe pour chaque matière",
    "Voulez-vous inclure les frais d'inscription (250 DH par matière en groupe) ?",
    "Voulez-vous traiter un autre cas ?"
]

placeholders = [
    "Ex: Ahmed",
    f"Ex: {', '.join(levels_list[:3]) if levels_list else 'Terminale, Première'}",
    f"Ex: {', '.join(subjects_list[:3]) if subjects_list else 'Français, Maths'}",
    "Ex: 12, 15 (ou laissez vide)",
    "Ex: indiv, groupe",
    "Ex: John Doe, Jane Smith (facultatif, peut être vide ou partiel)",
    f"Ex: {schools_list[0] if schools_list else 'Massignon Bouskoura'}",
    f"Ex: {centers_list[0] if centers_list else 'Centre A'} (ou laissez vide)",
    f"Ex: {', '.join(duree_types_list[:2]) if duree_types_list else 'MF Trimestre 1'}",
    "Ex: 1, 2",
    "Oui ou Non",
    "Oui ou Non"
]

# Fonctions de gestion des étapes
def handle_input_submission(step, response):
    if step == 1:  # Nom de l'étudiant
        if response.strip():
            students_data = collection_students.get(include=["metadatas"])
            student_exists = False
            for metadata in students_data['metadatas']:
                if metadata.get('student_name', '').strip().lower() == response.strip().lower():
                    student_exists = True
                    break
            
            st.session_state.responses['student_name'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            if student_exists:
                st.session_state.messages.append(("<div class='bot-message'>Étudiant existe</div>", True))
            else:
                st.session_state.messages.append(("<div class='bot-message'>Nouvel étudiant</div>", True))
            st.session_state.messages.append(("<div class='bot-message'>Quel est le niveau de l'étudiant ?</div>", True))
            st.session_state.step = 2
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer un nom valide.</div>", True))
    
    elif step == 2:  # Niveau
        if response.strip():
            matched_level, is_valid = match_value(response.strip(), levels_list)
            if is_valid:
                st.session_state.responses['user_level'] = matched_level
                st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                st.session_state.messages.append(("<div class='bot-message'>Quelles sont les matières qui intéressent l'étudiant ?</div>", True))
                st.session_state.step = 3
            else:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Le niveau '{response}' n'est pas valide. Veuillez choisir parmi : {', '.join(levels_list)}. Ressaisissez :</div>", True))
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer un niveau valide.</div>", True))
    
    elif step == 3:  # Matières
        if response.strip():
            subjects = [s.strip() for s in response.split(',')]
            all_valid = True
            matched_subjects = []
            for subj in subjects:
                matched_subj, is_valid = match_value(subj, subjects_list)
                if is_valid:
                    matched_subjects.append(matched_subj)
                else:
                    all_valid = False
                    break
            if all_valid:
                st.session_state.responses['user_subjects'] = ', '.join(matched_subjects)
                st.session_state.matched_subjects = matched_subjects
                st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                st.session_state.messages.append(("<div class='bot-message'>Quelles sont les notes de l'étudiant pour ces matières ? (facultatif, ex. 12, 15 ou laissez vide)</div>", True))
                st.session_state.step = 4
            else:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Une ou plusieurs matières ne sont pas valides. Veuillez choisir parmi : {', '.join(subjects_list)}. Ressaisissez :</div>", True))
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer des matières valides.</div>", True))
    
    elif step == 4:  # Notes (facultatif)
        subjects = st.session_state.matched_subjects
        if response.strip():
            try:
                grades = [float(g.strip()) for g in response.split(',')]
                if len(grades) > len(subjects):
                    st.session_state.messages.append(("<div class='bot-message'>Erreur : Vous avez saisi trop de notes ({len(grades)}) pour {len(subjects)} matières. Ressaisissez :</div>", True))
                elif any(g < 0 or g > 20 for g in grades):
                    st.session_state.messages.append(("<div class='bot-message'>Erreur : Les notes doivent être entre 0 et 20. Ressaisissez :</div>", True))
                else:
                    grades_extended = grades + [None] * (len(subjects) - len(grades))
                    st.session_state.responses['user_grades'] = response
                    st.session_state.subject_grades = dict(zip(subjects, grades_extended))
                    st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                    
                    recommendation_msg = "<div class='bot-message'>Recommandations basées sur les informations fournies :<br>"
                    for subject, grade in st.session_state.subject_grades.items():
                        if grade is not None and grade < 8:
                            recommendation_msg += f"- {subject} (Note : {grade}) : Note faible. Nous recommandons des <b>cours individuels</b> pour une mise à niveau efficace.<br>"
                        else:
                            reason = "aucune note fournie" if grade is None else f"note : {grade}"
                            recommendation_msg += f"- {subject} ({reason}) : Nous recommandons des <b>cours individuels</b> pour leur enseignement personnalisé et leur progression rapide adaptée à vos besoins.<br>"
                    recommendation_msg += f"Veuillez choisir le type de cours pour chaque matière dans l'ordre ({', '.join(subjects)}) : (ex. indiv, groupe)</div>"
                    st.session_state.messages.append((recommendation_msg, True))
                    st.session_state.step = 5
            except ValueError:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Les notes doivent être des nombres valides (ex. 12, 15). Ressaisissez ou laissez vide :</div>", True))
        else:
            st.session_state.responses['user_grades'] = ""
            st.session_state.subject_grades = {subject: None for subject in subjects}
            st.session_state.messages.append(("<div class='user-message'>Aucune note fournie</div>", False))
            
            recommendation_msg = "<div class='bot-message'>Recommandations basées sur les informations fournies :<br>"
            for subject in subjects:
                recommendation_msg += f"- {subject} (Aucune note) : Nous recommandons des <b>cours individuels</b> pour leur enseignement personnalisé et leur progression rapide adaptée à vos besoins.<br>"
            recommendation_msg += f"Veuillez choisir le type de cours pour chaque matière dans l'ordre ({', '.join(subjects)}) : (ex. indiv, groupe)</div>"
            st.session_state.messages.append((recommendation_msg, True))
            st.session_state.step = 5
    
    elif step == 5:  # Choix des cours (indiv/groupe)
        if response.strip():
            choices = [c.strip().lower() for c in response.split(',')]
            subjects = st.session_state.matched_subjects
            if len(choices) != len(subjects):
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Vous devez indiquer un choix pour chaque matière ({len(subjects)} choix requis). Ressaisissez (indiv/groupe) :</div>", True))
            elif not all(c in ['indiv', 'groupe'] for c in choices):
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Les choix doivent être 'indiv' ou 'groupe'. Ressaisissez :</div>", True))
            else:
                st.session_state.course_choices = dict(zip(subjects, choices))
                st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                
                group_subjects = [subject for subject, choice in st.session_state.course_choices.items() if choice == 'groupe']
                indiv_subjects = [subject for subject, choice in st.session_state.course_choices.items() if choice == 'indiv']
                
                for subject in indiv_subjects:
                    st.session_state.messages.append((f"<div class='bot-message'>Les cours individuels pour {subject} sont en préparation et seront disponibles ultérieurement.</div>", True))
                
                if not group_subjects:
                    st.session_state.messages.append(("<div class='bot-message'>Aucune matière en groupe sélectionnée. Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                    st.session_state.step = 12
                else:
                    st.session_state.matched_subjects = group_subjects
                    st.session_state.responses['user_subjects'] = ', '.join(group_subjects)
                    st.session_state.messages.append(("<div class='bot-message'>Quels sont les professeurs actuels de l'étudiant pour ces matières ?</div>", True))
                    st.session_state.step = 6
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez indiquer vos choix (indiv/groupe).</div>", True))
    
    elif step == 6:  # Professeurs
        if response.strip():
            st.session_state.responses['user_teachers'] = response.strip()
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        else:
            st.session_state.responses['user_teachers'] = None
            st.session_state.messages.append(("<div class='user-message'>Aucun spécifié</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quelle est l'école de l'étudiant (obligatoire) ?</div>", True))
        st.session_state.step = 7
    
    elif step == 7:  # École
        if response.strip():
            matched_school, is_valid = match_value(response.strip(), schools_list)
            if is_valid:
                st.session_state.responses['user_school'] = matched_school
                st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                st.session_state.messages.append(("<div class='bot-message'>Quel est le centre souhaité par l'étudiant ?</div>", True))
                st.session_state.step = 8
            else:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : L'école '{response}' n'est pas valide. Veuillez choisir parmi : {', '.join(schools_list)}. Ressaisissez :</div>", True))
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une école valide.</div>", True))
    
    elif step == 8:  # Centre
        if response.strip():
            matched_center, is_valid = match_value(response.strip(), centers_list)
            if is_valid:
                st.session_state.responses['user_center'] = matched_center
                st.session_state.messages.append(("<div class='user-message'>{response}</div>", False))
                st.session_state.messages.append(("<div class='bot-message'>Quel est le type de durée souhaité ?</div>", True))
                st.session_state.step = 9
            else:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Le centre '{response}' n'est pas valide. Veuillez choisir parmi : {', '.join(centers_list)} ou laissez vide. Ressaisissez :</div>", True))
        else:
            st.session_state.responses['user_center'] = None
            st.session_state.messages.append(("<div class='user-message'>Aucun spécifié</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quel est le type de durée souhaité ?</div>", True))
            st.session_state.step = 9
    
    elif step == 9:  # Type de durée
        if response.strip():
            matched_duree, is_valid = match_value(response.strip(), duree_types_list)
            if is_valid:
                st.session_state.responses['user_duree_type'] = matched_duree
                st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                
                group_subjects = st.session_state.matched_subjects
                with st.spinner("Recherche en cours..."):
                    output, all_recommendations, all_groups_for_selection, matched_subjects = get_recommendations(
                        st.session_state.responses['student_name'],
                        st.session_state.responses['user_level'],
                        ', '.join(group_subjects),
                        st.session_state.responses.get('user_teachers', ''),
                        st.session_state.responses['user_school'],
                        st.session_state.responses.get('user_center', '')
                    )
                st.session_state.all_recommendations = all_recommendations
                st.session_state.all_groups_for_selection = all_groups_for_selection
                st.session_state.matched_subjects = matched_subjects

                for msg in output:
                    st.session_state.messages.append((f"<div class='bot-message'>{msg}</div>", True))
                for subject in matched_subjects:
                    for rec in all_recommendations.get(subject, []):
                        st.session_state.messages.append((f"<div class='bot-message'>{rec}</div>", True))
                
                if not any(st.session_state.all_groups_for_selection.values()):
                    st.session_state.messages.append(("<div class='bot-message'>Aucun groupe disponible pour sélection.</div>", True))
                    st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                    st.session_state.step = 12
                else:
                    groups_message = "<div class='bot-message'>Voici les groupes recommandés pour les matières en groupe :<br>"
                    for subject in matched_subjects:
                        groups = st.session_state.all_groups_for_selection.get(subject, [])
                        if groups:
                            groups_message += f"<h3>{subject}</h3>"
                            for i, group in enumerate(groups, 1):
                                groups_message += f"{group['display']}<br>"
                    groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
                    st.session_state.messages.append((groups_message, True))
                    st.session_state.step = 10
            else:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Le type de durée '{response}' n'est pas valide. Veuillez choisir parmi : {', '.join(duree_types_list)}. Ressaisissez :</div>", True))
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer un type de durée valide.</div>", True))
    
    elif step == 10:  # Sélection des groupes
        if response.strip():
            try:
                selections = [int(s.strip()) for s in response.split(',')]
                matched_subjects = st.session_state.matched_subjects
                if len(selections) != len(matched_subjects):
                    st.session_state.messages.append(("<div class='bot-message'>Erreur : Vous devez sélectionner un groupe pour chaque matière ({len(matched_subjects)} sélections requises). Ressaisissez :</div>", True))
                    groups_message = "<div class='bot-message'>Voici les groupes recommandés pour les matières en groupe :<br>"
                    for subject in matched_subjects:
                        groups = st.session_state.all_groups_for_selection.get(subject, [])
                        if groups:
                            groups_message += f"<h3>{subject}</h3>"
                            for i, group in enumerate(groups, 1):
                                groups_message += f"{group['display']}<br>"
                    groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
                    st.session_state.messages.append((groups_message, True))
                else:
                    selected_groups = {}
                    for subject, selection in zip(matched_subjects, selections):
                        groups = st.session_state.all_groups_for_selection.get(subject, [])
                        if selection <= len(groups):
                            selected_groups[subject] = groups[selection - 1]
                        else:
                            st.session_state.messages.append(("<div class='bot-message'>Erreur : Sélection invalide pour {subject}. Ressaisissez :</div>", True))
                            groups_message = "<div class='bot-message'>Voici les groupes recommandés pour les matières en groupe :<br>"
                            for subj in matched_subjects:
                                grps = st.session_state.all_groups_for_selection.get(subj, [])
                                if grps:
                                    groups_message += f"<h3>{subj}</h3>"
                                    for i, grp in enumerate(grps, 1):
                                        groups_message += f"{grp['display']}<br>"
                            groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
                            st.session_state.messages.append((groups_message, True))
                            return
                    
                    st.session_state.responses['group_selections'] = response
                    st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                    
                    overlaps = check_overlaps(selected_groups)
                    if overlaps:
                        conflict_msg = "<div class='bot-message'><b>Conflit détecté :</b><br>"
                        for g1, g2 in overlaps:
                            conflict_msg += f"- Chevauchement entre {g1['matiere']} ({g1['display']}) et {g2['matiere']} ({g2['display']}) : "
                            conflict_msg += f"{g1['jour']} {g1['heure_debut']}-{g1['heure_fin']} vs {g2['jour']} {g2['heure_debut']}-{g2['heure_fin']}<br>"
                        conflict_msg += "Veuillez choisir une nouvelle combinaison de groupes :</div>"
                        st.session_state.messages.append((conflict_msg, True))
                        groups_message = "<div class='bot-message'>Voici les groupes recommandés pour les matières en groupe :<br>"
                        for subject in matched_subjects:
                            groups = st.session_state.all_groups_for_selection.get(subject, [])
                            if groups:
                                groups_message += f"<h3>{subject}</h3>"
                                for i, group in enumerate(groups, 1):
                                    groups_message += f"{group['display']}<br>"
                        groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
                        st.session_state.messages.append((groups_message, True))
                        return
                    
                    st.session_state.selected_groups = selected_groups
                    
                    tariffs_by_group, tariff_message, total_tariff_base = calculate_tariffs(
                        selected_groups,
                        st.session_state.responses['user_duree_type']
                    )
                    if tariffs_by_group is None:
                        st.session_state.messages.append((f"<div class='bot-message'>{tariff_message}</div>", True))
                        st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                        st.session_state.step = 12
                        return
                    
                    st.session_state.tariffs_by_group = tariffs_by_group
                    st.session_state.total_tariff_base = total_tariff_base
                    
                    st.session_state.messages.append((f"<div class='bot-message'>{tariff_message}</div>", True))
                    
                    num_group_subjects = len(matched_subjects)
                    if num_group_subjects > 0:
                        frais_inscription = num_group_subjects * 250
                        frais_message = f"<div class='bot-message'>Frais d'inscription possibles : {frais_inscription} DH ({num_group_subjects} matière(s) en groupe x 250 DH). Voulez-vous inclure ces frais ? (Oui/Non)</div>"
                        st.session_state.messages.append((frais_message, True))
                        st.session_state.step = 11
            except ValueError:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Les sélections doivent être des nombres (ex. 1, 2). Ressaisissez :</div>", True))
                groups_message = "<div class='bot-message'>Voici les groupes recommandés pour les matières en groupe :<br>"
                for subject in matched_subjects:
                    groups = st.session_state.all_groups_for_selection.get(subject, [])
                    if groups:
                        groups_message += f"<h3>{subject}</h3>"
                        for i, group in enumerate(groups, 1):
                            groups_message += f"{group['display']}<br>"
                groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
                st.session_state.messages.append((groups_message, True))
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez sélectionner des groupes.</div>", True))
    
    elif step == 11:  # Frais d'inscription
        choice = response.strip().lower()
        if choice in ['oui', 'yes']:
            num_group_subjects = len(st.session_state.matched_subjects)
            frais_inscription = num_group_subjects * 250
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            frais_message = f"<div class='bot-message'>Frais d'inscription inclus : {frais_inscription} DH ({num_group_subjects} matière(s) en groupe x 250 DH)</div>"
            st.session_state.messages.append((frais_message, True))
            
            total_final = st.session_state.total_tariff_base + frais_inscription
            total_message = f"<div class='bot-message'>Total final (tarifs + frais d'inscription) : {total_final:.2f} DH</div>"
            st.session_state.messages.append((total_message, True))
            
            st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
            st.session_state.step = 12
        elif choice in ['non', 'no']:
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            total_message = f"<div class='bot-message'>Total final (sans frais d'inscription) : {st.session_state.total_tariff_base:.2f} DH</div>"
            st.session_state.messages.append((total_message, True))
            
            st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
            st.session_state.step = 12
        else:
            st.session_state.messages.append(("<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'.</div>", True))
    
    elif step == 12:  # Traiter un autre cas
        choice = response.strip().lower()
        if choice in ['oui', 'yes']:
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
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
            st.session_state.subject_grades = {}
            st.session_state.course_choices = {}
            st.session_state.tariffs_by_group = {}
            st.session_state.total_tariff_base = 0
            st.rerun()
        elif choice in ['non', 'no']:
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Merci d'avoir utilisé le chatbot. Au revoir !</div>", True))
            st.session_state.step = 13
        else:
            st.session_state.messages.append(("<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'.</div>", True))

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
    st.session_state.subject_grades = {}
    st.session_state.course_choices = {}
    st.session_state.tariffs_by_group = {}
    st.session_state.total_tariff_base = 0
    st.rerun()

elif st.session_state.step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    input_key = f"input_step_{st.session_state.step}_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder=placeholders[st.session_state.step - 1])
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        handle_input_submission(st.session_state.step, st.session_state[input_key])
        st.session_state.input_counter += 1
        st.session_state.current_input = ""
        st.session_state.submitted = False
        st.rerun()