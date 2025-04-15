__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') 
import streamlit as st
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import random
import chromadb
from datetime import datetime, timedelta
import os 

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build path to chroma_db relative to the current file
chroma_path = os.path.join(current_dir, "..", "chroma_db5")
# Initialize ChromaDB
client = chromadb.PersistentClient(path=chroma_path)
collection_groupes = client.get_collection(name="groupes_vectorises8")
collection_seances = client.get_or_create_collection(name="seances_vectorises")
collection_combinaisons = client.get_or_create_collection(name="combinaisons_vectorises")

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

# Fonctions utilitaires
def match_value(user_input, valid_values):
    if not user_input or not valid_values:
        return user_input
    result = process.extractOne(user_input, valid_values)
    if result is None:
        return user_input
    best_match, score = result
    return best_match if score > 80 else user_input

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

    # Calcul des tarifs de base pour chaque groupe et récupération des id_forfait
    selected_forfait_ids = []  # Liste pour stocker les id_forfait
    for subject, group in selected_groups.items():
        groupe_data = collection_groupes.get(ids=[group['id_cours']], include=["metadatas"])
        if not groupe_data['metadatas'] or 'duree_tarifs' not in groupe_data['metadatas'][0]:
            return None, f"Erreur : Données de tarification non trouvées pour le cours {group['id_cours']}.", None
        
        duree_tarifs = groupe_data['metadatas'][0]['duree_tarifs'].split(';')
        print(f"duree_tarifs pour {group['id_cours']} : {duree_tarifs}")  # Débogage
        
        tarif_unitaire = None
        id_forfait = groupe_data['metadatas'][0].get('id_forfait', None)
        if id_forfait is None:
            return None, f"Erreur : Aucun id_forfait trouvé dans les métadonnées pour le cours {group['id_cours']}.", None
        
        for dt in duree_tarifs:
            if not dt.strip():  # Ignorer les entrées vides
                continue
            parts = dt.split(':')
            print(f"Parsing {dt} -> {parts}")  # Débogage
            if len(parts) != 3:
                print(f"Format invalide pour {dt} dans duree_tarifs de {group['id_cours']}")
                continue
            name, forfait_id, tarif = parts[0], parts[1], parts[2]
            try:
                if name == user_duree_type and forfait_id == id_forfait:
                    tarif_unitaire = float(tarif)
                    break
            except ValueError:
                print(f"Erreur de conversion en float pour tarif '{tarif}' dans {dt}")
                continue
        
        if tarif_unitaire is None:
            return None, f"Erreur : Type de durée '{user_duree_type}' avec id_forfait '{id_forfait}' non disponible ou mal formaté pour le cours {group['id_cours']}. Vérifiez duree_tarifs : {duree_tarifs}", None
        
        selected_forfait_ids.append(id_forfait)  # Ajouter l'id_forfait à la liste
        
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

    # Vérification des combinaisons pour réduction avec les id_forfait
    print("IDs forfait sélectionnés :", selected_forfait_ids)  # Débogage
    if len(selected_forfait_ids) > 1:  # Réduction applicable uniquement pour plusieurs matières
        combinaisons_data = collection_combinaisons.get(include=["metadatas"])
        combinaisons_dict = {}
        for metadata in combinaisons_data['metadatas']:
            id_combinaison = metadata['id_combinaison']
            id_forfait = str(metadata['id_forfait'])  # Conversion en chaîne
            reduction = float(metadata['reduction'])  # Conversion explicite en float
            if id_combinaison not in combinaisons_dict:
                combinaisons_dict[id_combinaison] = []
            combinaisons_dict[id_combinaison].append((id_forfait, reduction))
        
        print("Combinaisons dans ChromaDB :", combinaisons_dict)  # Débogage

        # Vérifier chaque combinaison
        for id_combinaison, pairs in combinaisons_dict.items():
            forfait_ids = [pair[0] for pair in pairs]
            reduction_percentage = pairs[0][1]  # Première réduction trouvée
            print(f"Vérification combinaison {id_combinaison} : {forfait_ids}")  # Débogage
            print(f"Comparaison avec selected_forfait_ids : {selected_forfait_ids}")
            all_present = all(id in selected_forfait_ids for id in forfait_ids)
            print(f"Tous les IDs présents ? {all_present}")  # Débogage
            if all_present:
                reduction_amount = total_tariff_base * (reduction_percentage / 100)
                total_tariff_base -= reduction_amount
                reduction_applied = reduction_amount
                reduction_description = f"Réduction pour combinaison ({id_combinaison}) : -{reduction_amount:.2f} DH ({reduction_percentage:.2f}%)"
                print(f"Combinaison {id_combinaison} appliquée : réduction {reduction_percentage}%")  # Débogage
                break  # Appliquer la première combinaison trouvée

    # Construction du message des tarifs
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
    matched_level = match_value(user_level, levels)
    matched_subjects = [match_value(subj.strip(), subjects) for subj in user_subjects.split(",")]
    matched_teachers = [match_value(teacher.strip(), teachers) for teacher in user_teachers.split(",")] if user_teachers else [None] * len(matched_subjects)
    matched_school = match_value(user_school, schools)
    matched_center = match_value(user_center, centers) if user_center else None

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
                    print(f"Attention : Pas assez d'écoles uniques ({len(unique_schools)}) pour {num_students} étudiants dans id_cours={metadata['id_cours']}")
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
    /* Fond gris pour toute la page */
    .stApp {
        background-color: #808080; /* Gris clair, ajustez la valeur si besoin (#808080 pour gris moyen) */
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
# Ajouter le logo en haut
logo_path = os.path.join(current_dir, "images", "logo.png")
try:
    st.image(logo_path)  # Ajustez le chemin et la largeur selon vos besoins
except FileNotFoundError:
    st.warning("Logo non trouvé. Veuillez placer 'logo.png' dans le répertoire du script.")
st.title("Chatbot de Recommandation de Groupes")
# Ajouter une barre latérale
profile_path = os.path.join(current_dir, "images", "profile1.png")
with st.sidebar:
    st.image("profile_path", width=280, use_container_width=False, output_format="auto")
    # Nom sous la photo
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
        st.session_state.low_grade_subjects = []
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
    st.session_state.low_grade_subjects = []
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
    "Quelles sont les notes de l'étudiant pour ces matières ?",
    "Quels sont les professeurs actuels de l'étudiant pour ces matières ?",
    "Quelle est l'école de l'étudiant (obligatoire)?",
    "Quel est le centre souhaité par l'étudiant ?",
    "Quel est le type de durée souhaité  ?"
]

placeholders = [
    "Ex: Ahmed",
    "Ex: BL - 1bac sc maths",
    "Ex: Français, Maths",
    "Ex: 12, 15",
    "Ex: Ahmed Belkadi, Sara Lahlou (ou laissez vide)",
    "Ex: Massignon Bouskoura",
    "Ex: Centre A (ou laissez vide)",
    "Ex: MF Trimestre 1"
]

# Fonctions de gestion des étapes
def handle_input_submission(step, response):
    if step == 1:
        if response.strip():
            st.session_state.responses['student_name'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quel est le niveau de l'étudiant ?</div>", True))
            st.session_state.step = 2
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 2:
        if response.strip():
            st.session_state.responses['user_level'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quelles sont les matières qui intéressent l'étudiant ?</div>", True))
            st.session_state.step = 3
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 3:
        if response.strip():
            st.session_state.responses['user_subjects'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quelles sont les notes de l'étudiant pour ces matières ?</div>", True))
            st.session_state.step = 4
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 4:
        if response.strip():
            st.session_state.responses['user_grades'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quels sont les professeurs actuels de l'étudiant pour ces matières ?</div>", True))
            st.session_state.step = 5
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 5:
        st.session_state.responses['user_teachers'] = response.strip() if response.strip() else None
        st.session_state.messages.append((f"<div class='user-message'>{response if response.strip() else 'Aucun spécifié'}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quelle est l'école de l'étudiant (obligatoire)?</div>", True))
        st.session_state.step = 6
    elif step == 6:
        if response.strip():
            st.session_state.responses['user_school'] = response
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            st.session_state.messages.append(("<div class='bot-message'>Quel est le centre souhaité par l'étudiant ?</div>", True))
            st.session_state.step = 7
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))
    elif step == 7:
        st.session_state.responses['user_center'] = response.strip() if response.strip() else None
        st.session_state.messages.append((f"<div class='user-message'>{response if response.strip() else 'Aucun spécifié'}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Quel est le type de durée souhaité ?</div>", True))
        st.session_state.step = 8
    elif step == 8:
        if response.strip():
            st.session_state.responses['user_duree_type'] = response.strip()
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
            
            # Vérifier les notes
            subjects = [s.strip() for s in st.session_state.responses['user_subjects'].split(',')]
            try:
                grades = [float(g.strip()) for g in st.session_state.responses['user_grades'].split(',')]
                if len(subjects) != len(grades):
                    st.session_state.messages.append(("<div class='bot-message'>Erreur : Le nombre de notes doit correspondre au nombre de matières.</div>", True))
                    st.session_state.messages.append(("<div class='bot-message'>Quelles sont les notes de l'étudiant pour ces matières ?</div>", True))
                    st.session_state.step = 4
                else:
                    low_grades = [(s, g) for s, g in zip(subjects, grades) if g < 8]
                    st.session_state.low_grade_subjects = [s for s, g in low_grades]
                    
                    if low_grades:
                        if len(low_grades) == 1:
                            subject, grade = low_grades[0]
                            msg = f"<div class='bot-message'>Nous avons remarqué que la note de {subject} est faible : {grade}.<br>"
                            msg += "Nous recommandons des cours individuels pour une bonne mise à niveau dans cette matière.<br>"
                            msg += "Voulez-vous continuer avec des cours individuels ou des cours en groupe ? (Oui pour individuels, Non pour groupes) :</div>"
                        else:
                            msg = "<div class='bot-message'>Nous avons remarqué que certaines notes sont faibles :<br>"
                            for subject, grade in low_grades:
                                msg += f"- {subject} : {grade}<br>"
                            msg += "Nous recommandons des cours individuels pour une bonne mise à niveau dans ces matières.<br>"
                            msg += "Voulez-vous continuer avec des cours individuels ou des cours en groupe ? (Oui pour individuels, Non pour groupes) :</div>"
                        st.session_state.messages.append((msg, True))
                        st.session_state.step = 9
                    else:
                        st.session_state.messages.append(("<div class='bot-message'>Merci ! Je recherche les groupes recommandés...</div>", True))
                        st.session_state.step = 10
            except ValueError:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Les notes doivent être des nombres valides (ex. 12, 15).</div>", True))
                st.session_state.messages.append(("<div class='bot-message'>Quelles sont les notes de l'étudiant pour ces matières ?</div>", True))
                st.session_state.step = 4
        else:
            st.session_state.messages.append(("<div class='bot-message'>Ce champ est obligatoire. Veuillez entrer une valeur.</div>", True))

def handle_course_choice(response):
    choice = response.strip().lower()
    if choice in ["oui", "yes"]:
        st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>D'accord, nous préparons une solution pour les cours individuels. Pour l'instant, c'est tout ! Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
        st.session_state.step = 12
    elif choice in ["non", "no"]:
        st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Merci ! Je recherche les groupes recommandés...</div>", True))
        st.session_state.step = 10
    else:
        st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'. Voulez-vous continuer avec des cours individuels ou des cours en groupe ?</div>", True))

def handle_group_selection(response):
    choices = [choice.strip() for choice in response.split(",")]
    matched_subjects = st.session_state.matched_subjects
    all_groups_for_selection = st.session_state.all_groups_for_selection

    if len(choices) != len(matched_subjects):
        st.session_state.messages.append((f"<div class='bot-message'>Erreur : Vous devez entrer exactement {len(matched_subjects)} numéros (un par matière : {', '.join(matched_subjects)}). Réessayez.</div>", True))
        groups_message = "<div class='bot-message'>Voici les groupes recommandés pour toutes les matières :<br>"
        for subject in matched_subjects:
            groups = all_groups_for_selection.get(subject, [])
            if groups:
                groups_message += f"<h3>{subject}</h3>"
                for i, group in enumerate(groups, 1):
                    groups_message += f"{group['display']}<br>"
        groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
        st.session_state.messages.append((groups_message, True))
    else:
        selected_groups = {}
        invalid_choices = False
        for subject, choice in zip(matched_subjects, choices):
            try:
                group_number = int(choice)
                groups = all_groups_for_selection.get(subject, [])
                if not groups:
                    st.session_state.messages.append((f"<div class='bot-message'>Erreur : Aucun groupe disponible pour {subject}.</div>", True))
                    invalid_choices = True
                    break
                if 1 <= group_number <= len(groups):
                    selected_groups[subject] = groups[group_number - 1]
                else:
                    st.session_state.messages.append((f"<div class='bot-message'>Erreur : Le numéro {group_number} n'est pas valide pour {subject} (choix entre 1 et {len(groups)}).</div>", True))
                    invalid_choices = True
                    break
            except ValueError:
                st.session_state.messages.append((f"<div class='bot-message'>Erreur : '{choice}' n'est pas un numéro valide pour {subject}.</div>", True))
                invalid_choices = True
                break

        if invalid_choices:
            groups_message = "<div class='bot-message'>Voici les groupes recommandés pour toutes les matières :<br>"
            for subject in matched_subjects:
                groups = all_groups_for_selection.get(subject, [])
                if groups:
                    groups_message += f"<h3>{subject}</h3>"
                    for i, group in enumerate(groups, 1):
                        groups_message += f"{group['display']}<br>"
            groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
            st.session_state.messages.append((groups_message, True))
        else:
            st.session_state.selected_groups = selected_groups
            st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))

            overlaps = check_overlaps(selected_groups)
            if overlaps:
                conflict_msg = "<div class='bot-message'><b>Conflit détecté :</b><br>"
                for g1, g2 in overlaps:
                    conflict_msg += f"- Chevauchement entre {g1['matiere']} ({g1['display']}) et {g2['matiere']} ({g2['display']}) : "
                    conflict_msg += f"{g1['jour']} {g1['heure_debut']}-{g1['heure_fin']} vs {g2['jour']} {g2['heure_debut']}-{g2['heure_fin']}<br>"
                conflict_msg += "Veuillez choisir une nouvelle combinaison de groupes :</div>"
                st.session_state.messages.append((conflict_msg, True))
                groups_message = "<div class='bot-message'>Voici les groupes recommandés pour toutes les matières :<br>"
                for subject in matched_subjects:
                    groups = all_groups_for_selection.get(subject, [])
                    if groups:
                        groups_message += f"<h3>{subject}</h3>"
                        for i, group in enumerate(groups, 1):
                            groups_message += f"{group['display']}<br>"
                groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
                st.session_state.messages.append((groups_message, True))
                st.session_state.selected_groups = {}
            else:
                recap_message = "<div class='bot-message'><b>Récapitulatif de vos choix :</b><br>"
                for subject, group in selected_groups.items():
                    recap_message += f"- {subject} : {group['display']}<br>"
                st.session_state.messages.append((recap_message, True))

                tariffs_by_group, tariff_message, total_tariff_base = calculate_tariffs(selected_groups, st.session_state.responses['user_duree_type'])
                if tariffs_by_group is None:
                    st.session_state.messages.append((f"<div class='bot-message'>{tariff_message}</div>", True))
                    st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
                    st.session_state.step = 12
                else:
                    st.session_state.tariffs_by_group = tariffs_by_group
                    st.session_state.total_tariff_base = total_tariff_base
                    st.session_state.messages.append((f"<div class='bot-message'>{tariff_message}</div>", True))
                    st.session_state.messages.append(("<div class='bot-message'>Voulez-vous inclure les frais d'inscription (250 DH) ?</div>", True))
                    st.session_state.step = 13

def handle_fees_choice(response):
    choice = response.strip().lower()
    total_tariff_base = st.session_state.total_tariff_base
    if choice in ["oui", "yes"]:
        st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        final_tariff = total_tariff_base + 250
        st.session_state.messages.append((f"<div class='bot-message'>Frais d'inscription inclus. <b>Total final :</b> {final_tariff:.2f} DH</div>", True))
    elif choice in ["non", "no"]:
        st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        final_tariff = total_tariff_base
        st.session_state.messages.append((f"<div class='bot-message'>Frais d'inscription non inclus. <b>Total final :</b> {final_tariff:.2f} DH</div>", True))
    else:
        st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
        st.session_state.messages.append(("<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'. Voulez-vous inclure les frais d'inscription (250 DH) ?</div>", True))
        return

    st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
    st.session_state.step = 12

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
    st.session_state.low_grade_subjects = []
    st.session_state.tariffs_by_group = {}
    st.session_state.total_tariff_base = 0
    st.rerun()

elif st.session_state.step in [1, 2, 3, 4, 5, 6, 7, 8]:
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

elif st.session_state.step == 9:
    input_key = f"input_step_9_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder="Oui ou Non")
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        handle_course_choice(st.session_state[input_key])
        st.session_state.input_counter += 1
        st.session_state.current_input = ""
        st.session_state.submitted = False
        st.rerun()

elif st.session_state.step == 10:
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
        st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
        st.session_state.step = 12
        st.rerun()
    st.session_state.all_recommendations = all_recommendations
    st.session_state.all_groups_for_selection = all_groups_for_selection
    st.session_state.matched_subjects = matched_subjects
    if not any(st.session_state.all_groups_for_selection.values()):
        st.session_state.messages.append(("<div class='bot-message'>Aucun groupe disponible pour sélection.</div>", True))
        st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ?<b> (Oui/Non) </b></div>", True))
        st.session_state.step = 12
    else:
        groups_message = "<div class='bot-message'>Voici les groupes recommandés pour toutes les matières :<br>"
        for subject in matched_subjects:
            groups = st.session_state.all_groups_for_selection.get(subject, [])
            if groups:
                groups_message += f"<h3>{subject}</h3>"
                for i, group in enumerate(groups, 1):
                    groups_message += f"{group['display']}<br>"
        groups_message += f"Veuillez entrer les numéros des groupes choisis pour chaque matière dans l'ordre ({', '.join(matched_subjects)}):</div>"
        st.session_state.messages.append((groups_message, True))
        st.session_state.step = 11
    st.rerun()

elif st.session_state.step == 11:
    input_key = f"input_step_11_{st.session_state.input_counter}"
    placeholder = f"Ex: 1, 2 (pour {', '.join(st.session_state.matched_subjects)})"
    response = st.text_input("Votre réponse :", key=input_key, placeholder=placeholder)
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        handle_group_selection(st.session_state[input_key])
        st.session_state.input_counter += 1
        st.session_state.current_input = ""
        st.session_state.submitted = False
        st.rerun()

elif st.session_state.step == 12:
    input_key = f"input_step_12_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder="Oui ou Non")
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        choice = st.session_state[input_key].strip().lower()
        st.session_state.messages.append((f"<div class='user-message'>{st.session_state[input_key]}</div>", False))
        if choice in ["oui", "yes"]:
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
            st.session_state.low_grade_subjects = []
            st.session_state.tariffs_by_group = {}
            st.session_state.total_tariff_base = 0
            st.rerun()
        elif choice in ["non", "no"]:
            st.session_state.messages.append(("<div class='bot-message'>Merci d'avoir utilisé le chatbot ! Au revoir.</div>", True))
            st.session_state.step = 14
            st.session_state.input_counter += 1
            st.session_state.current_input = ""
            st.session_state.submitted = False
            st.rerun()
        else:
            st.session_state.messages.append(("<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'.</div>", True))
            st.session_state.input_counter += 1
            st.session_state.current_input = ""
            st.session_state.submitted = False
            st.rerun()

elif st.session_state.step == 13:
    input_key = f"input_step_13_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder="Oui ou Non")
    
    if input_key in st.session_state:
        if st.session_state[input_key] != st.session_state.current_input:
            st.session_state.current_input = st.session_state[input_key]
            st.session_state.submitted = True

    if st.session_state.submitted:
        handle_fees_choice(st.session_state[input_key])
        st.session_state.input_counter += 1
        st.session_state.current_input = ""
        st.session_state.submitted = False
        st.rerun()