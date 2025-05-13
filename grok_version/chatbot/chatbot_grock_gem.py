import streamlit as st
import json
import chromadb
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime, timedelta
import os
import random
import re
import logging
from typing import Dict, List, Tuple

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.INFO)

# Configuration initiale
st.set_page_config(page_title="Chatbot de Recommandation de Groupes", page_icon="📚", layout="wide")

# Configuration de l'API Gemini
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = "AIzaSyCMwtOAXM70sUEx_6x8Fb-Y_p8D-QRS-Og"
    
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Erreur de configuration de l'API Gemini : {str(e)}")
    st.stop()

# Initialisation de ChromaDB
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
chroma_path = os.path.join(parent_dir, "chroma_db5")
client = chromadb.PersistentClient(path=chroma_path)
collection_groupes = client.get_collection(name="groupes_vectorises9")
collection_seances = client.get_or_create_collection(name="seances_vectorises")
collection_combinaisons = client.get_or_create_collection(name="combinaisons_vectorises")
collection_students = client.get_or_create_collection(name="students_vectorises")
students_list = collection_students.get(include=["metadatas"])

# Définition de la structure attendue pour un groupe
GROUP_STRUCTURE = {
    "id_cours": str,
    "name_cours": str,
    "centre": str,
    "heure_debut": str,
    "heure_fin": str,
    "jour": str,
    "matiere": str,
    "id_forfait": str,
    "nom_forfait": str,
    "type_duree_id": str,
    "num_students": int,
    "teacher": str,
    "schools": list,
    "students": list,
    "date_debut": str,
    "date_fin": str
}

def validate_group_structure(group):
    """Valide qu'un groupe a bien la structure attendue"""
    if not isinstance(group, dict):
        return False, "Le groupe doit être un dictionnaire"
    
    missing_keys = [key for key in GROUP_STRUCTURE if key not in group]
    if missing_keys:
        return False, f"Clés manquantes: {missing_keys}"
    
    for key, expected_type in GROUP_STRUCTURE.items():
        if not isinstance(group.get(key), expected_type):
            return False, f"Type incorrect pour {key} (attendu: {expected_type.__name__})"
    
    return True, ""

# Vérification des collections
if collection_groupes.count() == 0:
    st.error("Erreur : La collection ChromaDB des groupes est vide. Veuillez exécuter la vectorisation d'abord.")
    st.stop()
if collection_combinaisons.count() == 0:
    st.error("Erreur : La collection ChromaDB des combinaisons est vide. Veuillez exécuter la vectorisation des combinaisons d'abord.")
    st.stop()

# Vérification des données pour le niveau et les matières
test_level = "BL - 2bac sc PC"
test_subjects = ["Français", "Mathématiques"]
all_groups = collection_groupes.get(include=["metadatas"])
available_levels = set(metadata.get('niveau', '').strip().lower() for metadata in all_groups['metadatas'])
available_subjects = set(metadata.get('matiere', '').strip().lower() for metadata in all_groups['metadatas'])

# Fonction pour charger le modèle SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Récupérer les valeurs uniques depuis ChromaDB
schools = set()
levels = set()
subjects = set()
centers = set()
teachers = set()
forfaits = set()
for metadata in all_groups['metadatas']:
    try:
        schools.add(metadata['ecole'].split(", ")[0])
        levels.add(metadata['niveau'])
        subjects.add(metadata['matiere'])
        centers.add(metadata['centre'])
        teachers.add(metadata['teacher'])
        forfaits.add(metadata['id_forfait'])
    except (KeyError, IndexError):
        continue

schools_list = sorted(list(schools))
levels_list = sorted(list(levels))
subjects_list = sorted(list(subjects))
centers_list = sorted(list(centers))
teachers_list = sorted(list(teachers))
forfaits_list = sorted(list(forfaits))

# Fonctions utilitaires
def get_available_forfaits(level, subject):
    logger.debug(f"Récupération des forfaits pour niveau: '{level}', matière: '{subject}'")
    forfaits = {}
    all_groups = collection_groupes.get(include=["metadatas"])
    target_subject = subject.strip().lower()
    target_level = level.strip().lower()
    available_subjects = list(set(metadata['matiere'].strip().lower() for metadata in all_groups['metadatas'] if metadata.get('matiere')))
    
    matched_subject = target_subject
    if available_subjects:
        best_match, score = process.extractOne(target_subject, available_subjects)
        if score > 80:
            matched_subject = best_match
            logger.debug(f"Matière '{subject}' correspond à '{matched_subject}' (score: {score})")
        else:
            logger.debug(f"Aucune correspondance proche pour '{subject}' (meilleur score: {score})")
    
    for metadata in all_groups['metadatas']:
        metadata_niveau = metadata.get('niveau', '').strip().lower()
        metadata_matiere = metadata.get('matiere', '').strip().lower()
        if metadata_niveau == target_level and metadata_matiere == matched_subject:
            id_forfait = metadata.get('id_forfait')
            nom_forfait = metadata.get('nom_forfait', 'Forfait inconnu')
            duree_tarifs = metadata.get('duree_tarifs', '')
            if id_forfait:
                if id_forfait not in forfaits:
                    forfaits[id_forfait] = {'name': nom_forfait, 'types_duree': {}}
                if duree_tarifs:
                    try:
                        duree_entries = duree_tarifs.split(';')
                        for i, entry in enumerate(duree_entries, 1):
                            if entry:
                                parts = entry.split(':')
                                if len(parts) == 3:
                                    type_duree, entry_id_forfait, tarif = parts
                                    if entry_id_forfait == id_forfait:
                                        type_duree_id = f"{id_forfait}_{i}"
                                        forfaits[id_forfait]['types_duree'][type_duree_id] = {
                                            'name': type_duree,
                                            'tarif_unitaire': float(tarif)
                                        }
                        logger.debug(f"Forfait trouvé: {id_forfait} - {nom_forfait}, types_duree: {forfaits[id_forfait]['types_duree']}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Erreur lors du parsing de duree_tarifs pour {id_forfait}: {str(e)}")
            else:
                logger.debug(f"Forfait ignoré: id_forfait manquant pour {metadata}")
    
    if not forfaits:
        logger.warning(f"Aucun forfait trouvé pour niveau: '{level}', matière: '{subject}'")
    return forfaits

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
    try:
        return datetime.strptime(time_str, "%H:%M")
    except:
        return None

def has_overlap(group1, group2):
    """Vérifie si deux groupes ont un chevauchement horaire"""
    # Validation des structures
    valid1, msg1 = validate_group_structure(group1)
    valid2, msg2 = validate_group_structure(group2)
    
    if not valid1 or not valid2:
        logger.error(f"Structure de groupe invalide: {msg1 if not valid1 else msg2}")
        return False
    
    # Vérification des jours différents
    if group1['jour'] != group2['jour']:
        return False
        
    try:
        # Conversion des heures
        start1 = parse_time(group1['heure_debut'])
        end1 = parse_time(group1['heure_fin'])
        start2 = parse_time(group2['heure_debut'])
        end2 = parse_time(group2['heure_fin'])
    except Exception as e:
        logger.error(f"Erreur de parsing des heures: {str(e)}")
        return False

    if not all([start1, end1, start2, end2]):
        return False
        
    # Marge de 15 min si centres différents
    same_center = group1['centre'] == group2['centre']
    margin = timedelta(minutes=15) if not same_center else timedelta(0)
    
    return (start1 < end2 + margin) and (start2 < end1 + margin)

def check_overlaps(selected_groups):
    """Vérifie les chevauchements entre les groupes sélectionnés"""
    if not isinstance(selected_groups, dict):
        logger.error("Les groupes sélectionnés doivent être dans un dictionnaire")
        return []
    
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
    return sum(1 for metadata in relevant_seances if datetime.strptime(metadata['date_seance'], "%Y/%m/%d") > reference_date_fixed)

def calculate_tariffs(selected_groups, user_duree_types, user_type_duree_ids, forfaits_info):
    tariffs_by_group = {}
    total_tariff_base = 0
    reduction_applied = 0
    reduction_description = ""
    selected_forfait_ids = []

    for subject, group in selected_groups.items():
        user_duree_type = user_duree_types[subject]
        user_type_duree_id = user_type_duree_ids[subject]
        id_forfait = group['id_forfait']
        nom_forfait = forfaits_info[subject][id_forfait]['name']
        groupe_data = collection_groupes.get(ids=[group['id_cours']], include=["metadatas"])
        if not groupe_data['metadatas']:
            return None, f"Erreur : Données non trouvées pour le cours {group['id_cours']}.", None
        
        metadata = groupe_data['metadatas'][0]
        group_id_forfait = metadata.get('id_forfait')
        type_duree_id = metadata.get('type_duree_id')
        tarif_unitaire = metadata.get('tarifunitaire')
        
        if group_id_forfait is None or type_duree_id != user_type_duree_id or tarif_unitaire is None:
            return None, f"Erreur : Données invalides pour le cours {group['id_cours']}.", None
        
        selected_forfait_ids.append(id_forfait)
        remaining_sessions = get_remaining_sessions(group['id_cours'])
        tarif_total = remaining_sessions * float(tarif_unitaire)
        
        tariffs_by_group[subject] = {
            "id_cours": group['id_cours'],
            "id_forfait": id_forfait,
            "nom_forfait": nom_forfait,
            "remaining_sessions": remaining_sessions,
            "tarif_unitaire": float(tarif_unitaire),
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

    tariff_message = "<b>Détails des tarifs :</b><br>"
    for subject, info in tariffs_by_group.items():
        tariff_message += f"- {subject} ([{info['id_forfait']}] {info['nom_forfait']}, Type de durée : {user_duree_types[subject]}) : {info['remaining_sessions']} séances restantes, tarif unitaire {info['tarif_unitaire']} DH, tarif total {info['tarif_total']:.2f} DH<br>"
    tariff_message += f"<b>Total de base :</b> {sum(info['tarif_total'] for info in tariffs_by_group.values()):.2f} DH<br>"
    if reduction_applied > 0:
        tariff_message += f"{reduction_description}<br>"
        tariff_message += f"<b>Total après réduction :</b> {total_tariff_base:.2f} DH"
    else:
        tariff_message += f"<b>Total :</b> {total_tariff_base:.2f} DH"
    
    return tariffs_by_group, tariff_message, total_tariff_base

def get_recommendations(student_name, user_level, user_subjects, user_teachers, user_school, user_center, selected_forfaits, selected_types_duree, forfaits_info):
    import random
    logger.debug(f"get_recommendations: inputs: student_name={student_name}, level={user_level}, subjects={user_subjects}, "
                 f"teachers={user_teachers}, school={user_school}, center={user_center}, forfaits={selected_forfaits}, "
                 f"types_duree={selected_types_duree}")

    # Initialize outputs
    output = []
    all_recommendations = {}
    all_groups_for_selection = {}

    # Match inputs
    matched_level = match_value(user_level, levels)[0]
    matched_subjects = [match_value(subj.strip(), subjects)[0] for subj in user_subjects.split(",")]
    
    # Handle user_teachers as string, list, or None
    if isinstance(user_teachers, list):
        matched_teachers = [teacher.strip() for teacher in user_teachers if teacher] if user_teachers else [None] * len(matched_subjects)
    elif isinstance(user_teachers, str) and user_teachers:
        matched_teachers = [teacher.strip() for teacher in user_teachers.split(",")]
    else:
        matched_teachers = [None] * len(matched_subjects)
    
    matched_school = match_value(user_school, schools)[0]
    matched_center = match_value(user_center, centers)[0] if user_center else None

    # Initialize dictionaries for each subject
    for subject in matched_subjects:
        all_recommendations[subject] = []
        all_groups_for_selection[subject] = []

    # Fetch groups from database
    all_groups_data = collection_groupes.get(include=["metadatas", "documents"])
    logger.debug(f"get_recommendations: raw groups fetched: {all_groups_data}")

    # Define required and optional metadata keys
    required_keys = ['id_cours', 'name_cours', 'id_forfait', 'type_duree_id', 'centre', 'heure_debut', 'heure_fin', 'jour', 'matiere', 'niveau', 'nom_forfait']
    optional_keys = {
        'num_students': 0,
        'teacher': 'N/A',
        'ecole': 'Inconnu',
        'student': '',
        'date_debut': 'N/A',
        'date_fin': 'N/A',
        'criteria': 'Non spécifié'
    }

    for matched_subject, matched_teacher in zip(matched_subjects, matched_teachers):
        id_forfait = selected_forfaits.get(matched_subject)
        type_duree_id = selected_types_duree.get(matched_subject)
        if not id_forfait or not type_duree_id:
            output.append(f"Aucun forfait ou type de durée sélectionné pour {matched_subject}.")
            continue

        groups = {}
        rejected_groups = []
        for metadata, document in zip(all_groups_data['metadatas'], all_groups_data['documents']):
            # Validate required keys
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                logger.error(f"Invalid group (id_cours: {metadata.get('id_cours', 'unknown')}): Missing required keys: {missing_keys}")
                rejected_groups.append((metadata, f"Missing required keys: {missing_keys}"))
                continue

            # Match group criteria
            if (metadata['niveau'].strip().lower() == matched_level.strip().lower() and
                metadata['matiere'].strip().lower() == matched_subject.strip().lower() and
                metadata.get('id_forfait') == id_forfait and
                metadata.get('type_duree_id') == type_duree_id):
                
                # Create validated group with defaults
                validated_metadata = metadata.copy()
                for key, default in optional_keys.items():
                    validated_metadata[key] = metadata.get(key, default)

                # Process schools and students
                group_schools = [school.strip() for school in validated_metadata['ecole'].split(", ") if school.strip()] or ['Inconnu']
                group_students = [student.strip() for student in validated_metadata.get('student', '').split(", ") if student.strip()] or []
                num_students = int(validated_metadata['num_students'])

                if len(group_students) < num_students:
                    group_students.extend([f"Étudiant_{i}" for i in range(len(group_students), num_students)])
                group_students = group_students[:num_students]

                unique_schools = list(dict.fromkeys(group_schools))
                if len(unique_schools) < num_students:
                    unique_schools.extend(["Inconnu"] * (num_students - len(unique_schools)))
                unique_schools = unique_schools[:num_students]

                school_student_pairs = dict(zip(group_students, unique_schools))

                # Apply filters
                if matched_center and validated_metadata['centre'].lower() != matched_center.lower():
                    logger.debug(f"Group {validated_metadata['id_cours']} filtered out: centre {validated_metadata['centre']} != {matched_center}")
                    rejected_groups.append((metadata, f"Centre mismatch: {validated_metadata['centre']}"))
                    continue
                if matched_teacher and matched_teacher != 'N/A' and validated_metadata['teacher'] != matched_teacher:
                    logger.debug(f"Group {validated_metadata['id_cours']} filtered out: teacher {validated_metadata['teacher']} != {matched_teacher}")
                    rejected_groups.append((metadata, f"Teacher mismatch: {validated_metadata['teacher']}"))
                    continue
                # Relax school filter for testing
                # if matched_school and matched_school not in group_schools:
                #     logger.debug(f"Group {validated_metadata['id_cours']} filtered out: school {matched_school} not in {group_schools}")
                #     rejected_groups.append((metadata, f"School mismatch: {matched_school}"))
                #     continue

                groups[validated_metadata['id_cours']] = {
                    "id_cours": validated_metadata['id_cours'],
                    "name_cours": validated_metadata['name_cours'],
                    "num_students": num_students,
                    "description": document,
                    "centre": validated_metadata['centre'],
                    "teacher": validated_metadata['teacher'],
                    "schools": list(school_student_pairs.values()),
                    "students": list(school_student_pairs.keys()),
                    "date_debut": validated_metadata['date_debut'],
                    "date_fin": validated_metadata['date_fin'],
                    "heure_debut": validated_metadata['heure_debut'],
                    "heure_fin": validated_metadata['heure_fin'],
                    "jour": validated_metadata['jour'] if validated_metadata.get('jour') else "None",
                    "niveau": validated_metadata['niveau'],
                    "matiere": validated_metadata['matiere'],
                    "id_forfait": validated_metadata['id_forfait'],
                    "nom_forfait": validated_metadata['nom_forfait'],
                    "type_duree_id": validated_metadata['type_duree_id']
                }

        if not groups:
            output.append(f"Aucun groupe trouvé pour {matched_subject} avec le forfait [{id_forfait}] {forfaits_info[matched_subject][id_forfait]['name']} et le type de durée sélectionné.")
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

        # Simplified filtering logic
        if matched_center:
            # Priority 1: Match teacher, center, and school
            priority_groups = [g for g in group_list if 
                             (not matched_teacher or matched_teacher == g['teacher']) and 
                             g['centre'].lower() == matched_center.lower()]
                             # and matched_school in g['schools']  # Relaxed school filter
            priority_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_groups, "Professeur, Centre, École")

            # Priority 2: Match center
            if len(selected_groups) < 3:
                remaining_groups = [g for g in group_list if g['centre'].lower() == matched_center.lower() and g['id_cours'] not in selected_ids]
                add_groups(remaining_groups, "Centre")

        # Fallback: No center specified
        else:
            # Priority 1: Match teacher and school
            priority_groups = [g for g in group_list if 
                             (not matched_teacher or matched_teacher == g['teacher'])]
                             # and matched_school in g['schools']  # Relaxed school filter
            priority_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_groups, "Professeur, École")

            # Priority 2: Any groups
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
                f"<b>Forfait:</b> [{group['id_forfait']}] {group['nom_forfait']}<br>"
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
                "criteria": group['criteria'],
                "id_forfait": group['id_forfait'],
                "nom_forfait": group['nom_forfait'],
                "type_duree_id": group['type_duree_id']
            })
        
        all_recommendations[matched_subject] = recommendations
        all_groups_for_selection[matched_subject] = groups_for_selection

    logger.debug(f"get_recommendations: all_groups_for_selection: {all_groups_for_selection}")
    logger.debug(f"get_recommendations: all_recommendations: {all_recommendations}")
    logger.debug(f"get_recommendations: rejected groups: {rejected_groups}")

    output.append(f"<b>Les groupes recommandés pour l'étudiant</b> {student_name} :")
    return output, all_recommendations, all_groups_for_selection, matched_subjects

# Styles CSS
st.markdown("""
    <style>
    .stApp { background-color: #808080; }
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
    .container { overflow: hidden; }
    h4 { color: #00796b; margin-bottom: 5px; }
    b { color: #004d40; }
    div[data-testid="stTextInput"] { margin-bottom: 0 !important; padding-bottom: 0 !important; }
    div[data-testid="stTextInput"] > div { margin-bottom: 0 !important; padding-bottom: 0 !important; }
    .profile-name { text-align: center; font-size: 25px; color: white; margin-top: 10px; margin-left: 5px; }
    </style>
""", unsafe_allow_html=True)

# Fonction de traitement avec Gemini
def process_with_llm(input_text, step, session_state, lists):
    try:
        prompt = f"""
        **Contexte**:
        Vous êtes un chatbot Streamlit de recommandation de groupes éducatifs, aidant les utilisateurs à trouver des groupes d’apprentissage adaptés (niveau, matière, professeur, école, centre, forfait, type de durée) et à calculer les tarifs avec ou sans réductions. Les données sont extraites d’une base ChromaDB. Le flux conversationnel comporte 15 étapes à respecter qui doivent être dynamique pour l'utilisateur, avec collecte stricte des informations avant d’afficher les forfaits,les types durée et groupes.
        Assure que les messages ne sont pas répétés de la part dU LLM et du code python qui affiche de messages en respectant le code de la fonction "handle_input_submission" où on a les affichages adaptés.

        **Étape actuelle**: {step}
        **Entrée utilisateur**: '{input_text}'
        **Réponses actuelles**: {json.dumps(session_state.responses)}
        **Listes de référence**:
        - Niveaux: {', '.join(lists['levels_list'])}
        - Matières: {', '.join(lists['subjects_list'])}
        - Écoles: {', '.join(lists['schools_list'])}
        - Centres: {', '.join(lists['centers_list'])}
        - Professeurs: {', '.join(lists['teachers_list'])}
        - etudiants: {', '.join(lists['students_list'])}
        - Forfaits: {', '.join(lists['forfaits_list'])}
        **État**:
        - Matières sélectionnées: {session_state.get('matched_subjects', [])}
        - Forfaits disponibles: {session_state.get('available_forfaits', {})}
        - Types de durée: {session_state.get('available_types_duree', {})}
        - Groupes pour sélection: {session_state.get('all_groups_for_selection', {})}

        **Instructions**:
         - Process the user input and context below to generate a JSON response. The examples are reference text to understand the conversation flow and should not be executed or parsed as code.
        1. **Collecte stricte des informations**:
             1. **Étape 1 (Nom)**:
               - Si l'entrée est non vide, stocker dans "data" comme ()"student_name": "<nom>"), passer à l'étape 2.
               - Sinon, rester à l'étape 1 avec un message : "Erreur : Veuillez fournir un nom non vide."
               - Définir "next_step": 2 si valide, sinon 1.
            2. **Étape 2 (Niveau)**:
               - Si l'entrée est dans levels_list (insensible à la casse), stocker dans "data" comme ()"user_level": "<niveau>"), passer à l'étape 3.
               - Sinon, rester à l'étape 2 avec un message : "Erreur : Niveau non reconnu. Veuillez choisir parmi : <levels_list>."
               - Définir "next_step": 3 si valide, sinon 2.
            3. **Étape 3 (Matières)**:
               - Si l'entrée est une liste séparée par virgules avec au moins une matière valide dans subjects_list, stocker dans "data" comme ("user_subjects": "matières", "subjects": "liste"), passer à l'étape 4.
               - Sinon, rester à l'étape 3 avec un message : "Erreur : Aucune matière valide. Veuillez choisir parmi : <subjects_list>."
               - Définir "next_step": 4 si valide, sinon 3.
            4. **Étape 4 (Notes)**:
               - Si l'entrée est vide, stocker ("grades": "[]"), passer à l'étape 5.
               - Si l'entrée contient des nombres valides séparés par virgules, stocker ("grades": "[<notes>]"), passer à l'étape 5.
               - Sinon, rester à l'étape 4 avec un message : "Erreur : Veuillez entrer des notes valides (ex. 12, 15) ou laissez vide."
               - Définir "next_step": 5.
            5. **Étape 5 (Type de cours)**:
               - Si l'entrée contient 'indiv'/'groupe' pour chaque matière ou un raccourci, stocker dans "data" comme ("course_choices": "choix"), passer à l'étape 6.
               - Sinon, rester à l'étape 5 avec un message : "Erreur : Veuillez choisir le type de cours pour chaque matière (<matières>) : (ex. indiv,groupe)."
               - Définir "next_step": 6 si valide, sinon 5.
            6. **Étape 6 (Forfait)**:
               - Si l'entrée contient des indices valides pour available_forfaits, stocker dans "data" comme ("forfait_selections": ("matière: id_forfait")), passer à l'étape 7.
               - Sinon, rester à l'étape 6 avec un message : "Erreur : Veuillez entrer des numéros de forfaits valides (ex. 1,2)."
               - Définir "next_step": 7 si valide, sinon 6.
            7. **Étape 7 (Type de durée)**:
               - Si l'entrée contient des indices valides pour available_types_duree, stocker dans "data" comme ("type_duree_selections": ("matière": "type_id")), passer à l'étape 8.
               - Sinon, rester à l'étape 7 avec un message : "Erreur : Veuillez entrer des numéros de types de durée valides (ex. 1,2)."
               - Définir "next_step": 8 si valide, sinon 7.
            8. **Étape 8 (Professeurs)**:
               - Stocker dans "data" comme ("user_teachers": "<noms ou vide>"), passer à l'étape 9.
               - Message : "Parfait ! J'ai enregistré les professeurs <noms ou 'non spécifiés'>. Quelle est l'école de l'étudiant ?"
               - Définir "next_step": 9.
            9. **Étape 9 (École)**:
               - Si l'entrée est dans schools_list (insensible à la casse), stocker dans "data" comme ("user_school": "<école>"), passer à l'étape 10.
               - Sinon, rester à l'étape 9 avec un message : "Erreur : École non reconnue. Veuillez choisir parmi : <schools_list>."
               - Définir "next_step": 10 si valide, sinon 9.
            10. **Étape 10 (Centre)**:
               - Si l'entrée est vide ou dans centers_list (insensible à la casse), stocker dans "data" comme ("user_center": "<centre ou vide>"), passer à l'étape 11.
               - Sinon, rester à l'étape 10 avec un message : "Erreur : Centre non reconnu. Veuillez choisir parmi : <centers_list> ou laissez vide."
               - Définir "next_step": 11 si valide, sinon 10.
               - On affiche la liste des groupes puis on demande à l'utilisateur de choisir les groupes souhaités.
            11. **Étape 11 (Groupes)**:
               - Si l'entrée contient des indices valides pour all_groups_for_selection, stocker dans "data" comme ("group_selections": ("matière": "id_cours")), passer à l'étape 12.
               - Vérifier que tous les attributs requis sont présents : student_name, user_level, user_subjects, user_school, user_center, forfait_selections, type_duree_selections.
               - Sinon, rester à l'étape 11 avec un message : "Erreur : Veuillez entrer des numéros de groupes valides (ex. 1,2)."
               - Définir "next_step": 12 si valide, sinon 11.
            12. **Étape 12 (Frais)**:
               - Si l'entrée est 'Oui'/'Non', stocker dans "data" comme ("include_fees": "true/false"), passer à l'étape 13.
               - Sinon, rester à l'étape 12 avec un message : "Veuillez répondre par 'Oui' ou 'Non'."
               - Définir "next_step": 13.
            13. **Étape 13 (Commentaires)**:
               - Stocker dans "data" comme ("comments": "<commentaires>").
               - Si "réduction" est détecté avec "<pourcentage>%" (ex. "10%"), stocker ("reduction_percentage": "pourcentage"), passer à l'étape 15 avec message (sachant que les les mots qui sont entre <> sont des variables , tu les remplace par leurs valeurs ): "Commentaires enregistrés : \nTotal de base : <montant> DH\nRéduction de <pourcentage>% : -<montant> DH\nTotal final : <montant> DH\nVoulez-vous traiter un autre cas ? (Oui/Non)"
               - Si "réduction" est détecté sans pourcentage, passer à l'étape 14 avec message : "Veuillez préciser le pourcentage de réduction :"
               - Sinon, passer à l'étape 15 avec message : "Commentaires enregistrés : <commentaires ou 'Aucun'>\nTotal final : <montant> DH\nVoulez-vous traiter un autre cas ? (Oui/Non)"
               - Définir "next_step": 15 ou 14 selon le cas.
            14. **Étape 14 (Pourcentage)**:
               - Si l'entrée est un nombre entre 0 et 100, stocker dans "data" comme ("reduction_percentage": "pourcentage"), passer à l'étape 15 avec message humanisé proche de : "Total de base : <montant> DH\nRéduction de <pourcentage>% : -<montant> DH\nTotal final : <montant> DH\nVoulez-vous traiter un autre cas ? (Oui/Non)"
               - Sinon, rester à l'étape 14 avec un message : "Erreur : Veuillez entrer un nombre valide (ex. 10)."
               - Définir "next_step": 15 si valide, sinon 14.
            15. **Étape 15 (Autre cas)**:
               - Si l'entrée est 'Oui'/'Non', stocker dans "data" comme ("another_case": "true/false"), passer à l'étape 1 (si Oui) ou 0 (si Non).
               - Sinon, rester à l'étape 15 avec un message : "Veuillez répondre par 'Oui' ou 'Non'."
               - Définir "next_step": 1 (Oui), 0 (Non), ou 15 si invalide.
           - Si données manquantes ou invalides, redemander avec message clair et rester à l'étape actuelle.
        2. **Traitement des notes (étape 4)**:
           - Si note < 8: "Note faible en [matière] ([note]/20). Nous recommandons des cours individuels pour une mise à niveau efficace."
           - Si note ≥ 8 ou absente: "Note ([note]) ou aucune note. Nous recommandons des cours individuels pour un enseignement personnalisé."
           - Proposer: "Veuillez choisir le type de cours pour chaque matière ([matières]) : (ex. indiv,groupe ou 'je veux groupe pour toutes les matières')."
           - Passer à l'étape 5.
        3. **Choix groupe/indiv (étape 5)**:
           - Accepter:
             - Liste explicite (ex. 'groupe,indiv' / 'groupe,groupe' / 'indiv,groupe') correspondant aux matières dans matched_subjects.
             - Raccourcis comme 'je veux groupe pour toutes les matières' ou 'je veux groupe pour les deux matières' (appliquer 'groupe' à toutes les matières).
             - Raccourcis comme 'je veux indiv pour toutes les matières' (appliquer 'indiv' à toutes).
           - Normaliser chaque choix avec match_value ('groupe'/'indiv', score > 80).
           - Valider que le nombre de choix correspond au nombre de matières dans matched_subjects.
           - Si invalide (ex. nombre incorrect, choix non reconnu), redemander avec liste des matières et rester à l'étape 5.
           - Stocker dans "course_choices" (ex. ["groupe", "groupe"]).
           - Si au moins une matière est en 'groupe', passer à l'étape 6 avec message: "Vous avez choisi des cours en groupe pour [matières]. Les forfaits seront affichés à l'étape suivante."
           - Si aucune matière en 'groupe', passer à l'étape 15 avec message: "Aucune matière sélectionnée pour des cours en groupe. Voulez-vous traiter un autre cas ? (Oui/Non)"
        4. **Forfaits et types de durée (étape 6 et 7)**:
           - À l’étape 6, valider les indices de forfaits (ex. '1,2') contre available_forfaits pour chaque matière dans matched_subjects.
           - Convertir les indices en id_forfait (ex. available_forfaits[subject][list(available_forfaits[subject].keys())[index-1]]).
           - Stocker dans "forfait_selections" comme un dictionnaire {{matière: id_forfait}} (ex. {{"Physique - Chimie": "12677992", "Mathématiques": "12678012"}}).
           - Si indices invalides (ex. hors plage, non numériques), redemander avec liste des forfaits et rester à l'étape 6.
           - Passer à l’étape 7 avec message: "Veuillez choisir le type de durée pour chaque forfait : <liste des forfaits choisis>."
           - À l’étape 7, valider les choix de types de durée (indices valides dans available_types_duree). Stocker dans "type_duree_selections" comme un dictionnaire {{matière: type_duree_id}}.
        5. **Données manquantes**:
           - Avant étape 5, vérifier: 'student_name', 'user_level', 'user_subjects', 'matched_subjects'.
           - Avant étape 6, vérifier: 'matched_subjects', 'available_forfaits'.
           - Avant étape 10, vérifier: 'student_name', 'user_level', 'user_subjects', 'user_school', 'selected_forfaits', 'selected_types_duree'.
           - Professeurs et centre sont optionnels (accepter vide) mais il faut les demander avant de recommander les groupes.
           - Si données manquantes, revenir à l’étape correspondante avec message clair (ex: on demande tous les attributs avant de recommender les groupes).
        6. **Retours et modifications**:
           - Détecter 'revenir à [étape]' (ex. 'revenir aux matières' → étape 3) applique cela sur tous les champs .
           - Détecter 'changer [champ]' (ex. 'changer les matières' → étape 3, effacer réponses récentes), applique cela ,sur tous les champs.
           - Si 'changer' ou 'revenir' détecté, redemander la valeur correspondante et revenir vers les étapes qui se basent sur le champs modifié (ex. si on a modifié les notes on doit revenir vers l'étape après les notes c-à-d la réinterprétation des notes et on continue le flux).
           - Si 'changer' ou 'revenir' non détecté, passer à l’étape suivante.
           - si l'utilisateur dit 'je veux changer le nom de l'étudiant' ou 'je veux changer le nom de l'étudiant à Ahmed', on doit revenir à l'étape 1 et ecraser le champs 'student_name' avec la nouvelle valeur en restant à l'étape actuel.
           - si l'utilisateur veut modifier les matières ,on doit revenir à l'étape 3 et ecraser le champs 'user_subjects' avec la nouvelle valeur en restant à l'étape actuel pour bien vérifier et continuer le traitement.
        7. **Réductions**:
           - À l’étape 13, après l'étape des frais d'inscription, poser une question demandant des commentaires : "Y a-t-il des commentaires ou demandes supplémentaires ?"
           - Si l'entrée contient 'réduction', 'remise', 'rabais', 'discount' (score > 90):
             - Extraire tout pourcentage explicite (ex. "10%", "10", "dix pourcent") en utilisant une analyse sémantique ou regex (ex. ) ou reconnaissance de nombres en mots comme "dix", "vingt").
             - Si un pourcentage valide (0-100) est trouvé:
               - Stocker dans "data" comme ("reduction_percentage: nombre").
               - Passer à l’étape 15 avec un message confirmant la réduction : "Réduction de <pourcentage>% enregistrée. Le total réduit sera calculé. Voulez-vous traiter un autre cas ? (Oui/Non)"
               - Définir "next_step": 15.
             - Si aucun pourcentage n'est trouvé:
               - Passer à l’étape 14 avec un message demandant le pourcentage : "Quel est le pourcentage de réduction que vous souhaitez appliquer ? (Entre 0 et 100)"
               - Définir "next_step": 14.
           - Si aucune réduction n'est détectée:
             - Passer à l’étape 15 avec un message affichant le total final : "Total final : <montant> DH. Voulez-vous traiter un autre cas ? (Oui/Non)"
             - Définir "next_step": 15.
        8. **Étape 14 (Pourcentage de réduction)**:
           - Si l'entrée est un nombre entre 0 et 100 ou contient un pourcentage valide dans une phrase (ex. "je veux 10%"):
             - Extraire le pourcentage avec regex ou analyse sémantique.
             - Si valide (0-100):
               - Stocker dans "data" comme ("reduction_percentage: nombre").
               - Passer à l’étape 15 avec un message confirmant : "Réduction de <pourcentage>% enregistrée. Le total réduit sera calculé. Voulez-vous traiter un autre cas ? (Oui/Non)"
               - Définir "next_step": 15.
             - Si invalide:
               - Rester à l’étape 14 avec un message d'erreur : "Erreur : Veuillez entrer un nombre valide entre 0 et 100 (ex. 10)."
               - Définir "next_step": 14.
           - Ne jamais revenir à une étape antérieure (ex. étape 11 ou 12) ni demander les frais d'inscription à l'étape 14.
        9. **Réponse JSON**:
           - Format:
             ```json
             {{
               "step": <int>,
               "data": {{}},
               "message": "<string>",
               "error": "<string|null>",
               "suggestions": [],
               "next_step": <int>
             }}
             ```
           - Inclure données validées dans 'data' (ex. {{"forfait_selections": {{"Physique - Chimie": "12677992", "Mathématiques": "12678012"}}}}).
           - Fournir message naturel et amical , adapté au nom de l’étudiant (pas la peine de le mentionner dans toutes les étapes).
           - Si erreur, inclure dans "error" et rester à l'étape actuelle.

        **Étapes et attentes**:
        1. Nom: Texte non vide → "student_name".
        2. Niveau: Valider avec levels_list → "user_level".
        3. Matières: Liste séparée par virgules, valider avec subjects_list → "user_subjects", "subjects".
        4. Notes: Nombres ou vide, proposer cours individuels → "grades".
        5. Type de cours: 'indiv'/'groupe' par matière ou raccourci → "course_choices".
        6. Forfait: Indices valides dans available_forfaits, produire un dictionnaire {{matière: id_forfait}} → "forfait_selections".
        7. Type de durée: Indices valides dans available_types_duree → "type_duree_selections".
        8. Professeurs: Texte ou vide → "user_teachers".
        9. École: Valider avec schools_list → "user_school".
        10. Centre: Valider avec centers_list ou vide → "user_center".
        11. Groupes: Indices valides dans "all_groups_for_selection" → "group_selections" avec vérification de présence de tous les attributs qui sont requis avant cette étape.
        12. Frais: 'Oui'/'Non' → "include_fees".
        13. Commentaires: Détecter 'réduction' ou 'avec 10%' → "comments","reduction_percentage" et on ne demande pas le pourcentage dans l'étape 13 , on passe au calcul directement.
        14. Pourcentage:si réduction détectée dans l'étape 13 on demande un Nombre entre 0 et 100 → "reduction_percentage".
        15. Autre cas: 'Oui'/'Non' → "another_case".

        **Exemples**:
        - Étape 1, Entrée: "Ahmed larache" → {{"step": 1, "data": {{"student_name": "Ahmed larache"}}, "message": "Bonjour, Ahmed larache ! Quel est le niveau de l’étudiant ?", "error": null, "suggestions": [], "next_step": 2}}
        - Étape 2, Entrée: "BL - 2bac sc PC" → {{"step": 2, "data": {{"user_level": "BL - 2bac sc PC"}}, "message": "Niveau BL - 2bac sc PC validé. Quelles sont les matières ?", "error": null, "suggestions": [], "next_step": 3}}
        - Étape 3, Entrée: "Physique - Chimie,Mathématiques" → {{"step": 3, "data": {{"user_subjects": "Physique - Chimie,Mathématiques", "subjects": ["Physique - Chimie", "Mathématiques"]}}, "message": "Matières validées. Quelles sont les notes ?", "error": null, "suggestions": [], "next_step": 4}}
        - Étape 4, Entrée: "7,12" → {{"step": 4, "data": {{"grades": [7, 12]}}, "message": "Note faible en Physique - Chimie (7/20). Nous recommandons des cours individuels... Veuillez choisir le type de cours pour chaque matière (Physique - Chimie, Mathématiques) : (ex. indiv,groupe)", "error": null, "suggestions": [], "next_step": 5}}
        - Étape 5, Entrée: "groupe,groupe" → {{"step": 5, "data": {{"course_choices": ["groupe", "groupe"]}}, "message": "Vous avez choisi des cours en groupe pour Physique - Chimie et Mathématiques. Les forfaits seront affichés à l'étape suivante.", "error": null, "suggestions": [], "next_step": 6}}
        - Étape 6, Entrée: "1,1", Contexte: matched_subjects=["Physique - Chimie", "Mathématiques"], available_forfaits={{"Physique - Chimie": {{"12677992": {{"name": "BL-2BAC-PHYSIQUE - CHIMIE"}}}}, "Mathématiques": {{"12678012": {{"name": "BL-2BAC-MATHÉMATIQUES"}}}}}} → {{"step": 6, "data": {{"forfait_selections": {{"Physique - Chimie": "12677992", "Mathématiques": "12678012"}}}}, "message": "Forfaits validés. Veuillez choisir le type de durée pour chaque forfait.", "error": null, "suggestions": [], "next_step": 7}}
        - Étape 6, Entrée: "1,2", Contexte: matched_subjects=["Physique - Chimie", "Mathématiques"], available_forfaits={{"Physique - Chimie": {{"12677992": {{}}}}, "Mathématiques": {{"12678012": {{}}}}}} → {{"step": 6, "data": {{}}, "message": "Choix invalides. Veuillez choisir un forfait pour chaque matière (Physique - Chimie, Mathématiques) : <liste>", "error": "Indice 2 invalide pour Mathématiques", "suggestions": [], "next_step": 6}}   
        
        **Instructions pour l’étape 11**:

        1. L’entrée est une liste d’indices séparés par des virgules (ex. "1,2") correspondant aux groupes dans all_groups_for_selection pour chaque matière dans matched_subjects.
        2. Validez chaque indice:
           - Convertissez l’entrée en une liste (ex. "1,2" → ["1", "2"]).
           - Assurez-vous que la longueur de la liste correspond au nombre de matières dans matched_subjects.
           - Pour chaque matière et indice:
             - Vérifiez que l’indice est un nombre valide (ex. 1, 2, ...).
             - Vérifiez que l’indice est dans la plage des groupes disponibles pour la matière dans all_groups_for_selection[matiere] (ex. 1 ≤ indice ≤ len(all_groups_for_selection[matiere])).
           - Si valide, stockez les indices dans "group_selections" comme une liste (ex. ["1", "2"]).
        3. Si l’entrée est invalide:
           - Incluez une erreur dans "error" (ex. "Indice 2 invalide pour Physique - Chimie").
           - Fournissez un message listant les groupes disponibles pour chaque matière avec leurs indices.
           - Restez à l’étape 11.
        4. Si valide:
           - Stockez "group_selections" dans "data".
           - Passez à l’étape 12 avec un message confirmant les groupes sélectionnés (ex. "Groupes validés pour Physique - Chimie (Groupe 1) et Mathématiques (Groupe 2).").
        5. Ne retournez une liste vide ("group_selections": []) que si l’entrée est explicitement invalide.

         **Réponse**:
        Répondez avec un ton amical,humanisé et friendly ,Analysez l’entrée et fournissez une réponse JSON conforme. Assurez-vous que la réponse est cohérente avec les choix de l’utilisateur et les étapes précédentes. Évitez toute incohérence ou répétition inutile,faites attention aux questions posées et leurs réponses ,stocker les réponses en toute cohérence avec les questions posées.
        """

        gemini_response = gemini_model.generate_content(prompt)
        response_text = gemini_response.text.strip()

        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()

        try:
            response = json.loads(response_text)
            required_keys = ["step", "data", "message", "error", "suggestions", "next_step"]
            if not all(key in response for key in required_keys):
                raise ValueError("Réponse JSON incomplète")
            if response["step"] == 3 and "subjects" in response["data"]:
                session_state.matched_subjects = response["data"]["subjects"]
            logger.debug(f"Réponse Gemini : {response}")
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing de la réponse Gemini : {str(e)}")
            st.error(f"Erreur de parsing de la réponse Gemini : {str(e)}")
            return {
                "step": step,
                "data": {},
                "message": "Erreur interne, veuillez réessayer.",
                "error": "Erreur de parsing",
                "suggestions": [],
                "next_step": step
            }
        except ValueError as e:
            logger.error(f"Réponse Gemini invalide : {str(e)}")
            st.error(f"Réponse Gemini invalide : {str(e)}")
            return {
                "step": step,
                "data": {},
                "message": "Erreur interne, veuillez réessayer.",
                "error": "Réponse invalide",
                "suggestions": [],
                "next_step": step
            }
    except Exception as e:
        logger.error(f"Erreur lors de l’appel à l’API Gemini : {str(e)}")
        st.error(f"Erreur lors de l’appel à l’API Gemini : {str(e)}")
        return {
            "step": step,
            "data": {},
            "message": "Erreur interne, veuillez réessayer.",
            "error": "Erreur API",
            "suggestions": [],
            "next_step": step
        }

# Gestion des étapes avec Gemini
def handle_input_submission(step: int, response_text: str) -> None:
    logger.debug(f"Traitement de l'étape {step} avec entrée : {response_text}")
    logger.debug(f"État actuel de responses : {st.session_state.responses}")
    logger.debug(f"État actuel de matched_subjects : {st.session_state.matched_subjects}")
    
    lists = {
        "levels_list": levels_list,
        "subjects_list": subjects_list,
        "schools_list": schools_list,
        "centers_list": centers_list,
        "teachers_list": teachers_list,
        "forfaits_list": forfaits_list,
        "students_list": students_list,
    }
    llm_response = process_with_llm(response_text, step, st.session_state, lists)
    logger.debug(f"LLM response for step {step}: {llm_response}")
    
    # Mettre à jour l'état
    st.session_state.step = llm_response["next_step"]
    st.session_state.responses.update(llm_response["data"])
    
    # Ajouter le message de l'utilisateur s'il n'existe pas déjà
    user_message = f"<div class='user-message'>{response_text}</div>"
    if not any(msg[0] == user_message for msg in st.session_state.messages[-3:]):
        st.session_state.messages.append((user_message, False))
    
    # Ajouter le message du bot s'il n'existe pas déjà
    if llm_response["message"]:
        bot_message = f"<div class='bot-message'>{llm_response['message']}</div>"
        # Vérifier les doublons sémantiques selon l'étape
        is_duplicate = False
        if step == 8 and any("professeurs souhaités" in msg[0].lower() for msg in st.session_state.messages[-3:]):
            is_duplicate = True
        elif step == 9 and any("école" in msg[0].lower() and "enregistrée" in msg[0].lower() for msg in st.session_state.messages[-3:]):
            is_duplicate = True
        elif step == 10 and any("centre" in msg[0].lower() and "enregistré" in msg[0].lower() for msg in st.session_state.messages[-3:]):
            is_duplicate = True
        elif step == 13 and any("commentaires enregistrés" in msg[0].lower() or "total final" in msg[0].lower() for msg in st.session_state.messages[-3:]):
            is_duplicate = True
        elif step == 14 and any("réduction" in msg[0].lower() and "total réduit" in msg[0].lower() for msg in st.session_state.messages[-3:]):
            is_duplicate = True
        elif any(msg[0] == bot_message for msg in st.session_state.messages[-3:]):
            is_duplicate = True
        if not is_duplicate:
            st.session_state.messages.append((bot_message, True))
    
    # Ajouter un message d'erreur si nécessaire
    if llm_response.get("error"):
        error_message = f"<div class='bot-message'>Erreur : {llm_response['error']}</div>"
        if not any(msg[0] == error_message for msg in st.session_state.messages[-3:]):
            st.session_state.messages.append((error_message, True))

    # Étape 1 : Nom
    if llm_response["step"] == 1:
        name_input = response_text.strip()
        if not name_input:
            error_message = "<div class='bot-message'>Erreur : Veuillez fournir un nom non vide.</div>"
            if not any("Veuillez fournir un nom non vide" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 1
            return
        st.session_state.responses['student_name'] = name_input

    # Étape 2 : Niveau
    elif llm_response["step"] == 2:
        level_input = response_text.strip()
        if level_input.lower() not in [l.lower() for l in levels_list]:
            error_message = f"<div class='bot-message'>Erreur : Niveau non reconnu. Veuillez choisir parmi : {', '.join(levels_list)}</div>"
            if not any("Niveau non reconnu" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 2
            return
        st.session_state.responses['user_level'] = level_input

    # Étape 3 : Matières
    elif llm_response["step"] == 3:
        subjects_input = [s.strip() for s in response_text.split(',')]
        valid_subjects = [s for s in subjects_input if s.lower() in [sub.lower() for sub in subjects_list]]
        if not valid_subjects:
            error_message = f"<div class='bot-message'>Erreur : Aucune matière valide. Veuillez choisir parmi : {', '.join(subjects_list)}</div>"
            if not any("Aucune matière valide" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 3
            return
        st.session_state.responses['user_subjects'] = ', '.join(valid_subjects)
        st.session_state.matched_subjects = valid_subjects
        st.session_state.responses['subjects'] = valid_subjects

    # Étape 4 : Notes
    elif llm_response["step"] == 4:
        grades_input = response_text.strip()
        if grades_input:
            try:
                grades = [float(g.strip()) for g in grades_input.split(',')]
                st.session_state.responses['grades'] = grades
                # Proposer cours individuels si notes faibles
                if any(g < 10 for g in grades):
                    message = "<div class='bot-message'>Certaines notes sont faibles. Souhaitez-vous des cours individuels pour certaines matières ?</div>"
                    if not any("Certaines notes sont faibles" in msg[0] for msg in st.session_state.messages[-3:]):
                        st.session_state.messages.append((message, True))
            except ValueError:
                error_message = "<div class='bot-message'>Erreur : Veuillez entrer des notes valides (ex. 12, 15) ou laissez vide.</div>"
                if not any("Veuillez entrer des notes valides" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((error_message, True))
                st.session_state.step = 4
                return
        else:
            st.session_state.responses['grades'] = []

    # Étape 5 : Choix des cours
    elif llm_response["step"] == 5 and "course_choices" in llm_response["data"]:
        required_fields = ['student_name', 'user_level', 'user_subjects']
        missing_fields = [field for field in required_fields if field not in st.session_state.responses or not st.session_state.responses[field]]
        if missing_fields:
            missing_message = f"<div class='bot-message'>Données manquantes : {', '.join(missing_fields)}. Veuillez compléter les informations.</div>"
            if not any("Données manquantes" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((missing_message, True))
            if 'student_name' in missing_fields:
                st.session_state.step = 1
                st.session_state.messages.append(("<div class='bot-message'>Quel est le nom de l'étudiant !</div>", True))
            elif 'user_level' in missing_fields:
                st.session_state.step = 2
                st.session_state.messages.append(("<div class='bot-message'>Quel est votre niveau (ex. BL - 2bac sc PC) ?</div>", True))
            elif 'user_subjects' in missing_fields:
                st.session_state.step = 3
                st.session_state.messages.append(("<div class='bot-message'>Quelles sont les matières qui intéressent l'étudiant ?</div>", True))
            logger.error(f"Données manquantes à l'étape 5 : {missing_fields}")
            return
        
        course_choices = llm_response["data"]["course_choices"]
        if len(course_choices) != len(st.session_state.matched_subjects):
            error_message = f"<div class='bot-message'>Erreur : Nombre de choix ({len(course_choices)}) ne correspond pas au nombre de matières ({len(st.session_state.matched_subjects)}). Veuillez choisir le type de cours pour chaque matière ({', '.join(st.session_state.matched_subjects)}) : (ex. indiv,groupe)</div>"
            if not any("Nombre de choix" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 5
            logger.error(f"Nombre de choix incorrect : {course_choices}")
            return
        
        group_subjects = [subject for subject, choice in zip(st.session_state.matched_subjects, course_choices) if choice.lower() == "groupe"]
        indiv_subjects = [subject for subject, choice in zip(st.session_state.matched_subjects, course_choices) if choice.lower() == "indiv"]
        st.session_state.course_choices = dict(zip(st.session_state.matched_subjects, course_choices))
        logger.debug(f"Course choices : {st.session_state.course_choices}")
        logger.debug(f"Group subjects : {group_subjects}")
        logger.debug(f"Indiv subjects : {indiv_subjects}")
        
        for subject in indiv_subjects:
            if not any(f"Les cours individuels pour {subject}" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((f"<div class='bot-message'>Les cours individuels pour {subject} sont en préparation et seront disponibles ultérieurement.</div>", True))
        
        if group_subjects:
            st.session_state.matched_subjects = group_subjects
            st.session_state.responses['user_subjects'] = ', '.join(group_subjects)
            st.session_state.available_forfaits = {}
            forfaits_message = f"<div class='bot-message'>{st.session_state.responses['student_name']}, vous avez choisi des cours en groupe pour {', '.join(group_subjects)}.<br>Veuillez choisir un forfait pour chaque matière (entrez les numéros dans l'ordre : {', '.join(group_subjects)}) :<br>"
            any_forfaits = False
            for subject in group_subjects:
                forfaits = get_available_forfaits(st.session_state.responses['user_level'], subject)
                st.session_state.available_forfaits[subject] = forfaits
                forfaits_message += f"<h3>{subject}</h3>"
                if forfaits:
                    any_forfaits = True
                    for i, (id_forfait, info) in enumerate(forfaits.items(), 1):
                        forfaits_message += f"{i}. [{id_forfait}] {info['name']}<br>"
                else:
                    forfaits_message += f"Aucun forfait disponible pour {subject}.<br>"
            if any_forfaits:
                forfaits_message += f"Exemple : {'1,2' if len(group_subjects) > 1 else '1'} pour choisir le forfait pour {', '.join(group_subjects)}.</div>"
                if not any("Veuillez choisir un forfait pour chaque matière" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((forfaits_message, True))
                st.session_state.step = 6
            else:
                if not any("Aucun forfait disponible pour les matières sélectionnées" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append(("<div class='bot-message'>Aucun forfait disponible pour les matières sélectionnées. Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                st.session_state.step = 15
        else:
            if not any("Aucune matière sélectionnée pour des cours en groupe" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append(("<div class='bot-message'>Aucune matière sélectionnée pour des cours en groupe. Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
            st.session_state.step = 15

    # Étape 6 : Sélection des forfaits
    elif llm_response["step"] == 6 and "forfait_selections" in llm_response["data"]:
        st.session_state.selected_forfaits = llm_response["data"]["forfait_selections"]
        if not isinstance(st.session_state.selected_forfaits, dict):
            error_message = f"<div class='bot-message'>Erreur interne : Les sélections de forfaits doivent être au format dictionnaire. Veuillez réessayer en entrant les numéros des forfaits (ex. {'1,2' if len(st.session_state.matched_subjects) > 1 else '1'}).</div>"
            if not any("Erreur interne : Les sélections de forfaits" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 6
            logger.error(f"selected_forfaits n'est pas un dictionnaire : {st.session_state.selected_forfaits}")
            return
        
        st.session_state.available_types_duree = {}
        types_duree_message = "<div class='bot-message'>Veuillez choisir le type de durée pour chaque forfait :<br>"
        valid_subjects = []
        for subject, id_forfait in st.session_state.selected_forfaits.items():
            if subject in st.session_state.available_forfaits and id_forfait in st.session_state.available_forfaits[subject]:
                types_duree = st.session_state.available_forfaits[subject][id_forfait]['types_duree']
                st.session_state.available_types_duree[subject] = types_duree
                if types_duree:
                    valid_subjects.append(subject)
                    types_duree_message += f"<h3>{subject} ([{id_forfait}] {st.session_state.available_forfaits[subject][id_forfait]['name']})</h3>"
                    for i, (type_id, info) in enumerate(types_duree.items(), 1):
                        types_duree_message += f"{i}. {info['name']} (Tarif unitaire: {info['tarif_unitaire']} DH)<br>"
                else:
                    types_duree_message += f"<h3>{subject}</h3>Aucun type de durée disponible.<br>"
            else:
                types_duree_message += f"<h3>{subject}</h3>Forfait {id_forfait} invalide.<br>"
        if valid_subjects:
            st.session_state.matched_subjects = valid_subjects
            types_duree_message += f"Entrez les numéros des types de durée choisis pour chaque matière dans l'ordre ({', '.join(valid_subjects)}):"
            if not any("Veuillez choisir le type de durée pour chaque forfait" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((types_duree_message, True))
            st.session_state.step = 7
        else:
            if not any("Aucun type de durée disponible" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append(("<div class='bot-message'>Aucun type de durée disponible. Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
            st.session_state.step = 15

    # Étape 7 : Sélection des types de durée
    elif llm_response["step"] == 7 and "type_duree_selections" in llm_response["data"]:
        st.session_state.selected_types_duree = llm_response["data"]["type_duree_selections"]
        types_duree_message = f"<div class='bot-message'>Parfait {st.session_state.responses.get('student_name', 'Utilisateur')} ! Vous avez sélectionné les types de durée {', '.join(st.session_state.selected_types_duree)} pour {', '.join(st.session_state.matched_subjects)}. Pouvez-vous indiquer les professeurs souhaités pour chaque matière (ou laissez vide si aucun) ?</div>"
        st.session_state.messages.append((types_duree_message, True))
        st.session_state.step = 8

    # Étape 8 : Professeurs souhaités
    elif llm_response["step"] == 8:
        teachers_input = response_text.strip()
        if teachers_input and not re.match(r'^[\w\s,.-]+$', teachers_input):
            error_message = "<div class='bot-message'>Erreur : Noms de professeurs invalides (utilisez lettres, espaces, virgules ou tirets). Veuillez réessayer :</div>"
            if not any("Noms de professeurs invalides" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 8
            return
        st.session_state.responses['user_teachers'] = teachers_input
        teacher_message = f"<div class='bot-message'>Parfait {st.session_state.responses.get('student_name', 'Utilisateur')} ! J'ai enregistré les professeurs {teachers_input or 'non spécifiés'}. Quelle est l'école de l'étudiant ?</div>"
        if not any("Quelle est l'école de l'étudiant" in msg[0] for msg in st.session_state.messages[-3:]):
            st.session_state.messages.append((teacher_message, True))
        st.session_state.step = 9

    # Étape 9 : École
    elif llm_response["step"] == 9:
        school_input = response_text.strip()
        if school_input.lower() not in [s.lower() for s in schools_list]:
            error_message = f"<div class='bot-message'>Erreur : École non reconnue. Veuillez choisir parmi : {', '.join(schools_list)}</div>"
            if not any("École non reconnue" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 9
            return
        st.session_state.responses['user_school'] = school_input
        #school_message = f"<div class='bot-message'>Parfait {st.session_state.responses.get('student_name', 'Utilisateur')} ! L'école {school_input} a été enregistrée. Dans quel centre souhaitez-vous suivre les cours ?</div>"
        #if not any("Dans quel centre souhaitez-vous suivre les cours" in msg[0] for msg in st.session_state.messages[-3:]):
            #st.session_state.messages.append((school_message, True))
        st.session_state.step = 10

    # Étape 10 : Centre
    elif llm_response["step"] == 10:
        center_input = response_text.strip()
        if center_input and center_input.lower() not in [c.lower() for c in centers_list]:
            error_message = f"<div class='bot-message'>Erreur : Centre non reconnu. Veuillez choisir parmi : {', '.join(centers_list)} ou laissez vide.</div>"
            if not any("Centre non reconnu" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 10
            return
        st.session_state.responses['user_center'] = center_input
        center_message = f"<div class='bot-message'>Parfait {st.session_state.responses.get('student_name', 'Utilisateur')} ! Le centre {center_input or 'non spécifié'} a été enregistré. Passons à la sélection des groupes.</div>"
        st.session_state.messages.append((center_message, True))
        st.session_state.step = 11

    # Étape 11 : Sélection des groupes
    elif llm_response["step"] == 11 and "group_selections" in llm_response["data"]:
        required_fields = ['student_name', 'user_level', 'user_subjects', 'user_school', 'user_center']
        missing_fields = [field for field in required_fields if field not in st.session_state.responses or not st.session_state.responses[field]]
        if missing_fields or not st.session_state.selected_forfaits or not st.session_state.selected_types_duree:
            missing_message = "<div class='bot-message'>Données manquantes : "
            if missing_fields:
                missing_message += f"{', '.join(missing_fields)}. "
            if not st.session_state.selected_forfaits:
                missing_message += "Forfaits non sélectionnés. "
            if not st.session_state.selected_types_duree:
                missing_message += "Types de durée non sélectionnés. "
            missing_message += "Veuillez fournir les informations manquantes.</div>"
            if not any("Données manquantes" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((missing_message, True))
            if 'student_name' in missing_fields:
                st.session_state.step = 1
                st.session_state.messages.append(("<div class='bot-message'>Quel est le nom de l'étudiant !</div>", True))
            elif 'user_level' in missing_fields:
                st.session_state.step = 2
                st.session_state.messages.append(("<div class='bot-message'>Quel est votre niveau (ex. BL - 2bac sc PC) ?</div>", True))
            elif 'user_subjects' in missing_fields:
                st.session_state.step = 3
                st.session_state.messages.append(("<div class='bot-message'>Quelles sont les matières qui intéressent l'étudiant ?</div>", True))
            elif 'user_school' in missing_fields:
                st.session_state.step = 9
                st.session_state.messages.append(("<div class='bot-message'>Quelle est l'école de l'étudiant (obligatoire) ?</div>", True))
            elif 'user_center' in missing_fields:
                st.session_state.step = 10
                st.session_state.messages.append(("<div class='bot-message'>Dans quel centre souhaitez-vous suivre les cours ?</div>", True))
            elif not st.session_state.selected_forfaits:
                st.session_state.step = 6
            elif not st.session_state.selected_types_duree:
                st.session_state.step = 7
            return
        
        group_subjects = st.session_state.matched_subjects
        with st.spinner("Recherche en cours..."):
            output, all_recommendations, all_groups_for_selection, matched_subjects = get_recommendations(
                st.session_state.responses['student_name'],
                st.session_state.responses['user_level'],
                ', '.join(group_subjects),
                st.session_state.responses.get('user_teachers', ''),
                st.session_state.responses['user_school'],
                st.session_state.responses.get('user_center', ''),
                st.session_state.selected_forfaits,
                st.session_state.selected_types_duree,
                st.session_state.available_forfaits
            )
        st.session_state.all_recommendations = all_recommendations
        st.session_state.all_groups_for_selection = all_groups_for_selection
        st.session_state.matched_subjects = matched_subjects
        logger.debug(f"Matched subjects 11: {matched_subjects}")
        logger.debug(f"all_groups_for_selection at end of step 11: {st.session_state.all_groups_for_selection}")
        for msg in output:
            if not any(msg in m[0] for m in st.session_state.messages[-3:]):
                st.session_state.messages.append((f"<div class='bot-message'>{msg}</div>", True))
        for subject in matched_subjects:
            for rec in all_recommendations.get(subject, []):
                if not any(rec in m[0] for m in st.session_state.messages[-3:]):
                    st.session_state.messages.append((f"<div class='bot-message'>{rec}</div>", True))

        group_selections = llm_response["data"].get("group_selections", {})
        if isinstance(group_selections, list):
            try:
                indices = [s.strip() for s in response_text.split(',')] if response_text else group_selections
                if len(indices) != len(matched_subjects):
                    raise ValueError(f"Nombre d'indices incorrect : {len(indices)} requis, {len(matched_subjects)} fournis")
                group_selections = {}
                for subject, index in zip(matched_subjects, indices):
                    try:
                        index = int(index)
                        groups = st.session_state.all_groups_for_selection.get(subject, [])
                        if not groups:
                            raise ValueError(f"Aucun groupe disponible pour {subject}")
                        if 1 <= index <= len(groups):
                            group_selections[subject] = groups[index - 1]['id_cours']
                        else:
                            raise ValueError(f"Indice {index} hors plage pour {subject} (1-{len(groups)})")
                    except ValueError as e:
                        if str(e).startswith("Indice") or str(e).startswith("Aucun"):
                            raise
                        raise ValueError(f"Indice non numérique: {index}")
            except (ValueError, AttributeError) as e:
                error_message = f"<div class='bot-message'>Erreur : {str(e)}. Veuillez entrer les numéros des groupes (ex. {'1,2' if len(matched_subjects) > 1 else '1'}) :</div>"
                if not any(f"Erreur : {str(e)}" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((error_message, True))
                groups_message = f"<div class='bot-message'>Veuillez sélectionner un groupe pour chaque matière (entrez les numéros dans l'ordre : {', '.join(matched_subjects)}) :<br>"
                for subject in matched_subjects:
                    groups = st.session_state.all_groups_for_selection.get(subject, [])
                    if groups:
                        groups_message += f"<h3>{subject}</h3>"
                        for i, group in enumerate(groups, 1):
                            groups_message += f"{i}. {group['display']} (Centre: {group['centre']}, Jour: {group['jour']}, {group['heure_debut']}-{group['heure_fin']})<br>"
                    else:
                        groups_message += f"<h3>{subject}</h3>Aucun groupe disponible.<br>"
                groups_message += f"Exemple : {'1,2' if len(matched_subjects) > 1 else '1'}</div>"
                if not any("Veuillez sélectionner un groupe pour chaque matière" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((groups_message, True))
                st.session_state.step = 11
                logger.error(f"Erreur lors du parsing des indices : {str(e)}")
                return

        st.session_state.selected_groups = {}
        valid_subjects = []
        groups_message = "<div class='bot-message'>Groupes sélectionnés :<br>"
        for subject, id_cours in group_selections.items():
            if subject in st.session_state.all_groups_for_selection:
                groups = st.session_state.all_groups_for_selection[subject]
                selected_group = next((group for group in groups if group['id_cours'] == id_cours), None)
                if selected_group:
                    valid_subjects.append(subject)
                    st.session_state.selected_groups[subject] = selected_group
                    groups_message += f"<h3>{subject}</h3>[{id_cours}] {selected_group['display']} (Centre: {selected_group['centre']}, Jour: {selected_group['jour']}, {selected_group['heure_debut']}-{selected_group['heure_fin']})<br>"
                else:
                    groups_message += f"<h3>{subject}</h3>Groupe {id_cours} invalide.<br>"
            else:
                groups_message += f"<h3>{subject}</h3>Aucun groupe disponible.<br>"

        if valid_subjects and len(valid_subjects) == len(matched_subjects):
            st.session_state.matched_subjects = valid_subjects
            logger.debug(f"Groupes sélectionnés validés : {st.session_state.selected_groups}")
            st.session_state.responses['group_selections'] = ','.join(
                str(next(i + 1 for i, g in enumerate(st.session_state.all_groups_for_selection[subject]) if g['id_cours'] == group['id_cours']))
                for subject, group in st.session_state.selected_groups.items()
            )

            overlaps = check_overlaps(st.session_state.selected_groups)
            if overlaps:
                conflict_msg = "<div class='bot-message'><b>Conflit détecté :</b><br>"
                for g1, g2 in overlaps:
                    conflict_msg += f"- Chevauchement entre {g1['matiere']} ({g1['name_cours']}) et {g2['matiere']} ({g2['name_cours']}) : "
                    conflict_msg += f"{g1['jour']} {g1['heure_debut']}-{g1['heure_fin']} vs {g2['jour']} {g2['heure_debut']}-{g2['heure_fin']}<br>"
                conflict_msg += "Veuillez choisir une nouvelle combinaison de groupes :</div>"
                if not any("Conflit détecté" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((conflict_msg, True))
                groups_message = f"<div class='bot-message'>Veuillez sélectionner un groupe pour chaque matière (entrez les numéros dans l'ordre : {', '.join(matched_subjects)}) :<br>"
                for subject in matched_subjects:
                    groups = st.session_state.all_groups_for_selection.get(subject, [])
                    if groups:
                        groups_message += f"<h3>{subject}</h3>"
                        for i, group in enumerate(groups, 1):
                            groups_message += f"{i}. {group['display']} (Centre: {group['centre']}, Jour: {group['jour']}, {group['heure_debut']}-{group['heure_fin']})<br>"
                groups_message += f"Exemple : {'1,2' if len(matched_subjects) > 1 else '1'}</div>"
                if not any("Veuillez sélectionner un groupe pour chaque matière" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((groups_message, True))
                st.session_state.step = 11
                logger.debug("Chevauchement détecté, retour à l'étape 11")
                return

    # Étape 12 : Frais d'inscription
    elif llm_response["step"] == 12:
        choice = response_text.strip().lower()
        if choice in ['oui', 'yes']:
            frais_inscription = 250
            total_with_frais = st.session_state.total_tariff_base + frais_inscription
            st.session_state.total_with_frais = total_with_frais
            st.session_state.responses['include_fees'] = True
            frais_message = f"<div class='bot-message'>Frais d'inscription inclus : {frais_inscription} DH<br>Total avec frais : {total_with_frais:.2f} DH</div>"
            if not any("Frais d'inscription inclus" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((frais_message, True))
            comments_message = "<div class='bot-message'>Y a-t-il des commentaires ou demandes supplémentaires ?</div>"
            if not any("commentaires ou demandes supplémentaires" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((comments_message, True))
            st.session_state.step = 13
        elif choice in ['non', 'no']:
            st.session_state.total_with_frais = st.session_state.total_tariff_base
            st.session_state.responses['include_fees'] = False
            total_message = f"<div class='bot-message'>Total sans frais d'inscription : {st.session_state.total_tariff_base:.2f} DH</div>"
            if not any("Total sans frais d'inscription" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((total_message, True))
            comments_message = "<div class='bot-message'>Y a-t-il des commentaires ou demandes supplémentaires ?</div>"
            if not any("commentaires ou demandes supplémentaires" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((comments_message, True))
            st.session_state.step = 13
        else:
            error_message = "<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'.</div>"
            if not any("Veuillez répondre par 'Oui' ou 'Non'" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 12

    # Étape 13 : Commentaires
    elif llm_response["step"] == 13:
        comments_input = response_text.strip()
        st.session_state.responses['comments'] = comments_input
        # Détecter "réduction" ou "avec <pourcentage>%"
        reduction_detected = "réduction"or "reduction" in comments_input.lower()
        percentage_match = re.search(r'\s*(\d{1,2})%', comments_input.lower().strip())
        print(f"pourcentage 13: {percentage_match}")
        if reduction_detected and percentage_match:
            #st.session_state.reduction_percentage=float(percentage_match)
            try:
                percentage = float(percentage_match.group(1))
                if 0 <= percentage <= 100:
                    st.session_state.reduction_percentage = percentage
                    total_base = st.session_state.total_with_frais
                    reduction_amount = total_base * (percentage / 100)
                    total_final = total_base - reduction_amount
                    reduction_message = f"<div class='bot-message'>Commentaires enregistrés : {comments_input}<br><b>Total de base</b> : {total_base:.2f} DH <br><b>Réduction de {percentage}%</b> : -{reduction_amount:.2f} DH <br><b>Total final</b> : {total_final:.2f} DH<br>Voulez-vous traiter un autre cas ? (Oui/Non)</div>"
                    if not any("Total final" in msg[0] for msg in st.session_state.messages[-3:]):
                        st.session_state.messages.append((reduction_message, True))
                    st.session_state.step = 15
                    st.session_state.responses['reduction_percentage'] = percentage
                else:
                    error_message = "<div class='bot-message'>Erreur : Le pourcentage spécifié doit être entre 0 et 100. Veuillez préciser le pourcentage de réduction :</div>"
                    if not any("Le pourcentage spécifié doit être entre 0 et 100" in msg[0] for msg in st.session_state.messages[-3:]):
                        st.session_state.messages.append((error_message, True))
                    st.session_state.step = 14
            except ValueError:
                error_message = "<div class='bot-message'>Erreur : Pourcentage invalide dans les commentaires. Veuillez préciser le pourcentage de réduction :</div>"
                if not any("Pourcentage invalide dans les commentaires" in msg[0] for msg in st.session_state.messages[-3:]):
                    st.session_state.messages.append((error_message, True))
                st.session_state.step = 14
        elif reduction_detected:
            reduction_message = f"<div class='bot-message'>Commentaires enregistrés : {comments_input}<br>Veuillez préciser le pourcentage de réduction :</div>"    
            st.session_state.messages.append((reduction_message, True))
            st.session_state.step = 14
        else:
            total_message = f"<div class='bot-message'>Commentaires enregistrés : {comments_input or 'Aucun'}<br>Total final : {st.session_state.total_with_frais:.2f} DH<br>Voulez-vous traiter un autre cas ? (Oui/Non)</div>"
            if not any("Total final" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((total_message, True))
            st.session_state.step = 15

    # Étape 14 : Pourcentage de réduction
    elif llm_response["step"] == 14:
        try:
            percentage = float(response.strip())
            if 0 <= percentage <= 100:
                st.session_state.messages.append((f"<div class='user-message'>{response}</div>", False))
                st.session_state.reduction_percentage = percentage
                total_base = st.session_state.total_with_frais
                reduction_amount = st.session_state.total_with_frais * (percentage / 100)
                total_final = st.session_state.total_with_frais - reduction_amount
                reduction_message = f"<div class='bot-message'><b>Total de base</b> : {total_base:.2f} DH <br>"
                reduction_message += f"<b>Réduction de {percentage}%</b> : -{reduction_amount:.2f} DH <br>"
                reduction_message += f"<b>Total réduit</b> : {total_final:.2f} DH</div>"
                st.session_state.messages.append((reduction_message, True))
                
                st.session_state.messages.append(("<div class='bot-message'>Voulez-vous traiter un autre cas ? (Oui/Non)</div>", True))
                st.session_state.step = 15
            else:
                st.session_state.messages.append(("<div class='bot-message'>Erreur : Le pourcentage doit être entre 0 et 100. Veuillez réessayer :</div>", True))
                st.session_state.step = 14
        except ValueError:
            st.session_state.messages.append(("<div class='bot-message'>Erreur : Veuillez entrer un nombre valide (ex. 10). Veuillez réessayer :</div>", True))
            st.session_state.step = 14

    # Étape 15 : Nouveau cas
    elif llm_response["step"] == 15:
        choice = response_text.strip().lower()
        if choice in ['oui', 'yes']:
            st.session_state.responses['another_case'] = True
            if not any("D'accord, commençons un nouveau cas" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((f"<div class='user-message'>{response_text}</div>", False))
                st.session_state.messages.append(("<div class='bot-message'>D'accord, commençons un nouveau cas. Quel est le nom de l'étudiant ?</div>", True))
            st.session_state.step = 1
            st.session_state.responses = {}
            st.session_state.selected_groups = {}
            st.session_state.tariffs_by_group = {}
            st.session_state.total_tariff_base = 0
            st.session_state.total_with_discount = 0
            st.session_state.frais_inscription = 0
            st.session_state.total_with_frais = 0
            st.session_state.reduction_percentage = 0
            st.session_state.total_final = 0
            st.session_state.matched_subjects = []
            st.session_state.all_groups_for_selection = {}
            st.session_state.available_types_duree = {}
            st.session_state.selected_types_duree = {}
            st.session_state.available_forfaits = {}
        elif choice in ['non', 'no']:
            st.session_state.responses['another_case'] = False
            if not any("Merci pour votre interaction" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((f"<div class='user-message'>{response_text}</div>", False))
                st.session_state.messages.append(("<div class='bot-message'>Merci pour votre interaction. À bientôt !</div>", True))
            st.session_state.step = 0
        else:
            error_message = "<div class='bot-message'>Veuillez répondre par 'Oui' ou 'Non'.</div>"
            if not any("Veuillez répondre par 'Oui' ou 'Non'" in msg[0] for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append((error_message, True))
            st.session_state.step = 15

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
        st.session_state.available_forfaits = {}
        st.session_state.available_types_duree = {}
        st.session_state.selected_forfaits = {}
        st.session_state.selected_types_duree = {}
        st.session_state.reduction_percentage = 0
        st.rerun()
    st.write("---")
    st.write("**À propos**")
    st.write("Développé par IA pour optimiser la recherche de groupes éducatifs.")

# Initialisation de l'état
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
    st.session_state.available_forfaits = {}
    st.session_state.available_types_duree = {}
    st.session_state.selected_forfaits = {}
    st.session_state.selected_types_duree = {}
    st.session_state.reduction_percentage = 0

# Affichage des messages
st.markdown("<div class='container'>", unsafe_allow_html=True)
for message, is_bot in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Questions et placeholders
questions = [
    "Quel est le nom de l'étudiant !",
    "Quel est le niveau de l'étudiant (ex. BL - 2bac sc PC) ?",
    "Quelles sont les matières qui intéressent l'étudiant ?",
    "Quelles sont les notes de l'étudiant pour ces matières ? (facultatif)",
    "Veuillez choisir le type de cours pour chaque matière (indiv/groupe)",
    "Veuillez choisir un forfait pour chaque matière en groupe",
    "Veuillez choisir le type de durée pour chaque forfait",
    "Quels sont les professeurs actuels de l'étudiant pour ces matières ?",
    "Quelle est l'école de l'étudiant (obligatoire) ?",
    "Quel est le centre souhaité par l'étudiant ?",
    "Veuillez sélectionner un groupe pour chaque matière",
    "Voulez-vous inclure les frais d'inscription (250 DH par matière en groupe) ?",
    "Y a-t-il des commentaires ou demandes supplémentaires ?",
    "Quel est le pourcentage de réduction que vous souhaitez appliquer ?",
    "Voulez-vous traiter un autre cas ?"
]

placeholders = [
    "Ex: Ahmed larache",
    f"Ex: {', '.join(levels_list[:3]) if levels_list else 'BL - 2bac sc PC'}",
    f"Ex: {', '.join(subjects_list[:3]) if subjects_list else 'Français, Mathématiques'}",
    "Ex: 12, 15 (ou laissez vide)",
    "Ex: indiv,groupe ou 'je veux groupe pour toutes les matières'",
    "Ex: 1,2 (numéro du forfait)",
    "Ex: 1,2 (numéro du type de durée)",
    "Ex: John Doe, Jane Smith (facultatif, peut être vide)",
    f"Ex: {schools_list[0] if schools_list else 'Massignon Bouskoura'}",
    f"Ex: {centers_list[0] if centers_list else 'Centre A'} (ou laissez vide)",
    "Ex: 1,2",
    "Oui ou Non",
    "Ex: J'aimerais une réduction",
    "Ex: 10 (entre 0 et 100)",
    "Oui ou Non"
]

# Logique conversationnelle
if st.session_state.step == 0:
    st.session_state.messages = [("<div class='bot-message'>Bonjour,j'espère que tu vas. <br> Dis moi ! Quel est le nom de l'étudiant ?</div>", True)]
    st.session_state.step = 1
    st.session_state.current_input = ""
    st.session_state.submitted = False
    st.session_state.input_counter = 0
    st.rerun()

elif st.session_state.step in range(1, 16):
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