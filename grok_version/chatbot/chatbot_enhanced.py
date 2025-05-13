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
import uuid

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
    st.error("Erreur : La collection ChromaDB des groupes est vide.")
    st.stop()
if collection_combinaisons.count() == 0:
    st.error("Erreur : La collection ChromaDB des combinaisons est vide.")
    st.stop()

# Charger le modèle SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Récupérer les valeurs uniques depuis ChromaDB
all_groups = collection_groupes.get(include=["metadatas"])
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
                    except (ValueError, TypeError) as e:
                        logger.error(f"Erreur lors du parsing de duree_tarifs pour {id_forfait}: {str(e)}")
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
    valid1, msg1 = validate_group_structure(group1)
    valid2, msg2 = validate_group_structure(group2)
    if not valid1 or not valid2:
        logger.error(f"Structure de groupe invalide: {msg1 if not valid1 else msg2}")
        return False
    if group1['jour'] != group2['jour']:
        return False
    try:
        start1 = parse_time(group1['heure_debut'])
        end1 = parse_time(group1['heure_fin'])
        start2 = parse_time(group2['heure_debut'])
        end2 = parse_time(group2['heure_fin'])
    except Exception as e:
        logger.error(f"Erreur de parsing des heures: {str(e)}")
        return False
    if not all([start1, end1, start2, end2]):
        return False
    same_center = group1['centre'] == group2['centre']
    margin = timedelta(minutes=15) if not same_center else timedelta(0)
    return (start1 < end2 + margin) and (start2 < end1 + margin)

def check_overlaps(selected_groups):
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
    logger.debug(f"get_recommendations: inputs: student_name={student_name}, level={user_level}, subjects={user_subjects}, "
                 f"teachers={user_teachers}, school={user_school}, center={user_center}, forfaits={selected_forfaits}, "
                 f"types_duree={selected_types_duree}")
    output = []
    all_recommendations = {}
    all_groups_for_selection = {}
    matched_level = match_value(user_level, levels)[0]
    matched_subjects = [match_value(subj.strip(), subjects)[0] for subj in user_subjects.split(",")]
    if isinstance(user_teachers, list):
        matched_teachers = [teacher.strip() for teacher in user_teachers if teacher] if user_teachers else [None] * len(matched_subjects)
    elif isinstance(user_teachers, str) and user_teachers:
        matched_teachers = [teacher.strip() for teacher in user_teachers.split(",")]
    else:
        matched_teachers = [None] * len(matched_subjects)
    matched_school = match_value(user_school, schools)[0]
    matched_center = match_value(user_center, centers)[0] if user_center else None
    for subject in matched_subjects:
        all_recommendations[subject] = []
        all_groups_for_selection[subject] = []
    all_groups_data = collection_groupes.get(include=["metadatas", "documents"])
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
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                rejected_groups.append((metadata, f"Missing required keys: {missing_keys}"))
                continue
            if (metadata['niveau'].strip().lower() == matched_level.strip().lower() and
                metadata['matiere'].strip().lower() == matched_subject.strip().lower() and
                metadata.get('id_forfait') == id_forfait and
                metadata.get('type_duree_id') == type_duree_id):
                validated_metadata = metadata.copy()
                for key, default in optional_keys.items():
                    validated_metadata[key] = metadata.get(key, default)
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
                if matched_center and validated_metadata['centre'].lower() != matched_center.lower():
                    rejected_groups.append((metadata, f"Centre mismatch: {validated_metadata['centre']}"))
                    continue
                if matched_teacher and matched_teacher != 'N/A' and validated_metadata['teacher'] != matched_teacher:
                    rejected_groups.append((metadata, f"Teacher mismatch: {validated_metadata['teacher']}"))
                    continue
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
        if matched_center:
            priority_groups = [g for g in group_list if 
                             (not matched_teacher or matched_teacher == g['teacher']) and 
                             g['centre'].lower() == matched_center.lower()]
            priority_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_groups, "Professeur, Centre, École")
            if len(selected_groups) < 3:
                remaining_groups = [g for g in group_list if g['centre'].lower() == matched_center.lower() and g['id_cours'] not in selected_ids]
                add_groups(remaining_groups, "Centre")
        else:
            priority_groups = [g for g in group_list if 
                             (not matched_teacher or matched_teacher == g['teacher'])]
            priority_groups.sort(key=lambda x: count_school_students(x['schools'], matched_school), reverse=True)
            add_groups(priority_groups, "Professeur, École")
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
    .profile-name { text-align: center; font-size: 25px; color: white; margin-top: 10px; margin-left: 5px; }
    </style>
""", unsafe_allow_html=True)

# Fonction de traitement avec Gemini
def process_with_llm(input_text, session_state):
    try:
        # Extraire les parties sérialisables de session_state
        session_state_data = {
            "step": session_state.get('step', 0),
            "responses": session_state.get('responses', {}),
            "matched_subjects": session_state.get('matched_subjects', []),
            "available_forfaits": session_state.get('available_forfaits', {}),
            "available_types_duree": session_state.get('available_types_duree', {}),
            "all_groups_for_selection": session_state.get('all_groups_for_selection', {})
        }

        # Convertir les objets en chaînes pour l'f-string
        session_state_str = json.dumps(session_state_data, ensure_ascii=False)
        levels_list_str = ', '.join(levels_list)
        subjects_list_str = ', '.join(subjects_list)
        schools_list_str = ', '.join(schools_list)
        centers_list_str = ', '.join(centers_list)
        teachers_list_str = ', '.join(teachers_list)
        forfaits_list_str = ', '.join(forfaits_list)
        matched_subjects_str = json.dumps(session_state.get('matched_subjects', []), ensure_ascii=False)
        available_forfaits_str = json.dumps(session_state.get('available_forfaits', {}), ensure_ascii=False)
        available_types_duree_str = json.dumps(session_state.get('available_types_duree', {}), ensure_ascii=False)
        all_groups_for_selection_str = json.dumps(session_state.get('all_groups_for_selection', {}), ensure_ascii=False)

        prompt = f"""
        **Contexte**:
        Vous êtes un chatbot de recommandation de groupes éducatifs, aidant les utilisateurs à trouver des groupes adaptés (niveau, matière, professeur, école, centre, forfait, type de durée) et à calculer les tarifs. Les données sont extraites d’une base ChromaDB. Le flux conversationnel comporte 15 étapes, collectant les informations de manière stricte avant de proposer des forfaits et groupes.

        **Entrée utilisateur**: '{input_text}'
        **État actuel**:
        - Étape: {session_state_data['step']}
        - Réponses: {session_state_str}
        - Matières sélectionnées: {matched_subjects_str}
        - Forfaits disponibles: {available_forfaits_str}
        - Types de durée: {available_types_duree_str}
        - Groupes pour sélection: {all_groups_for_selection_str}
        **Listes de référence**:
        - Niveaux: {levels_list_str}
        - Matières: {subjects_list_str}
        - Écoles: {schools_list_str}
        - Centres: {centers_list_str}
        - Professeurs: {teachers_list_str}
        - Forfaits: {forfaits_list_str}

        **Instructions**:
        - Analysez l'entrée utilisateur et l'état actuel pour produire une réponse JSON.
        - Gérez les 15 étapes avec validations strictes, transitions logiques, et détection des intentions ("changer", "revenir").
        - Produisez des messages humanisés, amicaux, et contextuels.
        - Incluez des actions spécifiques (par exemple, "get_recommendations", "calculate_tariffs") lorsque nécessaire.

        **Étapes et validations**:
        1. **Nom (Étape 1)**:
           - Entrée non vide → Stocker "student_name", passer à l'étape 2.
           - Sinon → Message: "Veuillez fournir un nom non vide.", rester à l'étape 1.
        2. **Niveau (Étape 2)**:
           - Entrée dans levels_list (insensible à la casse) → Stocker "user_level", passer à l'étape 3.
           - Sinon → Message: "Niveau non reconnu. Choisissez parmi : {levels_list_str}.", rester à l'étape 2.
        3. **Matières (Étape 3)**:
           - Liste séparée par virgules, au moins une matière valide dans subjects_list → Stocker "user_subjects" (chaîne), "subjects" (liste), passer à l'étape 4.
           - Sinon → Message: "Aucune matière valide. Choisissez parmi : {subjects_list_str}.", rester à l'étape 3.
        4. **Notes (Étape 4)**:
           - Vide → Stocker "grades": [], passer à l'étape 5.
           - Nombres valides séparés par virgules → Stocker "grades": [notes], passer à l'étape 5.
           - Sinon → Message: "Entrez des notes valides (ex. 12, 15) ou laissez vide.", rester à l'étape 4.
        5. **Type de cours (Étape 5)**:
           - Liste explicite (ex. "indiv,groupe") ou raccourci (ex. "groupe pour toutes") correspondant à matched_subjects → Stocker "course_choices", passer à l'étape 6 si au moins un "groupe", sinon étape 15.
           - Sinon → Message: "Choisissez le type de cours pour chaque matière ({matched_subjects_str}) : (ex. indiv,groupe).", rester à l'étape 5.
        6. **Forfait (Étape 6)**:
           - Indices valides pour available_forfaits → Stocker "forfait_selections" ({{matière: id_forfait}}), passer à l'étape 7.
           - Action: Appeler la focntion qui retourne un esemble de forfaits get_available_forfaits(level, subject) pour chaque matière dans matched_subjects.
           - Sinon → Message: "Entrez des numéros de forfaits valides (ex. 1,2).", rester à l'étape 6.
        7. **Type de durée (Étape 7)**:
           - Indices valides pour available_types_duree → Stocker "type_duree_selections" ({{matière: type_duree_id}}), passer à l'étape 8.
           - Sinon → Message: "Entrez des numéros de types de durée valides (ex. 1,2).", rester à l'étape 7.
        8. **Professeurs (Étape 8)**:
           - Texte ou vide → Stocker "user_teachers", passer à l'étape 9.
        9. **École (Étape 9)**:
           - Entrée dans schools_list → Stocker "user_school", passer à l'étape 10.
           - Sinon → Message: "École non reconnue. Choisissez parmi : {schools_list_str}.", rester à l'étape 9.
        10. **Centre (Étape 10)**:
           - Vide ou dans centers_list → Stocker "user_center", passer à l'étape 11.
           - Sinon → Message: "Centre non reconnu. Choisissez parmi : {centers_list_str} ou laissez vide.", rester à l'étape 10.
        11. **Groupes (Étape 11)**:
           - Indices valides pour all_groups_for_selection → Stocker "group_selections" ({{matière: id_cours}}), action: appeler get_recommendations, vérifier chevauchements, passer à l'étape 12.
           - Vérifier: student_name, user_level, user_subjects, user_school, forfait_selections, type_duree_selections.
           - Sinon → Message: "Entrez des numéros de groupes valides (ex. 1,2).", rester à l'étape 11.
        12. **Frais (Étape 12)**:
           - "Oui"/"Non" → Stocker "include_fees" (true/false), action: appeler calculate_tariffs, passer à l'étape 13.
           - Sinon → Message: "Répondez par 'Oui' ou 'Non'.", rester à l'étape 12.
        13. **Commentaires (Étape 13)**:
           - Stocker "comments".
           - Si "réduction"/"remise" détecté avec pourcentage (ex. "10%") → Stocker "reduction_percentage", passer à l'étape 15.
           - Si "réduction" sans pourcentage → Passer à l'étape 14.
           - Sinon → Passer à l'étape 15.
        14. **Pourcentage (Étape 14)**:
           - Nombre entre 0 et 100 → Stocker "reduction_percentage", passer à l'étape 15.
           - Sinon → Message: "Entrez un nombre valide (ex. 10).", rester à l'étape 14.
        15. **Autre cas (Étape 15)**:
           - "Oui" → Réinitialiser l'état, passer à l'étape 1.
           - "Non" → Passer à l'étape 0.
           - Sinon → Message: "Répondez par 'Oui' ou 'Non'.", rester à l'étape 15.

        **Gestion des intentions**:
        - "Changer [champ]" (ex. "changer le nom à Ahmed") → Mettre à jour le champ, revenir à l'étape correspondante.
        - "Revenir à [étape/champ]" → Revenir à l'étape spécifiée.
        - Si intention détectée, effacer les données des étapes ultérieures.

        **Réponse JSON**:
        ```json
        {{
          "step": <int>,
          "data": {{}},
          "message": "<string>",
          "error": "<string|null>",
          "actions": [{{}}],
          "next_step": <int>
        }}
        ```
        - "data": Données validées à stocker.
        - "message": Message humanisé pour l'utilisateur.
        - "error": Erreur s'il y a lieu.
        - "actions": Actions à exécuter par le code Python.
        - "next_step": Prochaine étape.

        **Exemples**:
        - Étape 1, Entrée: "Ahmed" → 
          ```json
          {{
            "step": 1,
            "data": {{"student_name": "Ahmed"}},
            "message": "Bonjour Ahmed ! Quel est votre niveau ?",
            "error": null,
            "actions": [],
            "next_step": 2
          }}
          ```
        - Étape 3, Entrée: "Mathématiques,Physique" → 
          ```json
          {{
            "step": 3,
            "data": {{"user_subjects": "Mathématiques,Physique", "subjects": ["Mathématiques", "Physique"]}},
            "message": "Matières validées. Quelles sont vos notes ?",
            "error": null,
            "actions": [],
            "next_step": 4
          }}
          ```
        - Étape 6, Entrée: "1,2" → 
          ```json
          {{
            "step": 6,
            "data": {{"forfait_selections": {{"Mathématiques": "123", "Physique": "124"}}}},
            "message": "Forfaits sélectionnés. Choisissez les types de durée.",
            "error": null,
            "actions": [{{"type": "get_forfaits", "params": {{"subjects": ["Mathématiques", "Physique"], "level": "BL - 2bac sc PC"}}}}],
            "next_step": 7
          }}
          ```

        **Réponse**:
        Fournissez une réponse JSON conforme, avec un message amical et contextuel. Assurez-vous que les validations sont robustes et que les transitions sont cohérentes.
        """

        gemini_response = gemini_model.generate_content(prompt)
        response_text = gemini_response.text.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        try:
            response = json.loads(response_text)
            required_keys = ["step", "data", "message", "error", "actions", "next_step"]
            if not all(key in response for key in required_keys):
                raise ValueError("Réponse JSON incomplète")
            logger.debug(f"Réponse Gemini : {response}")
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing de la réponse Gemini : {str(e)}")
            return {
                "step": session_state.get('step', 0),
                "data": {},
                "message": "Erreur interne, veuillez réessayer.",
                "error": "Erreur de parsing",
                "actions": [],
                "next_step": session_state.get('step', 0)
            }
    except Exception as e:
        logger.error(f"Erreur lors de l’appel à l’API Gemini : {str(e)}")
        return {
            "step": session_state.get('step', 0),
            "data": {},
            "message": "Erreur interne, veuillez réessayer.",
            "error": "Erreur API",
            "actions": [],
            "next_step": session_state.get('step', 0)
        }

# Gestion des étapes
def handle_input_submission(response_text: str) -> None:
    logger.debug(f"Traitement de l'entrée : {response_text}")
    llm_response = process_with_llm(response_text, st.session_state)
    
    # Mettre à jour l'état
    st.session_state.step = llm_response["next_step"]
    st.session_state.responses.update(llm_response["data"])
    
    # Ajouter le message de l'utilisateur
    user_message = f"<div class='user-message'>{response_text}</div>"
    if not any(msg[0] == user_message for msg in st.session_state.messages[-3:]):
        st.session_state.messages.append((user_message, False))
    
    # Ajouter le message du bot
    if llm_response["message"]:
        bot_message = f"<div class='bot-message'>{llm_response["message"]}</div>"
        if not any(msg[0] == bot_message for msg in st.session_state.messages[-3:]):
            st.session_state.messages.append((bot_message, True))
    
    # Ajouter un message d'erreur si nécessaire
    if llm_response.get("error"):
        error_message = f"<div class='bot-message'>Erreur : {llm_response['error']}</div>"
        if not any(msg[0] == error_message for msg in st.session_state.messages[-3:]):
            st.session_state.messages.append((error_message, True))

    # Exécuter les actions spécifiées par l'LLM
    for action in llm_response.get("actions", []):
        if action["type"] == "get_forfaits":
            subjects = action["params"].get("subjects", [])
            level = action["params"].get("level", st.session_state.responses.get("user_level", ""))
            st.session_state.available_forfaits = {}
            for subject in subjects:
                forfaits = get_available_forfaits(level, subject)
                st.session_state.available_forfaits[subject] = forfaits
        elif action["type"] == "get_recommendations":
            with st.spinner("Recherche en cours..."):
                output, all_recommendations, all_groups_for_selection, matched_subjects = get_recommendations(
                    st.session_state.responses.get('student_name', ''),
                    st.session_state.responses.get('user_level', ''),
                    ', '.join(st.session_state.matched_subjects),
                    st.session_state.responses.get('user_teachers', ''),
                    st.session_state.responses.get('user_school', ''),
                    st.session_state.responses.get('user_center', ''),
                    st.session_state.selected_forfaits,
                    st.session_state.selected_types_duree,
                    st.session_state.available_forfaits
                )
            st.session_state.all_recommendations = all_recommendations
            st.session_state.all_groups_for_selection = all_groups_for_selection
            st.session_state.matched_subjects = matched_subjects
            for msg in output:
                if not any(msg in m[0] for m in st.session_state.messages[-3:]):
                    st.session_state.messages.append((f"<div class='bot-message'>{msg}</div>", True))
            for subject in matched_subjects:
                for rec in all_recommendations.get(subject, []):
                    if not any(rec in m[0] for m in st.session_state.messages[-3:]):
                        st.session_state.messages.append((f"<div class='bot-message'>{rec}</div>", True))
        elif action["type"] == "calculate_tariffs":
            tariffs_by_group, tariff_message, total_tariff_base = calculate_tariffs(
                st.session_state.selected_groups,
                {subject: st.session_state.available_forfaits[subject][group['id_forfait']]['types_duree'][group['type_duree_id']]['name']
                 for subject, group in st.session_state.selected_groups.items()},
                {subject: group['type_duree_id'] for subject, group in st.session_state.selected_groups.items()},
                st.session_state.available_forfaits
            )
            if tariffs_by_group:
                st.session_state.tariffs_by_group = tariffs_by_group
                st.session_state.total_tariff_base = total_tariff_base
                if not any(tariff_message in m[0] for m in st.session_state.messages[-3:]):
                    st.session_state.messages.append((f"<div class='bot-message'>{tariff_message}</div>", True))
            else:
                error_message = f"<div class='bot-message'>{tariff_message}</div>"
                if not any(tariff_message in m[0] for m in st.session_state.messages[-3:]):
                    st.session_state.messages.append((error_message, True))

# Interface Streamlit
logo_path = os.path.join(parent_dir, "images", "logo.png")
try:
    st.image(logo_path)
except FileNotFoundError:
    st.warning("Logo non trouvé.")
st.title("Chatbot de Recommandation de Groupes")
profile_path = os.path.join(parent_dir, "images", "profile1.png")

with st.sidebar:
    st.image(profile_path, width=280)
    st.markdown("<div class='profile-name'>ELARACHE Jalal</div>", unsafe_allow_html=True)
    st.header("Options")
    st.write("Bienvenue dans le Chatbot de Recommandation !")
    if st.button("Réinitialiser la conversation"):
        st.session_state.clear()
        st.session_state.step = 0
        st.session_state.messages = [("<div class='bot-message'>Bonjour ! Je vais vous aider à trouver des groupes recommandés.</div>", True)]
        st.session_state.responses = {}
        st.session_state.current_input = ""
        st.session_state.submitted = False
        st.session_state.input_counter = 0
        st.rerun()

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
    st.session_state.tariffs_by_group = {}
    st.session_state.total_tariff_base = 0
    st.session_state.available_forfaits = {}
    st.session_state.available_types_duree = {}
    st.session_state.selected_forfaits = {}
    st.session_state.selected_types_duree = {}

# Affichage des messages
st.markdown("<div class='container'>", unsafe_allow_html=True)
for message, is_bot in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Logique conversationnelle
if st.session_state.step == 0:
    st.session_state.messages = [("<div class='bot-message'>Bonjour, j'espère que vous allez bien ! Quel est le nom de l'étudiant ?</div>", True)]
    st.session_state.step = 1
    st.session_state.current_input = ""
    st.session_state.submitted = False
    st.session_state.input_counter = 0
    st.rerun()

elif st.session_state.step in range(1, 16):
    input_key = f"input_step_{st.session_state.step}_{st.session_state.input_counter}"
    response = st.text_input("Votre réponse :", key=input_key, placeholder="Entrez votre réponse ici...")
    if input_key in st.session_state and st.session_state[input_key] != st.session_state.current_input:
        st.session_state.current_input = st.session_state[input_key]
        st.session_state.submitted = True
    if st.session_state.submitted and st.session_state.current_input:
        handle_input_submission(st.session_state.current_input)
        st.session_state.input_counter += 1
        st.session_state.current_input = ""
        st.session_state.submitted = False
        st.rerun()