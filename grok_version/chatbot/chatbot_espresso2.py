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
from bs4 import BeautifulSoup

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

# Fonctions utilitaires (inchangées, incluses pour complétude)
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

    tariff_data = {
        "tariffs_by_group": tariffs_by_group,
        "total_tariff_base": total_tariff_base,
        "reduction_applied": reduction_applied,
        "reduction_description": reduction_description
    }
    
    return tariffs_by_group, tariff_data, total_tariff_base

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
    logger.debug(f"get_recommendations: raw groups fetched: {all_groups_data}")

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
                logger.error(f"Invalid group (id_cours: {metadata.get('id_cours', 'unknown')}): Missing required keys: {missing_keys}")
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
                    logger.debug(f"Group {validated_metadata['id_cours']} filtered out: centre {validated_metadata['centre']} != {matched_center}")
                    rejected_groups.append((metadata, f"Centre mismatch: {validated_metadata['centre']}"))
                    continue
                if matched_teacher and matched_teacher != 'N/A' and validated_metadata['teacher'] != matched_teacher:
                    logger.debug(f"Group {validated_metadata['id_cours']} filtered out: teacher {validated_metadata['teacher']} != {matched_teacher}")
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
            recommendation = {
                "id": i,
                "subject": matched_subject,
                "id_cours": group['id_cours'],
                "name_cours": group['name_cours'],
                "forfait": f"[{group['id_forfait']}] {group['nom_forfait']}",
                "num_students": group['num_students'],
                "teacher": group['teacher'],
                "centre": group['centre'],
                "date_debut": group['date_debut'],
                "date_fin": group['date_fin'],
                "heure_debut": group['heure_debut'],
                "heure_fin": group['heure_fin'],
                "jour": group['jour'],
                "description": group['description'],
                "criteria": group['criteria']
            }
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

    output.append(f"Recommandations générées pour l'étudiant {student_name}.")
    return output, all_recommendations, all_groups_for_selection, matched_subjects

def filter_redundant_messages(messages, new_message):
    """Filtre les messages redondants en comparant le contenu HTML"""
    def clean_html(html):
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join(soup.get_text().split())

    new_cleaned = clean_html(new_message)
    for msg, _ in messages[-5:]:
        msg_cleaned = clean_html(msg)
        if new_cleaned == msg_cleaned:
            logger.debug(f"Message redondant détecté : {new_cleaned}")
            return False
    return True

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

def process_with_llm(input_text, step, session_state, lists, system_data=None):
    try:
        system_data_json = json.dumps(system_data) if system_data else "{}"
        prompt = f"""
        **Contexte**:
        Vous êtes un chatbot Streamlit de recommandation de groupes éducatifs. Votre rôle est de centraliser la génération des réponses, en reformulant les données fournies par le système pour éviter les répétitions et garantir une conversation fluide. La première information demandée est le nom de l'étudiant, suivi du niveau. À l'étape des notes, plusieurs notes peuvent être fournies, chacune correspondant à une matière dans l'ordre de saisie.

        **Étape actuelle**: {step}
        **Entrée utilisateur**: '{input_text}'
        **Réponses actuelles**: {json.dumps(session_state.responses)}
        **Données du système (facultatif)**: {system_data_json}
        **Listes de référence**:
        - Niveaux: {', '.join(lists['levels_list'])}
        - Matières: {', '.join(lists['subjects_list'])}
        - Écoles: {', '.join(lists['schools_list'])}
        - Centres: {', '.join(lists['centers_list'])}
        - Professeurs: {', '.join(lists['teachers_list'])}
        **État**:
        - Matières sélectionnées: {session_state.get('matched_subjects', [])}
        - Forfaits disponibles: {session_state.get('available_forfaits', {})}
        - Types de durée: {session_state.get('available_types_duree', {})}
        - Groupes pour sélection: {session_state.get('all_groups_for_selection', {})}

        **Instructions**:
        1. **Centralisation des réponses**:
           - Si system_data est fourni, reformulez-le en un message naturel et concis.
           - Évitez les répétitions en ne reproduisant pas les messages déjà générés.
           - Si system_data est vide, traitez l'entrée utilisateur pour l'étape actuelle.
        2. **Interprétation des entrées utilisateur**:
           - Détectez les intentions comme 'revenir' ou 'changer'.
           - Validez l'entrée selon l'étape (ex. nom non vide à l'étape 1).
        3. **Étapes gérées**:
           - Étape 1: Valider le nom (non vide, vérifier si étudiant existe).
           - Étape 2: Valider le niveau (doit correspondre à levels_list).
           - Étape 3: Valider les matières.
           - Étape 4: Valider les notes (facultatif, une note par matière, entre 0 et 20).
           - Étape 5: Valider choix indiv/groupe.
           - Étapes 6-7: Sélection des forfaits et types de durée.
           - Étapes 8-10: Professeurs, école, centre.
           - Étapes 11+: Reformuler recommandations, tarifs, frais.
        4. **Reformulation des données du système**:
           - Pour l'étape 1, confirmez le nom et indiquez si l'étudiant existe.
           - Pour l'étape 2, validez le niveau et passez aux matières.
           - Pour l'étape 4, mappez chaque note à la matière correspondante dans l'ordre, indiquez si une note est manquante (None), et suggérez des cours individuels si note < 8, sinon des groupes.
        5. **Réponse JSON**:
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

        **Réponse**:
        Produisez une réponse JSON cohérente, avec un ton amical, en respectant l'ordre : nom en premier (étape 1), puis niveau (étape 2), et en traitant plusieurs notes à l'étape 4 en les associant aux matières dans l'ordre de saisie.
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
        return {
            "step": step,
            "data": {},
            "message": "Erreur interne, veuillez réessayer.",
            "error": "Erreur API",
            "suggestions": [],
            "next_step": step
        }

def handle_input_submission(step, response_text):
    logger.debug(f"Traitement de l'étape {step} avec entrée : {response_text}")
    logger.debug(f"État actuel de responses : {st.session_state.responses}")
    logger.debug(f"État actuel de matched_subjects : {st.session_state.matched_subjects}")
    
    lists = {
        "levels_list": levels_list,
        "subjects_list": subjects_list,
        "schools_list": schools_list,
        "centers_list": centers_list,
        "teachers_list": teachers_list
    }
    
    # Vérifier les intentions de modification ou retour
    response_lower = response_text.lower().strip()
    change_patterns = [
        (r"changer\s+(le\s+nom|nom\s*de\s*l'étudiant)", 1, 'student_name'),
        (r"changer\s+(le\s+niveau|niveau)", 2, 'user_level'),
        (r"changer\s+(les\s+matières|matières)", 3, 'user_subjects'),
        (r"changer\s+(les\s+notes|notes)", 4, 'grades'),
        (r"changer\s+(les\s+professeurs|professeurs)", 8, 'user_teachers'),
        (r"changer\s+(l'école|école)", 9, 'user_school'),
        (r"changer\s+(le\s+centre|centre)", 10, 'user_center'),
    ]
    return_patterns = [
        (r"revenir\s+(au\s+nom|nom)", 1),
        (r"revenir\s+(au\s+niveau|niveau)", 2),
        (r"revenir\s+(aux\s+matières|matières)", 3),
        (r"revenir\s+(aux\s+notes|notes)", 4),
        (r"revenir\s+(aux\s+professeurs|professeurs)", 8),
        (r"revenir\s+(à\s+l'école|école)", 9),
        (r"revenir\s+(au\s+centre|centre)", 10),
    ]

    for pattern, target_step, field in change_patterns:
        if re.search(pattern, response_lower):
            st.session_state.responses[field] = response_text if field == 'student_name' else ''
            system_data = {"action": "change_field", "field": field, "value": response_text}
            llm_response = process_with_llm(response_text, target_step, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            st.session_state.responses.update(llm_response["data"])
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='user-message'>{response_text}</div>", False))
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
            if llm_response["error"]:
                st.session_state.messages.append((f"<div class='bot-message'>Erreur : {llm_response['error']}</div>", True))
            return

    for pattern, target_step in return_patterns:
        if re.search(pattern, response_lower):
            system_data = {"action": "return_to_step", "target_step": target_step}
            llm_response = process_with_llm(response_text, target_step, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            st.session_state.responses.update(llm_response["data"])
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='user-message'>{response_text}</div>", False))
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
            if llm_response["error"]:
                st.session_state.messages.append((f"<div class='bot-message'>Erreur : {llm_response['error']}</div>", True))
            return

    # Traitement des étapes
    system_data = {}
    if step == 1:  # Nom de l'étudiant
        if response_text.strip():
            students_data = collection_students.get(include=["metadatas"])
            student_exists = any(metadata.get('student_name', '').strip().lower() == response_text.strip().lower() for metadata in students_data['metadatas'])
            system_data = {
                "student_name": response_text,
                "student_exists": student_exists
            }
        else:
            system_data = {"error": "empty_name"}

    elif step == 2:  # Niveau
        if response_text.strip():
            matched_level, is_valid = match_value(response_text.strip(), levels_list)
            system_data = {
                "user_level": matched_level,
                "is_valid": is_valid
            }
        else:
            system_data = {"error": "empty_level"}

    elif step == 3:  # Matières
        if response_text.strip():
            subjects = [s.strip() for s in response_text.split(',')]
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
                system_data = {"subjects": matched_subjects}
            else:
                system_data = {"error": "invalid_subjects", "subjects": subjects}
        else:
            system_data = {"error": "empty_subjects"}

    elif step == 4:  # Notes (facultatif)
        subjects = st.session_state.matched_subjects
        if response_text.strip() and response_text.lower() != 'non':
            try:
                grades = [float(g.strip()) for g in response_text.split(',')]
                if len(grades) > len(subjects):
                    system_data = {
                        "error": "too_many_grades",
                        "num_grades": len(grades),
                        "num_subjects": len(subjects),
                        "message": f"Vous avez fourni {len(grades)} notes, mais il y a seulement {len(subjects)} matière(s) ({', '.join(subjects)}). Veuillez fournir une note par matière ou laissez vide pour certaines."
                    }
                elif any(g < 0 or g > 20 for g in grades):
                    system_data = {"error": "invalid_grades", "message": "Les notes doivent être comprises entre 0 et 20."}
                else:
                    # Compléter avec None si moins de notes que de matières
                    grades_extended = grades + [None] * (len(subjects) - len(grades))
                    subject_grades = dict(zip(subjects, grades_extended))
                    recommendations = []
                    for subject, grade in subject_grades.items():
                        if grade is not None:
                            recommendation = "cours individuels" if grade < 8 else "cours en groupe"
                            recommendations.append({
                                "subject": subject,
                                "grade": grade,
                                "recommendation": recommendation
                            })
                        else:
                            recommendations.append({
                                "subject": subject,
                                "grade": None,
                                "reason": "aucune note fournie",
                                "recommendation": "cours en groupe"
                            })
                    system_data = {
                        "user_grades": response_text,
                        "subject_grades": subject_grades,
                        "recommendations": recommendations
                    }
            except ValueError:
                system_data = {"error": "invalid_grades_format", "message": "Les notes doivent être des nombres séparés par des virgules (ex. 12,15)."}
        else:
            subject_grades = {subject: None for subject in subjects}
            recommendations = [
                {
                    "subject": subject,
                    "grade": None,
                    "reason": "aucune note fournie",
                    "recommendation": "cours en groupe"
                } for subject in subjects
            ]
            system_data = {
                "user_grades": "",
                "subject_grades": subject_grades,
                "recommendations": recommendations
            }

    elif step == 5:  # Choix des cours (indiv/groupe)
        if response_text.strip():
            choices = [c.strip().lower() for c in response_text.split(',')]
            subjects = st.session_state.matched_subjects
            if len(choices) != len(subjects):
                system_data = {"error": "incorrect_choice_count", "num_choices": len(choices), "num_subjects": len(subjects)}
            elif not all(c in ['indiv', 'groupe'] for c in choices):
                system_data = {"error": "invalid_choices"}
            else:
                course_choices = dict(zip(subjects, choices))
                group_subjects = [subject for subject, choice in course_choices.items() if choice == 'groupe']
                indiv_subjects = [subject for subject, choice in course_choices.items() if choice == 'indiv']
                system_data = {
                    "course_choices": choices,
                    "group_subjects": group_subjects,
                    "indiv_subjects": indiv_subjects
                }
        else:
            system_data = {"error": "empty_choices"}

    elif step == 6:  # Choix des forfaits
        if response_text.strip():
            try:
                selections = [int(s.strip()) for s in response_text.split(',')]
                group_subjects = st.session_state.matched_subjects
                if len(selections) != len(group_subjects):
                    system_data = {"error": "incorrect_forfait_count", "num_selections": len(selections), "num_subjects": len(group_subjects)}
                else:
                    selected_forfaits = {}
                    valid = True
                    for subject, selection in zip(group_subjects, selections):
                        forfaits = st.session_state.available_forfaits.get(subject, {})
                        forfait_list = list(forfaits.keys())
                        if 1 <= selection <= len(forfait_list):
                            selected_forfaits[subject] = forfait_list[selection - 1]
                        else:
                            valid = False
                            system_data = {"error": "invalid_forfait_selection", "subject": subject, "selection": selection, "num_forfaits": len(forfait_list)}
                            break
                    if valid:
                        system_data = {"forfait_selections": selected_forfaits}
            except ValueError:
                system_data = {"error": "invalid_forfait_format"}
        else:
            system_data = {"error": "empty_forfaits"}

    elif step == 7:  # Choix des types de durée
        if response_text.strip():
            try:
                selections = [int(s.strip()) for s in response_text.split(',')]
                group_subjects = st.session_state.matched_subjects
                if len(selections) != len(group_subjects):
                    system_data = {"error": "incorrect_type_duree_count", "num_selections": len(selections), "num_subjects": len(group_subjects)}
                else:
                    selected_types_duree = {}
                    valid = True
                    for subject, selection in zip(group_subjects, selections):
                        types_duree = st.session_state.available_types_duree.get(subject, {})
                        type_list = list(types_duree.keys())
                        if 1 <= selection <= len(type_list):
                            selected_types_duree[subject] = type_list[selection - 1]
                        else:
                            valid = False
                            system_data = {"error": "invalid_type_duree_selection", "subject": subject, "selection": selection}
                            break
                    if valid:
                        system_data = {"type_duree_selections": selected_types_duree}
            except ValueError:
                system_data = {"error": "invalid_type_duree_format"}
        else:
            system_data = {"error": "empty_type_duree"}

    elif step == 8:  # Professeurs
        system_data = {"user_teachers": response_text.strip() or None}

    elif step == 9:  # École
        if response_text.strip():
            matched_school, is_valid = match_value(response_text.strip(), schools_list)
            if is_valid:
                system_data = {"user_school": matched_school}
            else:
                system_data = {"error": "invalid_school", "school": response_text}
        else:
            system_data = {"error": "empty_school"}

    elif step == 10:  # Centre
        if response_text.strip():
            matched_center, is_valid = match_value(response_text.strip(), centers_list)
            if is_valid:
                system_data = {"user_center": matched_center}
            else:
                system_data = {"error": "invalid_center", "center": response_text}
        else:
            system_data = {"user_center": None}
            required_fields = ['student_name', 'user_level', 'user_subjects', 'user_school']
            missing_fields = [field for field in required_fields if field not in st.session_state.responses or not st.session_state.responses[field]]
            if missing_fields or not st.session_state.selected_forfaits or not st.session_state.selected_types_duree:
                system_data = {"error": "missing_data", "missing_fields": missing_fields}
            else:
                with st.spinner("Recherche en cours..."):
                    output, all_recommendations, all_groups_for_selection, matched_subjects = get_recommendations(
                        st.session_state.responses['student_name'],
                        st.session_state.responses['user_level'],
                        st.session_state.responses['user_subjects'],
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
                system_data = {
                    "recommendations": all_recommendations,
                    "groups_for_selection": all_groups_for_selection,
                    "matched_subjects": matched_subjects,
                    "output": output
                }

    elif step == 11:  # Sélection des groupes
        if response_text.strip():
            try:
                selections = [int(s.strip()) for s in response_text.split(',')]
                matched_subjects = st.session_state.matched_subjects
                if len(selections) != len(matched_subjects):
                    system_data = {"error": "incorrect_group_count", "num_selections": len(selections), "num_subjects": len(matched_subjects)}
                else:
                    group_selections = {}
                    valid = True
                    for subject, selection in zip(matched_subjects, selections):
                        groups = st.session_state.all_groups_for_selection.get(subject, [])
                        if 1 <= selection <= len(groups):
                            group_selections[subject] = groups[selection - 1]['id_cours']
                        else:
                            valid = False
                            system_data = {"error": "invalid_group_selection", "subject": subject, "selection": selection}
                            break
                    if valid:
                        selected_groups = {}
                        for subject, id_cours in group_selections.items():
                            groups = st.session_state.all_groups_for_selection[subject]
                            selected_group = next((g for g in groups if g['id_cours'] == id_cours), None)
                            if selected_group:
                                selected_groups[subject] = selected_group
                        overlaps = check_overlaps(selected_groups)
                        if overlaps:
                            system_data = {
                                "error": "overlaps",
                                "overlaps": [
                                    {
                                        "group1": {"matiere": g1['matiere'], "name_cours": g1['name_cours'], "jour": g1['jour'], "heure_debut": g1['heure_debut'], "heure_fin": g1['heure_fin']},
                                        "group2": {"matiere": g2['matiere'], "name_cours": g2['name_cours'], "jour": g2['jour'], "heure_debut": g2['heure_debut'], "heure_fin": g2['heure_fin']}
                                    } for g1, g2 in overlaps
                                ],
                                "groups_for_selection": st.session_state.all_groups_for_selection
                            }
                        else:
                            user_duree_types = {subject: st.session_state.available_types_duree[subject][type_id]['name']
                                                for subject, type_id in st.session_state.selected_types_duree.items()}
                            tariffs_by_group, tariff_data, total_tariff_base = calculate_tariffs(
                                selected_groups,
                                user_duree_types,
                                st.session_state.selected_types_duree,
                                st.session_state.available_forfaits
                            )
                            if not tariffs_by_group:
                                system_data = {"error": "tariff_calculation_failed", "message": tariff_data}
                            else:
                                system_data = {
                                    "group_selections": group_selections,
                                    "tariffs": tariff_data,
                                    "selected_groups": selected_groups
                                }
            except ValueError:
                system_data = {"error": "invalid_group_format"}
        else:
            system_data = {"error": "empty_groups"}

    elif step == 12:  # Frais d'inscription
        choice = response_text.strip().lower()
        if choice in ['oui', 'yes', 'non', 'no']:
            system_data = {"frais_choice": choice}
        else:
            system_data = {"error": "invalid_frais_choice"}

    elif step == 13:  # Commentaires
        reduction_keywords = ['réduction', 'reduction', 'remise', 'rabais', 'discount']
        reduction_detected = any(keyword in response_lower for keyword in reduction_keywords)
        system_data = {
            "commentaires": response_text,
            "reduction_detected": reduction_detected
        }

    elif step == 14:  # Réduction
        try:
            percentage = float(response_text.strip())
            if 0 <= percentage <= 100:
                system_data = {"reduction_percentage": percentage}
            else:
                system_data = {"error": "invalid_percentage"}
        except ValueError:
            system_data = {"error": "invalid_percentage_format"}

    elif step == 15:  # Nouveau cas
        choice = response_text.strip().lower()
        if choice in ['oui', 'yes', 'non', 'no']:
            system_data = {"new_case_choice": choice}
        else:
            system_data = {"error": "invalid_new_case_choice"}

    llm_response = process_with_llm(response_text, step, st.session_state, lists, system_data)
    
    # Mettre à jour l'état
    st.session_state.step = llm_response["next_step"]
    st.session_state.responses.update(llm_response["data"])
    
    # Ajouter les messages filtrés
    if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
        st.session_state.messages.append((f"<div class='user-message'>{response_text}</div>", False))
        st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
        if llm_response["error"]:
            st.session_state.messages.append((f"<div class='bot-message'>Erreur : {llm_response['error']}</div>", True))

    # Gestion des étapes spécifiques
    if llm_response["step"] == 5 and "course_choices" in llm_response["data"]:
        course_choices = llm_response["data"]["course_choices"]
        subjects = st.session_state.matched_subjects
        group_subjects = [subject for subject, choice in zip(subjects, course_choices) if choice.lower() == 'groupe']
        indiv_subjects = [subject for subject, choice in zip(subjects, course_choices) if choice.lower() == 'indiv']
        st.session_state.course_choices = dict(zip(subjects, course_choices))
        
        if group_subjects:
            st.session_state.matched_subjects = group_subjects
            st.session_state.responses['user_subjects'] = ', '.join(group_subjects)
            st.session_state.available_forfaits = {}
            forfaits_data = {}
            for subject in group_subjects:
                forfaits = get_available_forfaits(st.session_state.responses['user_level'], subject)
                st.session_state.available_forfaits[subject] = forfaits
                forfaits_data[subject] = {id_forfait: info['name'] for id_forfait, info in forfaits.items()}
            system_data = {"forfaits": forfaits_data, "indiv_subjects": indiv_subjects}
            llm_response = process_with_llm("", 6, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            st.session_state.responses.update(llm_response["data"])
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
        else:
            system_data = {"action": "no_group_subjects", "indiv_subjects": indiv_subjects}
            llm_response = process_with_llm("", 15, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))

    elif llm_response["step"] == 6 and "forfait_selections" in llm_response["data"]:
        st.session_state.selected_forfaits = llm_response["data"]["forfait_selections"]
        types_duree_data = {}
        valid_subjects = []
        for subject, id_forfait in st.session_state.selected_forfaits.items():
            if subject in st.session_state.available_forfaits and id_forfait in st.session_state.available_forfaits[subject]:
                types_duree = st.session_state.available_forfaits[subject][id_forfait]['types_duree']
                st.session_state.available_types_duree[subject] = types_duree
                types_duree_data[subject] = {type_id: info['name'] for type_id, info in types_duree.items()}
                valid_subjects.append(subject)
        st.session_state.matched_subjects = valid_subjects
        system_data = {"types_duree": types_duree_data}
        llm_response = process_with_llm("", 7, st.session_state, lists, system_data)
        st.session_state.step = llm_response["next_step"]
        st.session_state.responses.update(llm_response["data"])
        if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
            st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))

    elif llm_response["step"] == 7 and "type_duree_selections" in llm_response["data"]:
        st.session_state.selected_types_duree = llm_response["data"]["type_duree_selections"]

    elif llm_response["step"] == 12 and "frais_choice" in llm_response["data"]:
        choice = llm_response["data"]["frais_choice"]
        if choice in ['oui', 'yes']:
            frais_inscription = 250 * len(st.session_state.matched_subjects)
            st.session_state.total_with_frais = st.session_state.total_tariff_base + frais_inscription
            system_data = {"frais_inscription": frais_inscription, "total_with_frais": st.session_state.total_with_frais}
            llm_response = process_with_llm(response_text, 13, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
        elif choice in ['non', 'no']:
            st.session_state.total_with_frais = st.session_state.total_tariff_base
            system_data = {"no_frais": True, "total": st.session_state.total_tariff_base}
            llm_response = process_with_llm(response_text, 13, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))

    elif llm_response["step"] == 15 and "new_case_choice" in llm_response["data"]:
        choice = llm_response["data"]["new_case_choice"]
        if choice in ['oui', 'yes']:
            system_data = {"action": "new_case"}
            llm_response = process_with_llm(response_text, 1, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            st.session_state.responses = {}
            st.session_state.selected_groups = {}
            st.session_state.tariffs_by_group = {}
            st.session_state.total_tariff_base = 0
            st.session_state.available_forfaits = {}
            st.session_state.available_types_duree = {}
            st.session_state.selected_forfaits = {}
            st.session_state.selected_types_duree = {}
            st.session_state.reduction_percentage = 0
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
        elif choice in ['non', 'no']:
            system_data = {"action": "end_conversation"}
            llm_response = process_with_llm(response_text, 0, st.session_state, lists, system_data)
            st.session_state.step = llm_response["next_step"]
            if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
                st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))

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
        st.session_state.messages = [("<div class='bot-message'>Bonjour ! Quel est le nom de l'étudiant ?</div>", True)]
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
    st.session_state.messages = [("<div class='bot-message'>Bonjour ! Quel est le nom de l'étudiant ?</div>", True)]
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
    "Quel est le nom de l'étudiant ?",
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
    "Ex: Ahmed Larache",
    f"Ex: {', '.join(levels_list[:3]) if levels_list else 'BL - 2bac sc PC'}",
    f"Ex: {', '.join(subjects_list[:3]) if subjects_list else 'Français, Mathématiques'}",
    "Ex: 12, 15 (ou laissez vide, une note par matière)",
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
    llm_response = process_with_llm("", 1, st.session_state, {
        "levels_list": levels_list,
        "subjects_list": subjects_list,
        "schools_list": schools_list,
        "centers_list": centers_list,
        "teachers_list": teachers_list
    }, {"action": "start_conversation", "request": "student_name"})
    st.session_state.step = llm_response["next_step"]
    if filter_redundant_messages(st.session_state.messages, llm_response["message"]):
        st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
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
        