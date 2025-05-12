import streamlit as st
import json
import chromadb
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime, timedelta
import os
import random

# Configuration initiale
st.set_page_config(page_title="Chatbot de Recommandation de Groupes", page_icon="📚", layout="wide")

# Configuration de l'API Gemini
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
    if not GOOGLE_API_KEY:
        st.warning("Clé API codée en dur utilisée. Pour la sécurité, configurez GOOGLE_API_KEY dans secrets.toml ou les variables d'environnement.")
    
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

# Vérification des collections
if collection_groupes.count() == 0:
    st.error("Erreur : La collection ChromaDB des groupes est vide. Veuillez exécuter la vectorisation d'abord.")
    st.stop()
if collection_combinaisons.count() == 0:
    st.error("Erreur : La collection ChromaDB des combinaisons est vide. Veuillez exécuter la vectorisation des combinaisons d'abord.")
    st.stop()

# Fonction pour charger le modèle SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Récupérer les valeurs uniques depuis ChromaDB
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
def get_available_forfaits(level, subject):
    forfaits = {}
    all_groups = collection_groupes.get(include=["metadatas"])
    target_subject = subject.strip().lower()
    available_subjects = list(set(metadata['matiere'].strip().lower() for metadata in all_groups['metadatas'] if metadata.get('matiere')))
    
    matched_subject = target_subject
    if available_subjects:
        best_match, score = process.extractOne(target_subject, available_subjects)
        if score > 80:
            matched_subject = best_match
    
    for metadata in all_groups['metadatas']:
        metadata_niveau = metadata.get('niveau', '').strip().lower()
        metadata_matiere = metadata.get('matiere', '').strip().lower()
        if metadata_niveau == level.strip().lower() and metadata_matiere == matched_subject:
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
                    except (ValueError, TypeError):
                        pass
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
    if group1.get('jour') != group2.get('jour'):
        return False
    start1 = parse_time(group1.get('heure_debut'))
    end1 = parse_time(group1.get('heure_fin'))
    start2 = parse_time(group2.get('heure_debut'))
    end2 = parse_time(group2.get('heure_fin'))
    if not all([start1, end1, start2, end2]):
        return False
    if group1.get('centre') == group2.get('centre'):
        return start1 < end2 and start2 < end1 and (end1 != start2 or start1 != end2)
    else:
        margin = timedelta(minutes=15)
        return start1 < end2 + margin and start2 < end1 + margin

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
    output = []
    matched_level = match_value(user_level, levels_list)[0]
    matched_subjects = [match_value(subj.strip(), subjects_list)[0] for subj in user_subjects.split(",")]
    matched_teachers = [teacher.strip() for teacher in user_teachers.split(",")] if user_teachers else [None] * len(matched_subjects)
    matched_school = match_value(user_school, schools_list)[0]
    matched_center = match_value(user_center, centers_list)[0] if user_center else None

    all_recommendations = {}
    all_groups_for_selection = {}

    for matched_subject, matched_teacher in zip(matched_subjects, matched_teachers):
        id_forfait = selected_forfaits.get(matched_subject)
        type_duree_id = selected_types_duree.get(matched_subject)
        if not id_forfait or not type_duree_id:
            output.append(f"Aucun forfait ou type de durée sélectionné pour {matched_subject}.")
            continue

        all_groups_data = collection_groupes.get(include=["metadatas", "documents"])
        groups = {}
        for metadata, document in zip(all_groups_data['metadatas'], all_groups_data['documents']):
            if (metadata['niveau'].strip().lower() == matched_level.strip().lower() and
                metadata['matiere'].strip().lower() == matched_subject.strip().lower() and
                metadata.get('id_forfait') == id_forfait and
                metadata.get('type_duree_id') == type_duree_id):
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
                    "matiere": metadata['matiere'],
                    "id_forfait": metadata['id_forfait'],
                    "nom_forfait": metadata['nom_forfait'],
                    "type_duree_id": metadata['type_duree_id']
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
        Vous êtes un chatbot Streamlit de recommandation de groupes éducatifs, aidant les utilisateurs à trouver des groupes d’apprentissage adaptés (niveau, matière, professeur, école, centre, forfait, type de durée) et à calculer les tarifs avec réductions. Les données sont extraites d’une base ChromaDB. Le flux conversationnel comporte 15 étapes.

        **Étape actuelle**: {step}
        **Entrée utilisateur**: '{input_text}'
        **Réponses actuelles**: {json.dumps(session_state.responses)}
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
        1. Analyser l’entrée pour extraire les données et intentions (réponse normale, hors séquence, retour comme 'revenir à [étape]', modification comme 'changer [champ]', demande de réduction).
        2. Valider les entrées avec un seuil de correspondance (80%) pour niveaux, matières, écoles, centres.
        3. Si des données sont manquantes (entrée vide, partielle, ou invalide), redemander spécifiquement les informations manquantes.
        4. À l’étape 4 (notes):
           - Si une note < 8: "Note faible en [matière] ([note]/20). Nous recommandons des cours individuels pour une mise à niveau efficace."
           - Si toutes les notes ≥ 8 ou absentes: "Note ([notes]) ou aucune note. Nous recommandons des cours individuels pour un enseignement personnalisé."
           - Proposer: "Veuillez choisir le type de cours pour chaque matière (indiv/groupe)."
        5. Gérer les retours ('revenir à matières' → étape 3) et modifications ('changer les matières' → étape 3, effacer les réponses associées).
        6. Détecter les demandes de réduction à l’étape 13 (mots comme 'réduction', 'remise', 'rabais' avec score > 90) et passer à l’étape 14.
        7. Générer un message naturel, amical, adapté au nom de l’étudiant ({session_state.responses.get('student_name', '')}).
        8. Retourner une réponse JSON au format:
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

        **Étapes et attentes**:
        1. Nom: Texte non vide.
        2. Niveau: Valider avec levels_list.
        3. Matières: Liste séparée par virgules, valider avec subjects_list.
        4. Notes: Nombres ou vide, même longueur que les matières, proposer cours individuels.
        5. Type de cours: 'indiv'/'groupe' par matière.
        6. Forfait: Indices valides dans available_forfaits.
        7. Type de durée: Indices valides dans available_types_duree.
        8. Professeurs: Texte ou vide (facultatif).
        9. École: Valider avec schools_list.
        10. Centre: Valider avec centers_list ou vide (facultatif).
        11. Groupes: Indices valides dans all_groups_for_selection.
        12. Frais: 'Oui'/'Non'.
        13. Commentaires: Détecter 'réduction' (score > 90).
        14. Pourcentage: Nombre entre 0 et 100.
        15. Autre cas: 'Oui'/'Non'.

        **Exemples**:
        - Étape 1, Entrée: "Ahmed" → {{"step": 1, "data": {{"student_name": "Ahmed"}}, "message": "Bonjour, Ahmed ! Quel est le niveau de l’étudiant ?", "error": null, "suggestions": [], "next_step": 2}}
        - Étape 4, Entrée: "6", Contexte: matched_subjects=["Mathématiques"] → {{"step": 4, "data": {{"grades": [6]}}, "message": "Note faible en Mathématiques (6/20). Nous recommandons des cours individuels pour une mise à niveau efficace. Veuillez choisir le type de cours pour Mathématiques (indiv/groupe).", "error": null, "suggestions": ["indiv", "groupe"], "next_step": 5}}
        - Retour: "revenir aux matières" → {{"step": -1, "data": {{"target_step": 3}}, "message": "On revient aux matières. Quelles matières souhaitez-vous ?", "error": null, "suggestions": [], "next_step": 3}}

        **Réponse**:
        Analysez l’entrée et fournissez une réponse JSON conforme.
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
            return response
        except json.JSONDecodeError as e:
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
def handle_input_submission(step, response_text):
    lists = {
        "levels_list": levels_list,
        "subjects_list": subjects_list,
        "schools_list": schools_list,
        "centers_list": centers_list,
        "teachers_list": teachers_list
    }
    llm_response = process_with_llm(response_text, step, st.session_state, lists)
    
    # Mettre à jour l'état
    st.session_state.step = llm_response["next_step"]
    st.session_state.responses.update(llm_response["data"])
    
    # Ajouter le message de l'utilisateur
    st.session_state.messages.append((f"<div class='user-message'>{response_text}</div>", False))
    
    # Ajouter le message du bot
    st.session_state.messages.append((f"<div class='bot-message'>{llm_response['message']}</div>", True))
    if llm_response["error"]:
        st.session_state.messages.append((f"<div class='bot-message'>Erreur : {llm_response['error']}</div>", True))
    
    # Post-traitement pour étapes spécifiques
    if llm_response["step"] == 5 and "course_choices" in llm_response["data"]:
        group_subjects = [subject for subject, choice in zip(st.session_state.matched_subjects, llm_response["data"]["course_choices"]) if choice == "groupe"]
        indiv_subjects = [subject for subject, choice in zip(st.session_state.matched_subjects, llm_response["data"]["course_choices"]) if choice == "indiv"]
        st.session_state.course_choices = dict(zip(st.session_state.matched_subjects, llm_response["data"]["course_choices"]))
        
        for subject in indiv_subjects:
            st.session_state.messages.append((f"<div class='bot-message'>Les cours individuels pour {subject} sont en préparation et seront disponibles ultérieurement.</div>", True))
        
        if group_subjects:
            st.session_state.matched_subjects = group_subjects
            st.session_state.responses['user_subjects'] = ', '.join(group_subjects)
            st.session_state.available_forfaits = {}
            for subject in group_subjects:
                forfaits = get_available_forfaits(st.session_state.responses['user_level'], subject)
                st.session_state.available_forfaits[subject] = forfaits
    elif llm_response["step"] == 6 and "forfait_selections" in llm_response["data"]:
        st.session_state.selected_forfaits = llm_response["data"]["forfait_selections"]
        st.session_state.available_types_duree = {}
        for subject, id_forfait in st.session_state.selected_forfaits.items():
            types_duree = st.session_state.available_forfaits[subject][id_forfait]['types_duree']
            st.session_state.available_types_duree[subject] = types_duree
    elif llm_response["step"] == 10 and llm_response["next_step"] == 11:
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
        for msg in output:
            st.session_state.messages.append((f"<div class='bot-message'>{msg}</div>", True))
        for subject in matched_subjects:
            for rec in all_recommendations.get(subject, []):
                st.session_state.messages.append((f"<div class='bot-message'>{rec}</div>", True))
    elif llm_response["step"] == 11 and "group_selections" in llm_response["data"]:
        st.session_state.selected_groups = llm_response["data"]["group_selections"]
        overlaps = check_overlaps(st.session_state.selected_groups)
        if overlaps:
            conflict_msg = "<div class='bot-message'><b>Conflit détecté :</b><br>"
            for g1, g2 in overlaps:
                conflict_msg += f"- Chevauchement entre {g1['matiere']} ({g1['display']}) et {g2['matiere']} ({g2['display']}) : "
                conflict_msg += f"{g1['jour']} {g1['heure_debut']}-{g1['heure_fin']} vs {g2['jour']} {g2['heure_debut']}-{g2['heure_fin']}<br>"
            conflict_msg += "Veuillez choisir une nouvelle combinaison de groupes :</div>"
            st.session_state.messages.append((conflict_msg, True))
            st.session_state.step = 11
        else:
            user_duree_types = {subject: st.session_state.available_types_duree[subject][type_id]['name']
                               for subject, type_id in st.session_state.selected_types_duree.items()}
            tariffs_by_group, tariff_message, total_tariff_base = calculate_tariffs(
                st.session_state.selected_groups,
                user_duree_types,
                st.session_state.selected_types_duree,
                st.session_state.available_forfaits
            )
            if tariffs_by_group:
                st.session_state.tariffs_by_group = tariffs_by_group
                st.session_state.total_tariff_base = total_tariff_base
                st.session_state.messages.append((f"<div class='bot-message'>{tariff_message}</div>", True))

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
    "Quel est le nom de l'étudiant ?",
    "Quel est le niveau de l'étudiant (ex. Terminale, Première) ?",
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
    "Ex: Ahmed",
    f"Ex: {', '.join(levels_list[:3]) if levels_list else 'Terminale, Première'}",
    f"Ex: {', '.join(subjects_list[:3]) if subjects_list else 'Français, Maths'}",
    "Ex: 12, 15 (ou laissez vide)",
    "Ex: indiv, groupe",
    "Ex: 1, 2 (numéro du forfait)",
    "Ex: 1, 2 (numéro du type de durée)",
    "Ex: John Doe, Jane Smith (facultatif, peut être vide ou partiel)",
    f"Ex: {schools_list[0] if schools_list else 'Massignon Bouskoura'}",
    f"Ex: {centers_list[0] if centers_list else 'Centre A'} (ou laissez vide)",
    "Ex: 1, 2",
    "Oui ou Non",
    "Ex: J'aimerais une réduction",
    "Ex: 10 (entre 0 et 100)",
    "Oui ou Non"
]

# Logique conversationnelle
if st.session_state.step == 0:
    st.session_state.messages = [("<div class='bot-message'>Bonjour ,Quel est le nom de l'étudiant ?</div>", True)]
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