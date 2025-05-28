# chatbot_project/tools/functions.py
from tools.chroma_utils import collection_groupes, collection_combinaisons, collection_seances, collection_students, model, niveaux, schools, subjects, centers, teachers,all_groups,available_subjects,available_levels
from datetime import datetime, timedelta
from rapidfuzz import process
import logging
import random

logger = logging.getLogger(__name__)

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

def match_value(user_input, valid_values):
    if not user_input or not valid_values:
        return user_input, False
    result = process.extractOne(user_input, valid_values)
    if result is None:
        return user_input, False
    best_match, score = result
    return best_match if score > 80 else user_input, score > 80


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
            if id_forfait and id_forfait not in forfaits:
                forfaits[id_forfait] = {'name': nom_forfait}

    if not forfaits:
        logger.warning(f"Aucun forfait trouvé pour niveau: '{level}', matière: '{subject}'")
    return forfaits


def get_types_duree_forfait(id_forfait):
    logger.debug(f"Récupération des types durée pour le forfait: '{id_forfait}'")
    types_duree = {}
    seen = set()
    all_groups = collection_groupes.get(include=["metadatas"])

    for metadata in all_groups['metadatas']:
        if metadata.get('id_forfait') == id_forfait:
            duree_tarifs = metadata.get('duree_tarifs', '')
            try:
                duree_entries = duree_tarifs.split(';')
                for entry in duree_entries:
                    if entry:
                        parts = entry.split(':')
                        if len(parts) == 3:
                            type_duree, entry_id_forfait, tarif = parts
                            key = (type_duree.strip().lower(), float(tarif))
                            if entry_id_forfait == id_forfait and key not in seen:
                                type_duree_id = f"{id_forfait}_{len(seen)+1}"
                                types_duree[type_duree_id] = {
                                    'name': type_duree,
                                    'tarif_unitaire': float(tarif)
                                }
                                seen.add(key)
            except (ValueError, TypeError) as e:
                logger.error(f"Erreur lors du parsing de duree_tarifs pour {id_forfait}: {str(e)}")

    return types_duree

def count_school_students(group_schools, user_school):
    return sum(1 for school in group_schools if school == user_school)


def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%H:%M")
    except Exception:
        return None

def has_overlap(group1, group2):
    valid1, _ = validate_group_structure(group1)
    valid2, _ = validate_group_structure(group2)
    if not valid1 or not valid2:
        return False
    if group1['jour'] != group2['jour']:
        return False
    start1 = parse_time(group1['heure_debut'])
    end1 = parse_time(group1['heure_fin'])
    start2 = parse_time(group2['heure_debut'])
    end2 = parse_time(group2['heure_fin'])
    if not all([start1, end1, start2, end2]):
        return False
    same_center = group1['centre'] == group2['centre']
    margin = timedelta(minutes=15) if not same_center else timedelta(0)
    return (start1 < end2 + margin) and (start2 < end1 + margin)


def check_overlaps(selected_groups):
    if not isinstance(selected_groups, dict):
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


def calculate_tariffs(selected_groups, user_type_duree_ids, forfaits_info):
    tariffs_by_group = {}
    total_tariff_base = 0
    reduction_applied = 0
    reduction_description = ""
    selected_forfait_ids = []

    for subject, group in selected_groups.items():
        user_type_duree_id = user_type_duree_ids[subject]
        id_forfait = group['id_forfait']
        nom_forfait = forfaits_info[subject][id_forfait]['name']
        groupe_data = collection_groupes.get(ids=[group['id_cours']], include=["metadatas"])
        if not groupe_data['metadatas']:
            return None, f"Erreur : Données non trouvées pour le cours {group['id_cours']}.", None

        metadata = groupe_data['metadatas'][0]
        type_duree_id = metadata.get('type_duree_id')
        tarif_unitaire = metadata.get('tarifunitaire')
        if metadata.get('id_forfait') is None or type_duree_id != user_type_duree_id or tarif_unitaire is None:
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
                reduction_description = f"<b>Réduction pour combinaison :</b> -{reduction_amount:.2f} DH ({reduction_percentage:.2f}%)"
                break

    tariff_message = "<b>Détails des tarifs :</b><br>"
    for subject, info in tariffs_by_group.items():
        tariff_message += f"<b>- {subject} :</b> {info['remaining_sessions']} séances restantes, tarif unitaire {info['tarif_unitaire']} DH, tarif total {info['tarif_total']:.2f} DH<br>"
    tariff_message += f"<b>- Total de base :</b> {sum(info['tarif_total'] for info in tariffs_by_group.values()):.2f} DH<br>"
    if reduction_applied > 0:
        tariff_message += f"- {reduction_description}<br>"
        tariff_message += f"<b>- Total après réduction :</b> {total_tariff_base:.2f} DH"
    else:
        tariff_message += f"<b>- Total :</b> {total_tariff_base:.2f} DH"

    return tariffs_by_group, tariff_message, total_tariff_base


def add_enseignant(nom, commentaire):
    collection_students.add(
        documents=[commentaire],
        metadatas=[{"enseignant": nom, "commentaire": commentaire}],
        ids=[f"enseignant-{nom.lower().replace(' ', '-')}"]
    )
    return True


def get_comments(enseignant):
    results = collection_students.get(where={"enseignant": enseignant})
    return [meta.get("commentaire", "") for meta in results.get("metadatas", [])]

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
    matched_level = match_value(user_level, niveaux)[0]
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
    rejected_groups = []

    for matched_subject, matched_teacher in zip(matched_subjects, matched_teachers):
        id_forfait = selected_forfaits.get(matched_subject)
        type_duree_id = selected_types_duree.get(matched_subject)
        if not id_forfait or not type_duree_id:
            output.append(f"Aucun forfait ou type de durée sélectionné pour {matched_subject}.")
            continue

        groups = {}
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
            output.append(f"<b>Attention</b> : Seulement {len(selected_groups)} groupe(s) trouvé(s) pour {matched_subject}.")

        recommendations = []
        groups_for_selection = []
        for i, group in enumerate(selected_groups[:3], 1):
            recommendation = (
                f"<h4>Groupe {i} ({matched_subject})</h4>"
                f"<b>Nom:</b> {group['name_cours']}<br>"
                f"<b>Forfait:</b> {group['nom_forfait']}<br>"
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

    output.append(f"<b>Les groupes recommandés pour l'étudiant {student_name} :</b>")
    return output, all_recommendations, all_groups_for_selection, matched_subjects
