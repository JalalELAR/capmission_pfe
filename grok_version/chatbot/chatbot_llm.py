# Imports originaux

import streamlit as st
from sentence_transformers import SentenceTransformer # Gardé si vous l'utilisez ailleurs, sinon peut être enlevé
from fuzzywuzzy import process
import random
import chromadb
from datetime import datetime, timedelta
import os
import json # NOUVEAU: Pour parser les réponses JSON du LLM

# NOUVEAU: Import pour Gemini
import google.generativeai as genai

# --- Configuration ChromaDB (inchangée) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
chroma_path = os.path.join(parent_dir, "chroma_db5")
client = chromadb.PersistentClient(path=chroma_path)

try:
    collection_groupes = client.get_collection(name="groupes_vectorises9")
    collection_seances = client.get_or_create_collection(name="seances_vectorises")
    collection_combinaisons = client.get_or_create_collection(name="combinaisons_vectorises")
    collection_students = client.get_or_create_collection(name="students_vectorises")

    if collection_groupes.count() == 0:
        st.error("Erreur : La collection ChromaDB des groupes est vide.")
        st.stop()
    if collection_combinaisons.count() == 0:
        st.error("Erreur : La collection ChromaDB des combinaisons est vide.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de ChromaDB : {e}")
    st.stop()

# --- Chargement des listes de validation (inchangé) ---
all_groups = collection_groupes.get(include=["metadatas", "documents"])
schools = set()
levels = set()
subjects = set()
centers = set()
teachers = set()
for metadata in all_groups.get('metadatas', []):
    try:
        if metadata.get('ecole'): schools.add(metadata['ecole'].split(", ")[0])
        if metadata.get('niveau'): levels.add(metadata['niveau'])
        if metadata.get('matiere'): subjects.add(metadata['matiere'])
        if metadata.get('centre'): centers.add(metadata['centre'])
        if metadata.get('teacher'): teachers.add(metadata['teacher'])
    except (KeyError, IndexError, AttributeError):
        continue # Ignorer les métadonnées incomplètes

schools_list = sorted(list(schools))
levels_list = sorted(list(levels))
subjects_list = sorted(list(subjects))
centers_list = sorted(list(centers))
teachers_list = sorted(list(teachers))

# NOUVEAU: Configuration de Gemini
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Ou un autre modèle Gemini
except KeyError:
    st.error("Clé API Gemini non trouvée. Veuillez configurer GEMINI_API_KEY dans st.secrets.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors de la configuration de Gemini: {e}")
    st.stop()

# --- Fonctions métier (get_available_forfaits, match_value, etc. - inchangées dans leur logique interne) ---
# Assurez-vous qu'elles sont appelées avec les bonnes données depuis st.session_state
def get_available_forfaits(level, subject):
    # ... (code inchangé, mais attention aux appels futurs depuis process_user_turn) ...
    forfaits = {}
    all_groups_data = collection_groupes.get(include=["metadatas"]) # Renommé pour clarté
    # Normaliser la matière pour la recherche
    target_subject = subject.strip().lower()
    # Liste des matières disponibles pour la correspondance approximative
    available_subjects = list(set(metadata['matiere'].strip().lower() for metadata in all_groups_data['metadatas'] if metadata.get('matiere')))

    # Recherche approximative si la correspondance exacte échoue
    matched_subject = target_subject
    if available_subjects:
        best_match_tuple = process.extractOne(target_subject, available_subjects)
        if best_match_tuple and best_match_tuple[1] > 80: # Vérifier si extractOne retourne quelque chose
             best_match = best_match_tuple[0]
             score = best_match_tuple[1]
             matched_subject = best_match
             # print(f"Correspondance approximative : '{subject}' -> '{best_match}' (score: {score})") # Optionnel

    for metadata in all_groups_data['metadatas']:
        metadata_niveau = metadata.get('niveau', '').strip().lower()
        metadata_matiere = metadata.get('matiere', '').strip().lower()
        if (metadata_niveau == level.strip().lower() and
            metadata_matiere == matched_subject):
            id_forfait = metadata.get('id_forfait')
            nom_forfait = metadata.get('nom_forfait', 'Forfait inconnu')
            type_duree_id = metadata.get('type_duree_id') # Assumer que c'est déjà l'ID correct
            nom_type_duree = metadata.get('nom_type_duree') # Nom du type de durée
            tarif_unitaire = metadata.get('tarifunitaire') # Tarif associé

            # S'assurer que id_forfait et type_duree_id existent
            if id_forfait and type_duree_id and nom_type_duree is not None and tarif_unitaire is not None:
                if id_forfait not in forfaits:
                    forfaits[id_forfait] = {
                        'name': nom_forfait,
                        'types_duree': {}
                    }
                # Ajouter le type de durée s'il n'existe pas déjà pour ce forfait
                if type_duree_id not in forfaits[id_forfait]['types_duree']:
                     try:
                         forfaits[id_forfait]['types_duree'][type_duree_id] = {
                             'name': nom_type_duree,
                             'tarif_unitaire': float(tarif_unitaire)
                         }
                     except (ValueError, TypeError):
                          print(f"Erreur de tarif pour {id_forfait}/{type_duree_id}: {tarif_unitaire}") # Log pour debug

    # if not forfaits:
    #     print(f"Aucun forfait trouvé pour niveau='{level}', matière='{subject}' (matched_subject='{matched_subject}')")
    # elif not any(forfait.get('types_duree') for forfait in forfaits.values()):
    #     print(f"Forfaits trouvés pour '{subject}', mais aucun type de durée disponible : {forfaits.keys()}")
    return forfaits


def match_value(user_input, valid_values):
    # ... (code inchangé) ...
    if not user_input or not valid_values:
        return user_input, False
    # Nettoyer l'input
    cleaned_input = str(user_input).strip()
    if not cleaned_input:
        return user_input, False

    result = process.extractOne(cleaned_input, valid_values)
    if result is None:
        return cleaned_input, False
    best_match, score = result
    # Ajouter une vérification de score minimum, par exemple 80 ou 85
    return best_match if score > 80 else cleaned_input, score > 80

def count_school_students(group_schools, user_school):
    # ... (code inchangé) ...
    if not isinstance(group_schools, list): # Petite robustesse
        return 0
    return sum(1 for school in group_schools if school == user_school)

def parse_time(time_str):
    # ... (code inchangé) ...
    try:
        return datetime.strptime(time_str, "%H:%M")
    except (ValueError, TypeError):
         return None # Gérer les erreurs de format

def has_overlap(group1, group2):
    # ... (code inchangé, mais vérifier les données d'entrée) ...
    if not all(k in group1 and k in group2 for k in ['jour', 'heure_debut', 'heure_fin', 'centre']):
        return False # Données incomplètes
    if group1['jour'] != group2['jour'] or group1.get('jour') is None:
        return False

    start1 = parse_time(group1['heure_debut'])
    end1 = parse_time(group1['heure_fin'])
    start2 = parse_time(group2['heure_debut'])
    end2 = parse_time(group2['heure_fin'])

    if not all([start1, end1, start2, end2]): # Si une heure est mal formatée
        return False

    if group1['centre'] == group2['centre']:
        # Chevauchement strict si centres identiques
        return start1 < end2 and start2 < end1
    else:
        # Chevauchement avec marge si centres différents
        margin = timedelta(minutes=15)
        # Vérifier si l'intervalle [start1, end1+margin] chevauche [start2, end2+margin]
        # Ou plus simplement: start1 < end2 + margin AND start2 < end1 + margin
        return start1 < (end2 + margin) and start2 < (end1 + margin)


def check_overlaps(selected_groups_details): # Prend maintenant les détails complets
    overlaps = []
    group_list = list(selected_groups_details.values())
    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            if has_overlap(group_list[i], group_list[j]):
                overlaps.append((group_list[i], group_list[j])) # Retourner les détails des groupes
    return overlaps

def get_remaining_sessions(id_cours):
    # ... (code inchangé) ...
    reference_date_fixed = datetime.strptime("2025/03/06", "%Y/%m/%d") # Garder la date de référence
    try:
        seances_data = collection_seances.get(include=["metadatas"])
        id_cours_str = str(id_cours) # Assurer que c'est une chaîne
        relevant_seances = [
            metadata for metadata in seances_data.get('metadatas', [])
            if metadata.get('id_cours') == id_cours_str and metadata.get('date_seance')
        ]

        remaining = 0
        for metadata in relevant_seances:
            try:
                seance_date = datetime.strptime(metadata['date_seance'], "%Y/%m/%d")
                if seance_date > reference_date_fixed:
                    remaining += 1
            except (ValueError, TypeError):
                print(f"Date de séance invalide pour {id_cours_str}: {metadata['date_seance']}")
                continue # Ignorer les dates invalides
        return remaining
    except Exception as e:
        print(f"Erreur dans get_remaining_sessions pour {id_cours}: {e}")
        return 0 # Retourner 0 en cas d'erreur majeure

def calculate_tariffs(selected_groups_details): # Prend les détails des groupes sélectionnés
    tariffs_by_group = {}
    total_tariff_base_before_reduc = 0
    reduction_applied = 0
    reduction_description = ""
    selected_forfait_ids = []

    if not selected_groups_details:
        return {}, "Aucun groupe sélectionné pour calculer les tarifs.", 0, 0

    for subject, group_details in selected_groups_details.items():
        id_cours = group_details.get('id_cours')
        id_forfait = group_details.get('id_forfait')
        type_duree_id = group_details.get('type_duree_id')
        nom_forfait = group_details.get('nom_forfait', 'Inconnu')
        nom_type_duree = group_details.get('nom_type_duree', 'Inconnu') # Récupérer le nom

        if not id_cours or not id_forfait or not type_duree_id:
            return None, f"Erreur: Données manquantes pour le groupe de {subject} (id_cours, id_forfait, type_duree_id).", 0, 0

        # Récupérer le tarif unitaire (il devrait être dans les détails passés ou refaire une requête)
        # Idéalement, get_recommendations ou l'étape de sélection devrait stocker toutes les infos nécessaires
        # Ici, on suppose qu'il est dans group_details, sinon il faudrait le rechercher
        tarif_unitaire = group_details.get('tarif_unitaire')
        if tarif_unitaire is None:
             # Tentative de récupération depuis la DB si manquant
             group_data_db = collection_groupes.get(ids=[id_cours], include=["metadatas"])
             if group_data_db and group_data_db['metadatas']:
                  metadata_db = group_data_db['metadatas'][0]
                  # Trouver le bon tarif pour le type_duree_id spécifique
                  # (La structure DB actuelle rend cela complexe, on suppose tarifunitaire direct)
                  tarif_unitaire = metadata_db.get('tarifunitaire') # Utilise le tarif unitaire stocké directement
                  if tarif_unitaire is None:
                     return None, f"Erreur : Tarif unitaire non trouvé pour le cours {id_cours}.", 0, 0
             else:
                 return None, f"Erreur : Groupe {id_cours} non trouvé dans la DB pour récupérer le tarif.", 0, 0

        try:
            tarif_unitaire_float = float(tarif_unitaire)
        except (ValueError, TypeError):
            return None, f"Erreur : Tarif unitaire invalide ({tarif_unitaire}) pour le cours {id_cours}.", 0, 0

        selected_forfait_ids.append(str(id_forfait)) # Assurer que c'est une string pour la comparaison
        remaining_sessions = get_remaining_sessions(id_cours)
        tarif_total = remaining_sessions * tarif_unitaire_float

        tariffs_by_group[subject] = {
            "id_cours": id_cours,
            "id_forfait": id_forfait,
            "nom_forfait": nom_forfait,
            "type_duree_id": type_duree_id,
            "nom_type_duree": nom_type_duree, # Ajouter le nom
            "remaining_sessions": remaining_sessions,
            "tarif_unitaire": tarif_unitaire_float,
            "tarif_total": tarif_total
        }
        total_tariff_base_before_reduc += tarif_total

    total_after_auto_reduc = total_tariff_base_before_reduc

    # Application de la réduction de combinaison (logique inchangée, mais attention aux types d'ID)
    if len(selected_forfait_ids) > 1:
        try:
            combinaisons_data = collection_combinaisons.get(include=["metadatas"])
            if combinaisons_data and combinaisons_data['metadatas']:
                combinaisons_dict = {}
                for metadata in combinaisons_data['metadatas']:
                    id_combinaison = metadata.get('id_combinaison')
                    id_forfait_comb = str(metadata.get('id_forfait')) # Comparer string avec string
                    reduction_comb = metadata.get('reduction')
                    if id_combinaison and id_forfait_comb and reduction_comb is not None:
                        try:
                            reduction_float = float(reduction_comb)
                            if id_combinaison not in combinaisons_dict:
                                combinaisons_dict[id_combinaison] = {'ids': set(), 'reduction': reduction_float}
                            combinaisons_dict[id_combinaison]['ids'].add(id_forfait_comb)
                        except ValueError:
                            continue # Ignorer réduction invalide

                # Trouver la meilleure combinaison applicable
                best_reduc_percentage = 0
                best_reduc_desc = ""
                applicable_comb_found = False

                selected_forfait_set = set(selected_forfait_ids)

                for id_combinaison, data in combinaisons_dict.items():
                    comb_forfait_ids = data['ids']
                    # Vérifier si TOUS les forfaits sélectionnés sont dans la combinaison ET
                    # si TOUS les forfaits de la combinaison sont dans la sélection
                    # (Correspondance exacte des ensembles)
                    if comb_forfait_ids == selected_forfait_set:
                         reduction_percentage = data['reduction']
                         if reduction_percentage > best_reduc_percentage: # Prend la plus avantageuse si plusieurs match
                             best_reduc_percentage = reduction_percentage
                             reduction_amount = total_tariff_base_before_reduc * (reduction_percentage / 100)
                             best_reduc_desc = f"Réduction automatique pour combinaison de forfaits : -{reduction_amount:.2f} DH ({reduction_percentage:.2f}%)"
                             applicable_comb_found = True

                if applicable_comb_found:
                    reduction_applied = total_tariff_base_before_reduc * (best_reduc_percentage / 100)
                    total_after_auto_reduc = total_tariff_base_before_reduc - reduction_applied
                    reduction_description = best_reduc_desc

        except Exception as e:
            print(f"Erreur lors de la recherche de combinaisons: {e}")
            # Continuer sans réduction automatique en cas d'erreur

    # Génération du message HTML
    tariff_message = "<b>Détails des tarifs :</b><br>"
    for subject, info in tariffs_by_group.items():
        tariff_message += f"- {subject} ([{info['id_forfait']}] {info['nom_forfait']}, {info['nom_type_duree']}) : {info['remaining_sessions']} séances restantes x {info['tarif_unitaire']:.2f} DH/séance = {info['tarif_total']:.2f} DH<br>"

    tariff_message += f"<b>Sous-total :</b> {total_tariff_base_before_reduc:.2f} DH<br>"
    if reduction_applied > 0:
        tariff_message += f"{reduction_description}<br>"
        tariff_message += f"<b>Total après réduction auto :</b> {total_after_auto_reduc:.2f} DH"
    else:
        # Pas besoin de ligne supplémentaire si pas de réduction auto
        tariff_message += f"<b>Total (avant frais/réduction demandée) :</b> {total_after_auto_reduc:.2f} DH"


    return tariffs_by_group, tariff_message, total_tariff_base_before_reduc, total_after_auto_reduc

def get_recommendations(student_name, user_level, user_subjects_list, user_teachers_raw, user_school, user_center, selected_forfaits_dict, selected_types_duree_dict):
    # MODIFIÉ: Prend des listes/dicts directement
    output = []
    all_recommendations = {}
    all_groups_for_selection = {}
    processed_subjects = [] # Garder trace des sujets traités

    # Validation rapide des entrées (pourrait être faite avant l'appel)
    if not user_level or not user_subjects_list or not user_school:
        return ["Erreur: Niveau, Matières ou École manquants."], {}, {}, []

    # Créer un mapping matière -> prof (gère les cas où moins de profs sont donnés)
    teachers_list_clean = [t.strip() for t in user_teachers_raw.split(',') if t.strip()] if user_teachers_raw else []
    teacher_map = {subj: (teachers_list_clean[i] if i < len(teachers_list_clean) else None) for i, subj in enumerate(user_subjects_list)}

    # Récupérer TOUS les groupes une seule fois pour l'efficacité
    try:
         all_groups_data_db = collection_groupes.get(include=["metadatas", "documents"])
         if not all_groups_data_db or not all_groups_data_db.get('ids'):
              return ["Erreur: Impossible de récupérer les données des groupes depuis ChromaDB."], {}, {}, []
    except Exception as e:
         print(f"Erreur DB dans get_recommendations: {e}")
         return [f"Erreur base de données lors de la recherche: {e}"], {}, {}, []


    for subject in user_subjects_list:
        processed_subjects.append(subject) # Marquer comme traité
        matched_teacher = teacher_map.get(subject)
        id_forfait_selected = selected_forfaits_dict.get(subject)
        type_duree_id_selected = selected_types_duree_dict.get(subject)

        # Récupérer nom forfait/durée pour affichage (besoin de forfaits_info ou recherche DB)
        # Simplification: On suppose qu'on a ces infos, sinon il faudrait les passer en argument ou les rechercher
        nom_forfait_selected = "Forfait " + str(id_forfait_selected) # Placeholder
        nom_type_duree_selected = "Durée " + str(type_duree_id_selected) # Placeholder
        tarif_unitaire_selected = 0 # Placeholder - IMPORTANT: à récupérer correctement

        if not id_forfait_selected or not type_duree_id_selected:
            output.append(f"<div class='bot-message'>Info manquante : Aucun forfait ou type de durée sélectionné pour {subject}. Impossible de recommander des groupes.</div>")
            continue

        # Filtrer les groupes récupérés pour la matière/niveau/forfait/durée actuels
        relevant_groups_for_subject = []
        for id_cours, metadata, document in zip(all_groups_data_db['ids'], all_groups_data_db['metadatas'], all_groups_data_db['documents']):
             # Vérifications robustes avec .get()
             if (metadata.get('niveau', '').strip().lower() == user_level.strip().lower() and
                 metadata.get('matiere', '').strip().lower() == subject.strip().lower() and
                 metadata.get('id_forfait') == id_forfait_selected and
                 metadata.get('type_duree_id') == type_duree_id_selected):

                  # Extraire et nettoyer les données du groupe
                  try:
                      num_students = int(metadata.get('num_students', 0))
                      # Nettoyer les écoles - gérer None ou ''
                      schools_raw = metadata.get('ecole', '')
                      group_schools = [s.strip() for s in schools_raw.split(',') if s.strip()] if schools_raw else []

                      # Créer données groupe (AJOUTER tarif unitaire et nom type durée ici)
                      group_info = {
                          "id_cours": id_cours,
                          "name_cours": metadata.get('name_cours', 'Cours sans nom'),
                          "num_students": num_students,
                          "description_doc": document, # Document brut
                          "centre": metadata.get('centre'),
                          "teacher": metadata.get('teacher'),
                          "schools": group_schools,
                          # "students": [], # Students list non utilisée directement pour reco
                          "date_debut": metadata.get('date_debut'),
                          "date_fin": metadata.get('date_fin'),
                          "heure_debut": metadata.get('heure_debut'),
                          "heure_fin": metadata.get('heure_fin'),
                          "jour": metadata.get('jour'),
                          "niveau": metadata.get('niveau'),
                          "matiere": metadata.get('matiere'),
                          "id_forfait": metadata.get('id_forfait'),
                          "nom_forfait": metadata.get('nom_forfait', nom_forfait_selected), # Utiliser nom du metadata si dispo
                          "type_duree_id": metadata.get('type_duree_id'),
                          "nom_type_duree": metadata.get('nom_type_duree', nom_type_duree_selected), # Utiliser nom du metadata si dispo
                          "tarif_unitaire": metadata.get('tarifunitaire') # Très important pour calculs futurs
                      }
                      # Vérifier que les champs horaires/jours sont présents pour éviter erreurs overlap
                      if all(group_info.get(k) for k in ['heure_debut', 'heure_fin', 'jour']):
                           relevant_groups_for_subject.append(group_info)
                      else:
                           print(f"Groupe {id_cours} ignoré (infos horaire/jour manquantes): {group_info.get('jour')}, {group_info.get('heure_debut')}, {group_info.get('heure_fin')}")

                  except (ValueError, TypeError, KeyError) as e:
                      print(f"Erreur traitement métadonnées groupe {id_cours}: {e}")
                      continue # Ignorer groupe si erreur

        if not relevant_groups_for_subject:
            output.append(f"<div class='bot-message'>Aucun groupe trouvé pour {subject} avec le niveau {user_level}, le forfait [{id_forfait_selected}] et le type de durée [{type_duree_id_selected}].</div>")
            continue

        # --- Logique de Priorisation et Sélection (adaptée pour utiliser relevant_groups_for_subject) ---
        selected_groups_for_subject = []
        selected_ids = set()

        # Fonction interne pour ajouter et éviter doublons
        def add_groups(new_groups, criteria="Non spécifié"):
            count_added = 0
            # Trier par pertinence école avant d'ajouter
            new_groups.sort(key=lambda g: count_school_students(g.get('schools', []), user_school), reverse=True)
            for group in new_groups:
                if group['id_cours'] not in selected_ids and len(selected_groups_for_subject) < 3:
                    group['criteria'] = criteria # Ajouter le critère de sélection
                    selected_groups_for_subject.append(group)
                    selected_ids.add(group['id_cours'])
                    count_added += 1
            return count_added

        # Appliquer les priorités
        group_list = relevant_groups_for_subject # Utiliser la liste filtrée

        # 1. Prof + Centre + École
        if matched_teacher and user_center:
             p1 = [g for g in group_list if g['teacher'] == matched_teacher and g['centre'] == user_center and user_school in g.get('schools', [])]
             add_groups(p1, "Professeur + Centre + École")

        # 2. Prof + Centre
        if matched_teacher and user_center and len(selected_groups_for_subject) < 3:
             p2 = [g for g in group_list if g['id_cours'] not in selected_ids and g['teacher'] == matched_teacher and g['centre'] == user_center]
             add_groups(p2, "Professeur + Centre")

        # 3. Centre + École
        if user_center and len(selected_groups_for_subject) < 3:
            p3 = [g for g in group_list if g['id_cours'] not in selected_ids and g['centre'] == user_center and user_school in g.get('schools', [])]
            add_groups(p3, "Centre + École")

        # 4. Centre
        if user_center and len(selected_groups_for_subject) < 3:
            p4 = [g for g in group_list if g['id_cours'] not in selected_ids and g['centre'] == user_center]
            add_groups(p4, "Centre")

        # 5. Prof + École
        if matched_teacher and len(selected_groups_for_subject) < 3:
            p5 = [g for g in group_list if g['id_cours'] not in selected_ids and g['teacher'] == matched_teacher and user_school in g.get('schools', [])]
            add_groups(p5, "Professeur + École")

        # 6. Prof
        if matched_teacher and len(selected_groups_for_subject) < 3:
            p6 = [g for g in group_list if g['id_cours'] not in selected_ids and g['teacher'] == matched_teacher]
            add_groups(p6, "Professeur")

        # 7. École
        if len(selected_groups_for_subject) < 3:
            p7 = [g for g in group_list if g['id_cours'] not in selected_ids and user_school in g.get('schools', [])]
            add_groups(p7, "École")

        # 8. Remplissage aléatoire si moins de 3 groupes
        if len(selected_groups_for_subject) < 3:
            remaining = [g for g in group_list if g['id_cours'] not in selected_ids]
            random.shuffle(remaining)
            add_groups(remaining, "Autres groupes disponibles")

        # --- Formatage de la sortie pour cette matière ---
        if not selected_groups_for_subject:
             output.append(f"<div class='bot-message'>Désolé, aucun groupe correspondant trouvé pour {subject} après application des critères.</div>")
             continue

        recommendations_html = []
        groups_for_selection_list = []
        output.append(f"<div class='bot-message'><b>Recommandations pour {subject} :</b></div>")

        for i, group in enumerate(selected_groups_for_subject, 1):
            # Formatter les écoles pour l'affichage
            schools_display = ", ".join(sorted(list(set(group.get('schools', ['N/A'])))))
            recommendation = (
                f"<div class='bot-message' style='border-left: 3px solid #00796b; padding-left: 8px; margin-left: 10px;'>"
                f"<b>Option {i} (ID: {group['id_cours']})</b><br>"
                f"<b>Nom:</b> {group.get('name_cours', 'N/A')}<br>"
                f"<b>Professeur:</b> {group.get('teacher', 'N/A')}<br>"
                f"<b>Centre:</b> {group.get('centre', 'N/A')}<br>"
                f"<b>Horaires:</b> {group.get('jour', 'N/A')} de {group.get('heure_debut', 'N/A')} à {group.get('heure_fin', 'N/A')}<br>"
                f"<b>Forfait:</b> [{group.get('id_forfait')}] {group.get('nom_forfait', 'N/A')}<br>"
                f"<b>Type Durée:</b> {group.get('nom_type_duree', 'N/A')}<br>"
                # f"<b>Tarif/séance:</b> {group.get('tarif_unitaire', 'N/A')} DH<br>" # Optionnel ici
                f"<b>Nb Étudiants:</b> {group.get('num_students', 'N/A')}<br>"
                f"<b>Écoles Présentes:</b> {schools_display}<br>"
                f"<i>Critère de sélection principal: {group.get('criteria', 'N/A')}</i>"
                f"</div>"
            )
            output.append(recommendation) # Ajouter chaque reco formatée à la sortie globale

            # Préparer les données pour la sélection future par l'utilisateur
            groups_for_selection_list.append({
                "display_index": i, # Numéro pour la sélection utilisateur
                "id_cours": group['id_cours'],
                "name_cours": group['name_cours'],
                "display_text": f"{i}. {group['name_cours']} ({group['teacher']}, {group['centre']}, {group['jour']} {group['heure_debut']})", # Texte pour la sélection
                # Inclure TOUTES les données nécessaires pour les étapes suivantes (overlap, tarif)
                "centre": group['centre'],
                "heure_debut": group['heure_debut'],
                "heure_fin": group['heure_fin'],
                "jour": group['jour'],
                "matiere": group['matiere'],
                "id_forfait": group['id_forfait'],
                "nom_forfait": group['nom_forfait'],
                "type_duree_id": group['type_duree_id'],
                "nom_type_duree": group['nom_type_duree'],
                "tarif_unitaire": group['tarif_unitaire'] # ESSENTIEL pour calcul tarif
            })

        # Stocker les résultats pour cette matière
        # all_recommendations[subject] = recommendations_html # Plus nécessaire si on ajoute à output directement
        all_groups_for_selection[subject] = groups_for_selection_list

    # Message final si certains sujets n'ont pas pu être traités
    unprocessed_subjects = [s for s in user_subjects_list if s not in processed_subjects]
    if unprocessed_subjects:
         output.append(f"<div class='bot-message'>Note : Les matières suivantes n'ont pas pu être traitées (peut-être infos manquantes ?) : {', '.join(unprocessed_subjects)}</div>")

    return output, all_groups_for_selection # Ne retourne plus all_recommendations, c'est dans output


# --- NOUVEAU: Fonctions pour l'intégration LLM ---

def llm_call(prompt, task_type="nlg"):
    """Appelle l'API Gemini et gère la réponse."""
    try:
        # print(f"\n--- PROMPT ({task_type}) ---") # Debug
        # print(prompt) # Debug
        response = gemini_model.generate_content(prompt)
        # print(f"--- RESPONSE ({task_type}) ---") # Debug
        # print(response.text) # Debug

        if task_type == "nlu":
            # Essayer de nettoyer et parser la réponse NLU comme JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as json_err:
                print(f"Erreur parsing JSON NLU: {json_err}")
                print(f"Réponse brute NLU: {response.text}")
                # Retourner un format d'erreur standardisé
                return {"intent": "error", "entities": {}, "error_message": "Impossible de parser la réponse NLU."}
        else: # NLG
             # Enlever les marqueurs de code block si présents dans la réponse NLG
             return response.text.replace("```html", "").replace("```", "").strip()

    except Exception as e:
        print(f"Erreur API Gemini ({task_type}): {e}")
        if task_type == "nlu":
             return {"intent": "error", "entities": {}, "error_message": f"Erreur API Gemini: {e}"}
        else:
             return "<div class='bot-message'>Désolé, une erreur technique est survenue avec l'assistant IA. Veuillez réessayer.</div>" # Message NLG de fallback

# --- Fonction create_nlu_prompt (CORRIGÉE) ---
# --- Fonction create_nlu_prompt (CORRIGÉE v4 - Null Handling + Interprétation) ---
def create_nlu_prompt(user_input, session_state):
    """Crée le prompt pour l'analyse NLU par Gemini."""
    responses = session_state.get('responses', {})
    subjects_in_context = session_state.get('matched_subjects', [])
    num_subjects = len(subjects_in_context)
    needed = session_state.get('needed_info', set())
    history = session_state.get('messages', [])
    last_bot_message_html = history[-1][0] if history and history[-1][1] else "Bonjour !"
    last_bot_message = last_bot_message_html.split('>')[1].split('<')[0] if '>' in last_bot_message_html else last_bot_message_html

    selected_forfaits_str = str(responses.get('selected_forfaits', {}))
    selected_types_duree_str = str(responses.get('selected_types_duree', {}))
    selected_groups_summary_str = str(responses.get('selected_groups_summary', 'Non sélectionnés'))
    needed_list_str = str(list(needed))

    # Identifier les champs facultatifs pour la gestion du "rien"
    optional_fields_context = []
    if 'user_grades' in needed: optional_fields_context.append('user_grades')
    if 'user_teachers' in needed: optional_fields_context.append('user_teachers')
    if 'user_center' in needed: optional_fields_context.append('user_center')
    if 'commentaires' in needed: optional_fields_context.append('commentaires') # Considéré facultatif

    prompt = f"""
    Rôle: Tu es un assistant analysant l'entrée d'un utilisateur pour un chatbot de recommandation de groupes scolaires en français. Tu dois INTERPRÉTER le langage naturel et le traduire en données structurées. Sois concis dans tes analyses.
    Contexte Conversationnel Actuel:
    - Étudiant: {responses.get('student_name', 'N/A')}
    - Niveau: {responses.get('user_level', 'N/A')}
    - Matières en traitement: {subjects_in_context} (Nombre: {num_subjects})
    - École: {responses.get('user_school', 'N/A')}
    - Centre: {responses.get('user_center', 'N/A')}
    - Dernière question/message du bot: "{last_bot_message}"
    - Informations recherchées: {needed_list_str}
    - Champs actuellement considérés comme facultatifs: {optional_fields_context}

    Listes de Validation (utiliser fuzzy matching):
    - Niveaux: {levels_list}
    - Matières: {subjects_list}
    - Écoles: {schools_list}
    - Centres: {centers_list}

    Entrée Utilisateur: "{user_input}"

    Instructions:
    1. Analyse l'intention principale ('provide_info', 'ask_question', 'request_discount', 'confirm', 'deny', 'go_back', 'select_choice', 'provide_comment', 'other').
    2. Si 'provide_info', extrais les entités pertinentes. Valide contre les listes fournies.
        - Entités possibles: student_name, user_level, user_subjects, user_grades, course_choices, user_teachers, user_school, user_center, inscription_fee_choice, reduction_percentage, another_case_choice.

    3. *** GESTION DES RÉPONSES VIDES/NULLES ***:
        - Si l'utilisateur répond "rien", "aucune", "pas de X", "sans", "aucun", "non merci" ou similaire en référence à un champ IDENTIFIÉ COMME FACULTATIF dans le contexte (voir `Champs actuellement considérés comme facultatifs`), extrais l'entité correspondante avec une chaîne vide `""`.
        - Ne fais PAS cela pour les champs obligatoires (ex: niveau, matières, école). Pour eux, une telle réponse est une erreur ou une intention 'deny'.
        - Exemples (si la question portait sur les profs, qui est facultatif):
            - Input: "rien" -> {{"intent": "provide_info", "entities": {{"user_teachers": ""}}, ...}}
            - Input: "aucun" -> {{"intent": "provide_info", "entities": {{"user_teachers": ""}}, ...}}
        - Exemple (si la question portait sur le centre, facultatif):
            - Input: "pas de préférence" -> {{"intent": "provide_info", "entities": {{"user_center": ""}}, ...}}
        - Exemple (si la question portait sur les commentaires, facultatif):
            - Input: "non" -> {{"intent": "provide_info", "entities": {{"comment": ""}}, ...}} # Interpréter 'non' comme 'rien à dire'

    4. *** CAS SPÉCIAL 'course_choices' ***:
        - Interprète la réponse (liste OU phrase) en utilisant le nombre de matières ({num_subjects}).
        - L'entité 'course_choices' DOIT être une liste de strings ('indiv'/'groupe') de longueur {num_subjects}.
        - Ex (si {num_subjects}=2): "indiv, groupe" -> {{"course_choices": ["indiv", "groupe"]}}; "les deux en individuel" -> {{"course_choices": ["indiv", "indiv"]}}

    5. Si 'select_choice', extrais `numeric_selections` (list[int]) et `selection_context`.
    6. Si 'go_back', identifie `target`.
    7. Si 'provide_comment', extrais `comment`. Si l'utilisateur dit "rien" ou similaire, extrais `comment: ""`.
    8. Retourne UNIQUEMENT un objet JSON valide: {{"intent": ..., "entities": {{...}}, "target": ..., "error_message": ...}}

    Exemples JSON (structure seulement):
    Input: "rien" (pour centre facultatif) -> {{{{ "intent": "provide_info", "entities": {{{{ "user_center": "" }}}}, "target": null, "error_message": null }}}}
    Input: "je veux faire des cours individuels pour les deux matières" (si {num_subjects}=2) -> {{{{ "intent": "provide_info", "entities": {{{{ "course_choices": ["indiv", "indiv"] }}}}, "target": null, "error_message": null }}}}

    Réponds seulement avec l'objet JSON.
    """
    return prompt

# --- Fonction create_nlg_prompt (CORRIGÉE v3 - Questions Ouvertes + Clarification avec Listes) ---
# --- Fonction create_nlg_prompt (CORRIGÉE v5 - Concision et Variation) ---
def create_nlg_prompt(next_action, action_results, session_state):
    """Crée le prompt pour la génération de réponse (NLG) par Gemini."""
    responses = session_state.get('responses', {})
    student_name = responses.get('student_name', 'l\'utilisateur') # Utiliser un terme générique par défaut
    action_code = next_action.get("action")
    flags = session_state.get('flags', {})
    history = session_state.get('messages', [])
    # Essayer d'obtenir le tour précédent pour éviter répétition exacte
    previous_bot_message = history[-2][0] if len(history) > 1 and history[-2][1] else None

    # Base du prompt - Instructions renforcées
    prompt = f"""
    Rôle: Tu es 'MentorBot', un assistant chatbot. Tu guides l'utilisateur pour trouver des groupes scolaires.
    Ton: Amical, serviable, mais CONCIS et ALLANT DROIT AU BUT. ÉVITE la redondance.
    Instructions Générales:
    1.  **Sois Bref:** Formule des phrases courtes et claires. Évite les mots inutiles.
    2.  **Varie tes Réponses:** N'utilise pas les mêmes phrases d'introduction ou de transition à chaque fois. Sois créatif mais reste simple.
    3.  **Évite le Nom:** N'utilise le nom de l'étudiant ({student_name}) que très rarement (ex: accueil initial), sinon adresse-toi directement à l'utilisateur (tu/vous selon le ton général).
    4.  **Format Simple:** Utilise <b>, <br>, <ul>, <li> si besoin pour la clarté. Pas de markdown. Couleur neutre.
    5.  **Évite Répétition Exacte:** Si possible, ne répète pas exactement le message précédent ({previous_bot_message}).

    Contexte Actuel (pour info seulement):
    - Niveau: {responses.get('user_level', 'N/S')}
    - Matières: {session_state.get('matched_subjects', [])}
    - Réduc Demandée: {'Oui' if flags.get('discount_requested') else 'Non'}

    Tâche: Génère la prochaine réponse du chatbot basée sur l'Action Système.
    Action Système: '{action_code}'
    Résultats/Données Fournis (si applicable): {action_results if action_results else 'N/A'}
    """

    # Instructions spécifiques par action - En mettant l'accent sur la concision
    if action_code == "ask_question":
        info_key = next_action.get("info_key")
        question = ""
        placeholder = ""
        # Définir les questions de manière plus directe
        if info_key == "student_name":
            question = "Quel est le prénom de l'étudiant(e) ? 🧑‍🎓"
            placeholder = "Prénom"
        elif info_key == "user_level":
            question = "Ok, et sa classe cette année ?"
            placeholder = "Ex: 2bac sc ex PC"
        elif info_key == "user_subjects":
            question = "Quelles matières sont concernées ? 📚"
            placeholder = "Ex: Maths, Physique"
        elif info_key == "user_grades":
             subjects_str = ', '.join(session_state.get('matched_subjects', ['les matières']))
             question = f"Notes récentes pour {subjects_str} ? (Facultatif)"
             placeholder = "Ex: 12, 15 (ou laisser vide / 'aucune')" # Indiquer que 'aucune' est ok
        elif info_key == "course_choices":
             subjects_str = ', '.join(session_state.get('matched_subjects', ['les matières']))
             # Message reco simplifié
             reco_msg = "Pour chaque matière, préfères-tu 'indiv' ou 'groupe' ?<br>"
             # Ajouter conseil rapide si notes < 10 ? (Optionnel)
             # ...
             question = f"{reco_msg}(Réponse type: 'indiv, groupe' ou 'les deux en groupe')"
             placeholder = "indiv / groupe"
        elif info_key == "selected_forfaits":
             subject = next_action.get("subject")
             forfaits = next_action.get("results",{}).get("forfaits") or session_state.get('available_forfaits', {}).get(subject, {})
             if forfaits:
                 options_html = "".join([f"<li>{i}. [{fid}] {finfo['name']}</li>" for i, (fid, finfo) in enumerate(forfaits.items(), 1)])
                 question = f"Forfaits pour <b>{subject}</b>:<ul>{options_html}</ul>Choisis le numéro :"
                 placeholder = f"Numéro (1-{len(forfaits)})"
             else: question = f"Aucun forfait trouvé pour <b>{subject}</b> au niveau {responses.get('user_level')}."
        elif info_key == "selected_types_duree":
             subject = next_action.get("subject")
             types_duree = next_action.get("results", {}).get("types_duree") # Utiliser les résultats passés
             # ... (logique pour récupérer types_duree si pas dans results) ...
             id_forfait = responses.get('selected_forfaits', {}).get(subject)
             nom_forfait = session_state.get('available_forfaits', {}).get(subject, {}).get(id_forfait, {}).get('name', f'Forfait {id_forfait}')
             if types_duree:
                  options_html = "".join([f"<li>{i}. {tdinfo['name']} ({tdinfo['tarif_unitaire']:.0f} DH/séance)</li>" for i, (tid, tdinfo) in enumerate(types_duree.items(), 1)])
                  question = f"Type de durée pour <b>{subject}</b> ({nom_forfait}) ?<ul>{options_html}</ul>Indique le numéro :"
                  placeholder = f"Numéro (1-{len(types_duree)})"
             else: question = f"Aucun type de durée pour le forfait {nom_forfait} de <b>{subject}</b>."
        elif info_key == "user_teachers":
             question = f"Professeurs à éviter/retrouver ? (Facultatif)"
             placeholder = "Noms (ou 'aucun')"
        elif info_key == "user_school":
             question = "Quelle est l'école actuelle ? 🏫"
             placeholder = "Nom de l'école"
        elif info_key == "user_center":
             question = "Préférence de centre ? (Facultatif)"
             placeholder = "Nom du centre (ou 'aucun')"
        elif info_key == "inscription_fee_choice":
             frais = 250
             question = f"Ajouter les frais d'inscription ({frais} DH) ? (Oui/Non)"
             placeholder = "Oui / Non"
        elif info_key == "commentaires":
             question = "Commentaires ou demandes particulières ?"
             placeholder = "Écrire ici (ou 'rien')"
        elif info_key == "another_case_choice":
             question = "Faire une autre recherche ? (Oui/Non)"
             placeholder = "Oui / Non"
        else: question = f"Information manquante : {info_key} ?" # Fallback direct
        # --- Instruction finale pour ask_question ---
        prompt += f"\nInstructions: Pose la question suivante de manière directe et concise. Utilise une brève transition variée si ce n'est pas la première question."
        prompt += f"\nQuestion:\n{question}"
        prompt += f"\nPlaceholder Interface: {placeholder}"

    elif action_code == "ask_clarification":
        errors = next_action.get("errors", ["Information invalide."])
        original_input = next_action.get("original_input", "")
        error_field = None # Détection du champ (inchangée)
        # ... (logique de détection error_field inchangée) ...
        if any("Niveau" in err for err in errors): error_field = "level"
        elif any("Matière" in err for err in errors): error_field = "subject"
        # ... etc ...

        error_list_html = "<ul>" + "".join([f"<li>{err}</li>" for err in errors]) + "</ul>"
        prompt += f"\nInstructions: Informe l'utilisateur d'une erreur avec sa réponse ('{original_input}'). Sois bref. Liste les erreurs."
        prompt += f"\nErreurs:\n{error_list_html}"
        # Ajouter la liste d'aide si pertinente
        valid_options_message = ""
        # ... (logique pour générer valid_options_message inchangée) ...
        if error_field == "level": valid_options_message = f"<br>Niveaux possibles : <i>{', '.join(levels_list)}</i>"
        # ... etc ...
        if valid_options_message:
             prompt += f"\nAjoute l'aide suivante:\n{valid_options_message}"
        prompt += f"\nTermine en demandant de réessayer."

    elif action_code == "student_status":
         is_new = action_results.get("is_new", True)
         accueil_msg = f"Enchanté {student_name} ! 😊" if is_new else f"Re-bonjour {student_name} ! 👋"
         # Prochaine question (directe)
         next_q_key = next(iter(session_state.get('needed_info', {'user_level'})), 'user_level')
         next_q_prompt = "Ta classe cette année ?" if next_q_key == 'user_level' else f"{next_q_key} ?"
         next_q_placeholder = "Ex: 1bac sc ex" if next_q_key == 'user_level' else ""
         prompt += f"\nInstructions: Affiche l'accueil '{accueil_msg}' puis pose directement la question '{next_q_prompt}'."
         prompt += f"\nPlaceholder Interface: '{next_q_placeholder}'."

    # --- Réviser les autres actions pour la concision ---
    elif action_code == "show_recommendations":
        output_messages = action_results.get("output", [])
        groups_for_selection = action_results.get("groups_for_selection", {})
        subjects_needing_selection = list(groups_for_selection.keys())
        prompt += f"\nInstructions: Annonce brièvement ('Voici les options trouvées : ✨'). Affiche les blocs HTML fournis."
        prompt += "\nBlocs HTML:\n" + "\n".join(output_messages)
        if subjects_needing_selection:
            prompt += f"\n\nTermine en demandant le numéro pour chaque matière ({', '.join(subjects_needing_selection)})."
            prompt += f"\nPlaceholder: Ex: {', '.join(['1'] * len(subjects_needing_selection))}"
        else:
            prompt += f"\n\nTermine en passant au calcul des tarifs." # Transition simple

    elif action_code == "show_tariffs":
        tariff_html = action_results.get("tariff_message_html", "")
        prompt += f"\nInstructions: Introduis ('Ok, voici les tarifs :'). Affiche le bloc HTML."
        prompt += f"\nBloc HTML:\n{tariff_html}"
        if next_action.get("sub_action") == "ask_inscription_fee":
             frais = 250
             prompt += f"\nAjoute la question : 'Ajouter frais inscription ({frais} DH) ? (Oui/Non)'."
             prompt += f"\nPlaceholder: Oui / Non"
        else:
             prompt += f"\nTermine avec une transition vers les commentaires (ex: 'Des demandes particulières ?')."

    # ... (Adapter les autres prompts : ask_group_selection, ask_discount_percentage, etc. pour être plus directs)

    elif action_code == "finalize":
         prompt += f"\nInstructions: Remercie brièvement et dis au revoir."
         prompt += f"\nExemple: Merci d'avoir utilisé MentorBot ! Bonne journée. 👋"

    # Fallback (inchangé)
    else:
        prompt += f"\nInstructions: Action système '{action_code}' inattendue. Message court d'attente."
        prompt += f"\nExemple: Un instant..."

    prompt += "\n\nRéponse du Chatbot (format HTML simple uniquement) :"
    return prompt

# NOUVEAU: Fonction de validation des entités extraites par le LLM
def validate_entities(entities, session_state):
    validated_data = {}
    errors = []
    subjects_in_context = session_state.get('matched_subjects', [])
    context = entities.get("selection_context") # Pour les choix numériques

    for key, value in entities.items():
        original_value = value # Garder trace de la valeur brute
        is_valid = False
        matched_value = value # Valeur qui sera stockée si valide

        if value is None or value == '': # Ignorer les valeurs vides retournées par LLM
             continue

        try: # Bloc try général pour attraper des erreurs inattendues
             if key == 'user_level':
                 matched_value, is_valid = match_value(value, levels_list)
                 if not is_valid: errors.append(f"Niveau '{original_value}' non reconnu. Valides: {', '.join(levels_list)}")
             elif key == 'user_subjects':
                 if isinstance(value, list):
                     valid_subjects = []
                     all_subj_valid = True
                     for subj in value:
                         matched_subj, subj_is_valid = match_value(subj, subjects_list)
                         if subj_is_valid:
                             valid_subjects.append(matched_subj)
                         else:
                             all_subj_valid = False
                             errors.append(f"Matière '{subj}' non reconnue. Valides: {', '.join(subjects_list)}")
                     if all_subj_valid:
                          matched_value = valid_subjects
                          is_valid = True
                 else:
                     errors.append("Le format des matières est invalide (attendu: liste).")
             elif key == 'user_school':
                 matched_value, is_valid = match_value(value, schools_list)
                 if not is_valid: errors.append(f"École '{original_value}' non reconnue. Valides: {', '.join(schools_list)}")
             elif key == 'user_center':
                 if not value: # Permettre vide pour centre facultatif
                     matched_value = None
                     is_valid = True
                 else:
                     matched_value, is_valid = match_value(value, centers_list)
                     if not is_valid: errors.append(f"Centre '{original_value}' non reconnu. Valides: {', '.join(centers_list)} ou laisser vide.")
             elif key == 'inscription_fee_choice':
                 choice = str(value).lower().strip()
                 if choice in ['oui', 'yes', 'o', 'y']:
                     matched_value = True; is_valid = True
                 elif choice in ['non', 'no', 'n']:
                     matched_value = False; is_valid = True
                 else: errors.append("Réponse pour frais d'inscription invalide ('Oui' ou 'Non' attendu).")
             elif key == 'reduction_percentage':
                  try:
                      perc = float(value)
                      if 0 <= perc <= 100: matched_value = perc; is_valid = True
                      else: errors.append("Le pourcentage de réduction doit être entre 0 et 100.")
                  except (ValueError, TypeError): errors.append("Le pourcentage de réduction doit être un nombre.")
             elif key == 'another_case_choice':
                 choice = str(value).lower().strip()
                 if choice in ['oui', 'yes', 'o', 'y']: matched_value = True; is_valid = True
                 elif choice in ['non', 'no', 'n']: matched_value = False; is_valid = True
                 else: errors.append("Réponse pour 'autre cas' invalide ('Oui' ou 'Non' attendu).")
             elif key == 'numeric_selections':
                 if isinstance(value, list):
                     valid_selections = []
                     all_num_valid = True
                     # Vérifier la cohérence avec le contexte (forfait, durée, groupe)
                     num_expected = len(subjects_in_context)
                     if context == 'forfait':
                         data_source = session_state.get('available_forfaits', {})
                     elif context == 'type_duree':
                         data_source = session_state.get('available_types_duree', {})
                     elif context == 'groupe':
                         data_source = session_state.get('all_groups_for_selection', {})
                     else:
                         errors.append("Contexte de sélection numérique inconnu."); all_num_valid = False; data_source={}

                     if len(value) != num_expected:
                         errors.append(f"Nombre de sélections incorrect (attendu: {num_expected}, reçu: {len(value)}).")
                         all_num_valid = False
                     else:
                         for i, sel in enumerate(value):
                              subject = subjects_in_context[i]
                              options_for_subject = data_source.get(subject, {})
                              num_options = len(options_for_subject) if isinstance(options_for_subject, (dict, list)) else 0
                              if isinstance(options_for_subject, list): # Cas des groupes
                                   num_options = len(options_for_subject)

                              try:
                                   sel_int = int(sel)
                                   if 1 <= sel_int <= num_options:
                                       valid_selections.append(sel_int)
                                   else:
                                       errors.append(f"Sélection '{sel}' pour {subject} hors limites (1-{num_options}).")
                                       all_num_valid = False
                              except (ValueError, TypeError):
                                   errors.append(f"Sélection '{sel}' pour {subject} n'est pas un nombre valide.")
                                   all_num_valid = False
                     if all_num_valid:
                          matched_value = valid_selections
                          is_valid = True
                          validated_data['selection_context'] = context # Stocker aussi le contexte validé
                 else: errors.append("Format de sélection numérique invalide (attendu: liste de nombres).")

             # Cas simples (string non vide) - peuvent être affinés
             elif key == 'student_name': matched_value = str(value).strip(); is_valid = bool(matched_value)
             elif key == 'user_grades': matched_value = str(value).strip(); is_valid = True # Validation simple pour l'instant
             elif key == 'course_choices': # Valider 'indiv'/'groupe' et longueur
                 if isinstance(value, list):
                     if len(value) == len(subjects_in_context):
                         if all(c.lower().strip() in ['indiv', 'groupe'] for c in value):
                             matched_value = [c.lower().strip() for c in value]
                             is_valid = True
                         else: errors.append("Choix de cours invalides (doivent être 'indiv' ou 'groupe').")
                     else: errors.append(f"Nombre de choix de cours incorrect (attendu {len(subjects_in_context)}).")
                 else: errors.append("Format de choix de cours invalide (attendu: liste).")
             elif key == 'user_teachers': matched_value = str(value).strip(); is_valid = True # Pas de validation stricte
             elif key == 'comment': matched_value = str(value).strip(); is_valid = True

             # Ajouter la valeur validée si tout est OK
             if is_valid:
                 validated_data[key] = matched_value
             # else: # L'erreur est déjà ajoutée si non valide (sauf cas ignorés comme valeur vide)
                 # pass

        except Exception as e:
            print(f"Erreur de validation inattendue pour {key}={value}: {e}")
            errors.append(f"Erreur interne lors de la validation de '{key}'.")

    return {"valid": not errors, "validated_data": validated_data, "errors": errors}

# NOUVEAU: Fonction pour déterminer la prochaine action
def determine_next_action(session_state):
    responses = session_state.get('responses', {})
    flags = session_state.get('flags', {})
    needed = session_state.get('needed_info', set())

    # 0. Vérifier si l'étudiant existe (juste après avoir eu le nom)
    if 'student_name' in responses and 'student_status_checked' not in flags:
         return {"action": "check_student_status"} # Action spécifique

    # 1. Priorité aux clarifications / erreurs
    if flags.get('needs_clarification'):
        return {"action": "ask_clarification", "errors": flags.get('validation_errors', []), "original_input": flags.get('last_user_input', '')}
    if flags.get('overlap_conflict'):
         # L'action handle_overlap a été déclenchée, maintenant on redemande la sélection
         return {"action": "ask_group_selection", "reason": "overlap"} # Raison pour NLG

    # 2. Collecter les infos de base manquantes dans l'ordre logique
    if 'student_name' not in responses: return {"action": "ask_question", "info_key": "student_name"}
    if 'user_level' not in responses: return {"action": "ask_question", "info_key": "user_level"}
    if 'user_subjects' not in responses: return {"action": "ask_question", "info_key": "user_subjects"}
    # Optionnel : notes
    if 'user_grades' not in responses and 'user_subjects' in responses:
        return {"action": "ask_question", "info_key": "user_grades"}
    # Choix type cours
    if 'course_choices' not in responses and 'user_subjects' in responses:
        # S'assurer que les notes (même vides) sont là avant de demander type de cours
        if 'user_grades' in responses:
             return {"action": "ask_question", "info_key": "course_choices"}

    # 3. Collecter détails pour matières 'groupe'
    group_subjects = []
    if 'user_subjects' in responses and 'course_choices' in responses:
         subjects = responses.get('user_subjects', [])
         choices = responses.get('course_choices', [])
         group_subjects = [subjects[i] for i, choice in enumerate(choices) if choice == 'groupe' and i < len(subjects)]
         # Mettre à jour la liste des sujets actifs si elle a changé
         if group_subjects != session_state.get('matched_subjects'):
             session_state['matched_subjects'] = group_subjects
             # Réinitialiser les choix dépendants si les matières changent
             responses.pop('selected_forfaits', None)
             responses.pop('selected_types_duree', None)
             responses.pop('group_selections', None)
             # ... réinitialiser aussi available_forfaits etc. ...
             session_state.pop('available_forfaits', None)
             session_state.pop('available_types_duree', None)
             session_state.pop('all_groups_for_selection', None)
             flags.pop('recommendations_shown', None)
             flags.pop('tariffs_calculated', None)


    if group_subjects: # Si au moins une matière en groupe
        current_forfaits = responses.get('selected_forfaits', {})
        current_durees = responses.get('selected_types_duree', {})
        for subj in group_subjects:
             if subj not in current_forfaits:
                  # Vérifier si les forfaits sont déjà chargés pour ce sujet
                  if subj not in session_state.get('available_forfaits', {}):
                       return {"action": "get_forfaits", "subject": subj} # Déclenche le calcul puis la question
                  else:
                       return {"action": "ask_question", "info_key": "selected_forfaits", "subject": subj}
             if subj not in current_durees:
                  # Vérifier si les durées sont chargées (dépend des forfaits chargés)
                  id_forfait = current_forfaits.get(subj)
                  if subj not in session_state.get('available_types_duree', {}) or \
                     id_forfait not in session_state.get('available_forfaits',{}).get(subj,{}):
                      # Devrait avoir été chargé par get_forfaits, sinon erreur
                      # On suppose qu'ils sont chargés et on pose la question
                      # Récupérer les types durée à partir de available_forfaits
                      types_duree_options = session_state.get('available_forfaits', {}).get(subj, {}).get(id_forfait, {}).get('types_duree', {})
                      return {"action": "ask_question", "info_key": "selected_types_duree", "subject": subj, "results": {"types_duree": types_duree_options}}
                  else:
                      types_duree_options = session_state.get('available_types_duree', {}).get(subj, {})
                      return {"action": "ask_question", "info_key": "selected_types_duree", "subject": subj, "results": {"types_duree": types_duree_options}}


        # 4. Collecter préférences (prof/centre) si matières groupe existent
        if 'user_teachers' not in responses: return {"action": "ask_question", "info_key": "user_teachers"}
        if 'user_school' not in responses: return {"action": "ask_question", "info_key": "user_school"} # École est obligatoire avant centre/reco
        if 'user_center' not in responses: return {"action": "ask_question", "info_key": "user_center"}

        # 5. Lancer les recommandations si tout est prêt pour les matières groupe
        # Vérifier que TOUTES les infos nécessaires pour TOUTES les matières groupe sont là
        all_group_details_collected = all(s in responses.get('selected_forfaits', {}) and s in responses.get('selected_types_duree', {}) for s in group_subjects)
        if all_group_details_collected and \
           'user_school' in responses and \
           'user_center' in responses and \
           'user_teachers' in responses and \
           'recommendations_shown' not in flags:
             return {"action": "get_recommendations"}

        # 6. Demander sélection de groupe si recos montrées mais pas de sélection
        if flags.get('recommendations_shown') and 'group_selections' not in responses:
            # S'assurer qu'il y avait des groupes à sélectionner
            if session_state.get('all_groups_for_selection'):
                 return {"action": "ask_group_selection", "results": {"groups_for_selection": session_state.get('all_groups_for_selection')}}
            else: # Pas de groupe à sélectionner, passer à la suite
                 flags['no_groups_to_select'] = True # Marquer ce cas
                 pass # Continue vers la section tarif/finalisation

        # 7. Calculer tarifs si groupes sélectionnés (ou si pas de groupes à sélectionner)
        if ('group_selections' in responses or flags.get('no_groups_to_select')) and 'tariffs_calculated' not in flags:
             return {"action": "calculate_tariffs"}

        # 8. Demander frais si tarifs calculés et groupes sélectionnés
        if flags.get('tariffs_calculated') and not flags.get('no_groups_to_select') and 'inscription_fee_choice' not in responses:
             return {"action": "ask_question", "info_key": "inscription_fee_choice"}

    # Fin des étapes spécifiques aux groupes. Si on arrive ici:
    # - Soit il n'y avait que des cours indiv
    # - Soit toutes les étapes groupe (jusqu'aux frais) sont finies

    # 9. Demander commentaires
    # Conditions: Tarifs calculés OU pas de groupes à sélectionner OU pas de matières groupe du tout
    # ET Commentaires non encore demandés
    ready_for_comments = flags.get('tariffs_calculated') or flags.get('no_groups_to_select') or not group_subjects
    if ready_for_comments and 'commentaires' not in responses:
         # S'assurer que frais sont traités si nécessaire
         if flags.get('no_groups_to_select') or 'inscription_fee_choice' in responses or not group_subjects:
             return {"action": "ask_question", "info_key": "commentaires"}

    # 10. Demander % réduction si demandée et pas fournie, et après commentaires
    if flags.get('discount_requested') and 'reduction_percentage' not in responses and 'commentaires' in responses:
         # Passer le total actuel (après réduc auto + frais) pour affichage
         current_total = session_state.get('current_total_for_discount', 0)
         return {"action": "ask_discount_percentage", "results": {"current_total": current_total}}

    # 11. Finaliser (montrer récap final, puis demander autre cas)
    ready_to_finalize = 'commentaires' in responses and (not flags.get('discount_requested') or 'reduction_percentage' in responses)
    if ready_to_finalize:
        if 'final_summary_shown' not in flags:
            return {"action": "show_final_summary"}
        else:
            # Vérifier si la réponse 'autre cas' a déjà été donnée
            if 'another_case_choice' not in responses:
                return {"action": "ask_question", "info_key": "another_case_choice"}
            else:
                # La conversation est terminée (l'utilisateur a répondu oui ou non)
                # 'process_user_turn' gérera le reset ou la fin.
                return {"action": "end_conversation"} # Action pour ne rien faire de plus

    # Fallback: Si on arrive ici, état potentiellement incohérent
    print(f"WARN: determine_next_action fallback. State: R={responses}, F={flags}, N={needed}")
    # Tenter de redemander une info manquante si applicable
    if needed:
         next_needed = next(iter(needed))
         return {"action": "ask_question", "info_key": next_needed}
    # Sinon, erreur générique
    return {"action": "error", "message": "Je suis un peu perdu, pourriez-vous repréciser votre demande ?"}

# NOUVEAU: Fonction principale de traitement d'un tour utilisateur
# --- Fonction process_user_turn (CORRIGÉE pour UnboundLocalError) ---
def process_user_turn(user_input, session_state):
    session_state.flags['needs_clarification'] = False # Réinitialiser flag
    session_state.flags['validation_errors'] = [] # Réinitialiser erreurs
    session_state.flags['last_user_input'] = user_input # Stocker pour contexte clarification

    bot_action_results = {} # Pour stocker les résultats des appels Python
    next_action = {} # <<< NOUVEAU: Initialiser next_action ici

    # 1. NLU
    nlu_prompt = create_nlu_prompt(user_input, session_state)
    llm_nlu_response = llm_call(nlu_prompt, task_type='nlu')

    intent = llm_nlu_response.get("intent", "error")
    entities = llm_nlu_response.get("entities", {})
    target = llm_nlu_response.get("target")
    nlu_error = llm_nlu_response.get("error_message")

    # 2. Gestion initiale des intentions et erreurs NLU
    if intent == "error" or nlu_error:
        session_state.flags['needs_clarification'] = True
        session_state.flags['validation_errors'] = [nlu_error or "Impossible de comprendre votre demande."]
    elif intent == "other":
         session_state.flags['needs_clarification'] = True
         session_state.flags['validation_errors'] = ["Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ou préciser ?"]
    elif intent == "ask_question":
         # Répondre qu'on ne gère pas les questions générales pour l'instant
         session_state.flags['needs_clarification'] = True
         session_state.flags['validation_errors'] = ["Concentrons-nous sur la recherche de groupes pour le moment ! 😉 Si vous avez une question spécifique sur le processus, j'essaierai d'y répondre plus tard."]

    # 3. Traitement spécifique avant validation (si pas déjà besoin de clarification)
    if not session_state.flags.get('needs_clarification'):
        if intent == "request_discount":
            session_state.flags['discount_requested'] = True
            session_state.messages.append((f"<div class='user-message'>{user_input}</div>", False))
            # Message confirmation viendra de NLG basé sur determine_next_action
        elif intent == "go_back":
            session_state.messages.append((f"<div class='user-message'>{user_input}</div>", False))
            if target and target in session_state.responses:
                session_state.responses.pop(target, None)
                session_state.needed_info.add(target)
                # TODO: Ajouter logique de réinitialisation des étapes dépendantes si nécessaire
                session_state.messages.append((f"<div class='bot-message'>Ok, revenons sur '{target}'.</div>", True))
            else:
                session_state.flags['needs_clarification'] = True
                session_state.flags['validation_errors'] = [f"Je ne peux pas revenir sur '{target}' ou la cible est inconnue."]
        # Ajouter ici d'éventuels traitements pour 'confirm', 'deny' s'ils ne nécessitent pas de validation
        elif intent == "confirm":
             # Logique si une confirmation était attendue
             pass
        elif intent == "deny":
             # Logique si un refus était attendu (ex: frais inscription)
             if 'inscription_fee_choice' in session_state.needed_info: # Si on attendait réponse sur frais
                 session_state.responses['inscription_fee_choice'] = False # Marquer comme refusé
                 session_state.needed_info.discard('inscription_fee_choice')
                 session_state.messages.append((f"<div class='user-message'>{user_input}</div>", False))


    # 4. Validation & Mise à jour de l'état (si pas déjà besoin de clarification et si intent pertinent)
    if not session_state.flags.get('needs_clarification') and intent in ["provide_info", "select_choice", "provide_comment"]:
         # Afficher réponse user seulement si elle n'a pas déjà été affichée (ex: pour request_discount)
         if intent != "request_discount": # Devrait déjà être géré mais sécurité
              session_state.messages.append((f"<div class='user-message'>{user_input}</div>", False))

         validation_result = validate_entities(entities, session_state)

         if validation_result["valid"]:
             validated_data = validation_result["validated_data"]
             # Mettre à jour session_state.responses et gérer les cas spécifiques
             for key, value in validated_data.items():
                  session_state.responses[key] = value
                  session_state.needed_info.discard(key) # Marquer comme obtenu

                  # Logique post-validation pour sélections numériques
                  if key == 'numeric_selections':
                      context = validated_data.get("selection_context")
                      selections = validated_data.get("numeric_selections", [])
                      subjects_context = session_state.get('matched_subjects', [])
                      # --- Mapping Forfait ---
                      if context == 'forfait':
                          selected_forfaits = session_state.responses.get('selected_forfaits', {})
                          data_source = session_state.get('available_forfaits', {})
                          all_mapped = True
                          for i, sel_idx in enumerate(selections):
                              if i < len(subjects_context):
                                   subj = subjects_context[i]
                                   options = list(data_source.get(subj, {}).keys())
                                   if 0 <= sel_idx - 1 < len(options):
                                       selected_forfaits[subj] = options[sel_idx - 1]
                                       # Charger les types de durée associés pour ce forfait/sujet
                                       if subj not in session_state.get('available_types_duree', {}):
                                            session_state.setdefault('available_types_duree', {})[subj] = \
                                                data_source.get(subj, {}).get(options[sel_idx - 1], {}).get('types_duree', {})
                                   else: all_mapped = False; break # Index invalide
                              else: all_mapped = False; break # Plus de sélections que de sujets
                          if all_mapped: session_state.responses['selected_forfaits'] = selected_forfaits
                          else: # Erreur de mapping (ne devrait pas arriver si validation OK)
                              session_state.flags['needs_clarification'] = True
                              session_state.flags['validation_errors'] = ["Erreur interne lors du mapping des forfaits."]
                      # --- Mapping Type Durée ---
                      elif context == 'type_duree':
                           selected_durees = session_state.responses.get('selected_types_duree', {})
                           all_mapped = True
                           for i, sel_idx in enumerate(selections):
                               if i < len(subjects_context):
                                   subj = subjects_context[i]
                                   id_forfait_ctx = session_state.responses.get('selected_forfaits', {}).get(subj)
                                   if id_forfait_ctx:
                                        duree_options_dict = session_state.get('available_forfaits', {}).get(subj, {}).get(id_forfait_ctx, {}).get('types_duree', {})
                                        duree_options_keys_ordered = list(duree_options_dict.keys())
                                        if 0 <= sel_idx - 1 < len(duree_options_keys_ordered):
                                             selected_durees[subj] = duree_options_keys_ordered[sel_idx - 1]
                                        else: all_mapped = False; break # Index invalide
                                   else: all_mapped = False; break # Forfait manquant
                               else: all_mapped = False; break # Plus de sélections que de sujets
                           if all_mapped: session_state.responses['selected_types_duree'] = selected_durees
                           else:
                               session_state.flags['needs_clarification'] = True
                               session_state.flags['validation_errors'] = ["Erreur interne lors du mapping des types de durée."]
                      # --- Mapping Groupe ---
                      elif context == 'groupe':
                           selected_groups_ids = {} # Stocker ID
                           selected_groups_details_map = {} # Stocker détails
                           data_source = session_state.get('all_groups_for_selection', {})
                           all_mapped = True
                           for i, sel_idx in enumerate(selections):
                               if i < len(subjects_context):
                                   subj = subjects_context[i]
                                   options = data_source.get(subj, [])
                                   if 0 <= sel_idx - 1 < len(options):
                                       chosen_group = options[sel_idx - 1]
                                       selected_groups_ids[subj] = chosen_group['id_cours']
                                       selected_groups_details_map[subj] = chosen_group
                                   else: all_mapped = False; break # Index invalide
                               else: all_mapped = False; break # Plus de sélections que de sujets
                           if all_mapped:
                                session_state.responses['group_selections'] = selected_groups_ids
                                session_state.selected_groups_details = selected_groups_details_map
                                session_state.flags['overlap_conflict'] = False # Reset overlap flag si nouvelle sélection
                           else:
                                session_state.flags['needs_clarification'] = True
                                session_state.flags['validation_errors'] = ["Erreur interne lors du mapping des groupes."]

         else: # Validation échouée
             session_state.flags['needs_clarification'] = True
             session_state.flags['validation_errors'] = validation_result["errors"]

    # --- Fin Validation ---

    # 5. Déterminer la prochaine action (soit clarification, soit via determine_next_action)
    if session_state.flags.get('needs_clarification'):
        # <<< CORRECTION: Définir next_action ici si clarification nécessaire >>>
        next_action = {
            "action": "ask_clarification",
            "errors": session_state.flags.get('validation_errors', []),
            "original_input": session_state.flags.get('last_user_input', '')
        }
    else:
        # Pas besoin de clarification, déterminer la prochaine étape normale
        next_action = determine_next_action(session_state)
        action_code = next_action.get("action")

        # 6. Exécuter la logique métier Python si nécessaire (basé sur next_action)
        #    (Cette partie semble correcte et modifie next_action ou bot_action_results si besoin)
        #    Exemple:
        if action_code == "check_student_status":
            student_name = session_state.responses.get('student_name')
            try:
                # Utiliser get avec filtre where pour recherche exacte (plus robuste)
                # students_data = collection_students.get(where={"student_name": student_name}) # Marche si metadata indexé
                # Alternative: récupérer tout et filtrer (moins efficace si bcp d'étudiants)
                all_students = collection_students.get(include=["metadatas"])
                found_student = next((meta for meta in all_students.get('metadatas',[]) if meta.get('student_name','').strip().lower() == student_name.strip().lower()), None)
                is_new = not bool(found_student)

                session_state.flags['student_status_checked'] = True
                session_state.flags['student_is_new'] = is_new
                bot_action_results = {"is_new": is_new}
                next_action = {"action": "student_status"} # Forcer l'annonce maintenant
            except Exception as e:
                print(f"Erreur recherche étudiant {student_name}: {e}")
                session_state.flags['student_status_checked'] = True
                session_state.flags['student_is_new'] = True # Supposer nouveau en cas d'erreur
                bot_action_results = {"is_new": True}
                next_action = {"action": "student_status"}

        elif action_code == "get_forfaits":
             subject = next_action.get("subject")
             level = session_state.responses.get('user_level')
             if level and subject:
                  forfaits = get_available_forfaits(level, subject)
                  # Stocker même si vide, pour savoir qu'on a cherché
                  session_state.setdefault('available_forfaits', {})[subject] = forfaits
                  # L'action suivante sera de poser la question de sélection
                  next_action = {"action": "ask_question", "info_key": "selected_forfaits", "subject": subject, "results": {"forfaits": forfaits}}
             else: next_action = {"action": "error", "message": "Niveau ou matière manquant pour chercher les forfaits."}

        elif action_code == "get_recommendations":
             # ... (logique get_recommendations inchangée) ...
             # S'assurer qu'elle modifie bien next_action ou bot_action_results
              try:
                 student_name = session_state.responses.get('student_name')
                 user_level = session_state.responses.get('user_level')
                 user_subjects_list = session_state.get('matched_subjects', []) # Utiliser la liste nettoyée
                 user_teachers_raw = session_state.responses.get('user_teachers', '')
                 user_school = session_state.responses.get('user_school')
                 user_center = session_state.responses.get('user_center') # Peut être None
                 selected_forfaits_dict = session_state.responses.get('selected_forfaits', {})
                 selected_types_duree_dict = session_state.responses.get('selected_types_duree', {})

                 # Vérifier que toutes les données sont présentes avant l'appel
                 if not all([student_name, user_level, user_subjects_list, user_school]):
                      next_action = {"action": "error", "message": "Informations essentielles manquantes (nom, niveau, matières, école) pour obtenir les recommandations."}
                 elif user_subjects_list and not all(s in selected_forfaits_dict and s in selected_types_duree_dict for s in user_subjects_list):
                      next_action = {"action": "error", "message": "Détails de forfait/durée manquants pour certaines matières de groupe."}
                 else:
                      output_msgs, groups_select = get_recommendations(
                           student_name, user_level, user_subjects_list, user_teachers_raw, user_school,
                           user_center, selected_forfaits_dict, selected_types_duree_dict
                      )
                      session_state.all_groups_for_selection = groups_select
                      session_state.flags['recommendations_shown'] = True
                      bot_action_results = {"output": output_msgs, "groups_for_selection": groups_select}
                      if not groups_select or not any(groups_select.values()): # Vérifier si des groupes ont été retournés
                          next_action = {"action": "no_recommendations_found"}
                      else:
                          next_action = {"action": "show_recommendations"} # Action pour NLG

              except Exception as e:
                  print(f"Erreur dans get_recommendations call: {e}")
                  next_action = {"action": "error", "message": f"Erreur technique ({type(e).__name__}) lors de la recherche des groupes."}


        elif action_code == "calculate_tariffs":
             # ... (logique calculate_tariffs avec check_overlaps inchangée) ...
             # S'assurer qu'elle modifie bien next_action ou bot_action_results
             selected_groups_details = session_state.get('selected_groups_details', {})
             if selected_groups_details:
                 overlaps = check_overlaps(selected_groups_details)
                 if overlaps:
                     session_state.flags['overlap_conflict'] = True
                     overlap_details_html = "<ul>"
                     for g1, g2 in overlaps:
                          overlap_details_html += f"<li>Chevauchement entre <b>{g1.get('matiere','?')}</b> ({g1.get('jour','?')}, {g1.get('heure_debut','?')}-{g1.get('heure_fin','?')}) et <b>{g2.get('matiere','?')}</b> ({g2.get('jour','?')}, {g2.get('heure_debut','?')}-{g2.get('heure_fin','?')})</li>"
                     overlap_details_html += "</ul>"
                     # Réinitialiser sélection pour forcer re-choix
                     session_state.responses.pop('group_selections', None)
                     session_state.selected_groups_details = {}
                     session_state.flags.pop('tariffs_calculated', None) # Annuler calcul tarif
                     # Action pour informer et déclencher re-sélection au tour suivant
                     next_action = {"action": "handle_overlap", "results": {"overlaps_details_html": overlap_details_html}}

                 else: # Pas d'overlap, calculer
                     try:
                         tariffs_by_group, tariff_msg_html, total_base_val, total_after_auto_reduc_val = calculate_tariffs(selected_groups_details)
                         if tariffs_by_group is None:
                              next_action = {"action": "error", "message": tariff_msg_html or "Erreur inconnue calcul tarif."}
                         else:
                              session_state.tariffs_by_group = tariffs_by_group
                              session_state.total_tariff_base = total_base_val
                              session_state.total_after_auto_reduc = total_after_auto_reduc_val
                              session_state.flags['tariffs_calculated'] = True
                              bot_action_results = {"tariff_message_html": tariff_msg_html, "total_base": total_after_auto_reduc_val}
                              next_action = {"action": "show_tariffs"}
                              # Ajouter sous-action si frais inscription nécessaire
                              if len(selected_groups_details) > 0 and 'inscription_fee_choice' not in session_state.responses:
                                   next_action["sub_action"] = "ask_inscription_fee"
                              else: # Calculer total pour réduc demandée
                                   frais = 250 if session_state.responses.get('inscription_fee_choice') else 0
                                   session_state.current_total_for_discount = total_after_auto_reduc_val + frais
                     except Exception as e:
                          print(f"Erreur dans calculate_tariffs call: {e}")
                          next_action = {"action": "error", "message": f"Erreur technique ({type(e).__name__}) calcul tarifs."}
             else: # Pas de groupes sélectionnés
                 session_state.flags['tariffs_calculated'] = True
                 session_state.total_tariff_base = 0
                 session_state.total_after_auto_reduc = 0
                 session_state.current_total_for_discount = 0
                 # Déterminer la prochaine action (probablement commentaires)
                 next_action = determine_next_action(session_state) # Rappeler pour avancer

        elif action_code == "show_final_summary":
             # ... (logique show_final_summary inchangée) ...
             # S'assurer qu'elle assigne bot_action_results
             total_final = session_state.get('total_after_auto_reduc', 0)
             frais = 250 if session_state.responses.get('inscription_fee_choice') else 0
             total_final += frais
             reduc_percent = session_state.responses.get('reduction_percentage', 0)
             reduc_amount = 0
             base_for_reduc_dem = total_final # Base avant réduc demandée
             if reduc_percent > 0:
                 reduc_amount = base_for_reduc_dem * (reduc_percent / 100)
                 total_final -= reduc_amount

             # Construire le message HTML final détaillé
             final_details_html = ""
             # Ajouter détails par groupe s'ils existent
             if session_state.get('tariffs_by_group'):
                 final_details_html += "<b>Récapitulatif :</b><br>"
                 for subj, info in session_state.get('tariffs_by_group', {}).items():
                     final_details_html += f"- {subj} ({info['nom_forfait']}, {info['nom_type_duree']}): {info['tarif_total']:.2f} DH<br>"
                 final_details_html += f"<b>Sous-total :</b> {session_state.get('total_tariff_base', 0):.2f} DH<br>"
                 # Ajouter réduction auto si appliquée
                 reduc_auto = session_state.get('total_tariff_base', 0) - session_state.get('total_after_auto_reduc', 0)
                 if reduc_auto > 0.01: # Marge pour erreurs float
                      reduc_auto_percent = (reduc_auto / session_state.get('total_tariff_base', 1)) * 100 # Eviter div par 0
                      final_details_html += f"Réduction combinaison: -{reduc_auto:.2f} DH ({reduc_auto_percent:.1f}%)<br>"
                      final_details_html += f"<b>Total après réduc auto :</b> {session_state.get('total_after_auto_reduc', 0):.2f} DH<br>"

             # Ajouter frais et réduction demandée
             if frais > 0: final_details_html += f"<b>Frais d'inscription :</b> +{frais:.2f} DH<br>"
             if reduc_percent > 0: final_details_html += f"<b>Réduction demandée ({reduc_percent:.1f}%) :</b> -{reduc_amount:.2f} DH<br>"
             final_details_html += f"<br><b>MONTANT TOTAL FINAL : {total_final:.2f} DH</b>"

             bot_action_results = {"final_total": total_final, "tariff_details_html": final_details_html}
             session_state.flags['final_summary_shown'] = True
             # L'action reste show_final_summary pour NLG

        elif action_code == "end_conversation":
             # Gérer le cas 'oui' pour réinitialiser (déjà géré dans l'interface principale)
             # Gérer le cas 'non' -> action "finalize" pour NLG
             if not session_state.responses.get('another_case_choice', True): # Si 'non' a été choisi
                 next_action = {"action": "finalize"}
             else: # Si 'oui' a été choisi ou état inconnu, ne rien faire ici (reset géré ailleurs)
                 pass

        # --- Fin de la logique métier ---

    # 7. Génération de la réponse Bot (NLG)
    # next_action est maintenant garantie d'être définie
    nlg_prompt = create_nlg_prompt(next_action, bot_action_results, session_state)
    bot_message_html = llm_call(nlg_prompt, task_type='nlg')

    # Stocker le message généré pour l'historique (si ce n'est pas une action silencieuse)
    if next_action.get("action") not in ["end_conversation"]: # Ne pas ajouter de message si on termine juste
         session_state.messages.append((bot_message_html, True))

    # Stocker le message des tarifs pour le récap final
    if next_action.get("action") == "show_tariffs":
         session_state['generated_tariff_message'] = bot_action_results.get("tariff_message_html","")

# --- Interface Streamlit (adaptée) ---

st.markdown("""
    <style>
    /* ... (Styles CSS inchangés) ... */
    .stApp { background-color: #black; } /* Fond légèrement différent */
    .bot-message { background-color: #e3f2fd; border-radius: 15px; padding: 12px; margin: 8px 0; width: 75%; float: left; color: #black; font-family: Arial, sans-serif; border: 1px solid #bbdefb; }
        .bot-message {
        background-color: #e3f2fd; /* Fond bleu clair */
        border-radius: 15px;
        padding: 12px;
        margin: 8px 0;
        width: 75%;
        float: left;
        color: black; /* MODIFIÉ: Couleur du texte en noir */
        font-family: Arial, sans-serif;
        border: 1px solid #bbdefb;
    }
    .container { overflow-y: auto; height: 500px; /* Hauteur fixe pour scroll */ border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background-color: white;} /* Encadrement */
    h4 { color: #00796b; margin-bottom: 5px; }
    b { color: #004d40; }
    /* Input fixe en bas */
    .main .stTextInput { position: fixed; bottom: 10px; width: calc(100% - 350px); /* Ajuster selon largeur sidebar */ left: 310px; background-color: #f0f2f6; padding: 10px 0; z-index: 99; }
    </style>
""", unsafe_allow_html=True)

# --- Interface Streamlit ---
logo_path = os.path.join(parent_dir, "images", "logo.png")
try:
    st.image(logo_path, width=500) # Logo un peu plus petit
except FileNotFoundError:
    st.warning("Logo non trouvé.")

st.title("MentorBot Recommandations ✨")

profile_path = os.path.join(parent_dir, "images", "profile1.png")
with st.sidebar:
    st.image(profile_path, width=280, use_container_width=False, output_format="auto")
    st.markdown("<div class='profile-name'>ELARACHE Jalal</div>", unsafe_allow_html=True) # Gardé comme demandé
    st.header("Options")
    st.write("Votre assistant pour trouver les meilleurs groupes !")

    if st.button("🔄 Réinitialiser la conversation"):
        # MODIFIÉ: Logique de reset adaptée au nouvel état
        keys_to_clear = list(st.session_state.keys()) # Obtenir les clés avant modification
        for key in keys_to_clear:
             # Garder les clés essentielles de Streamlit et notre état initial
             if key not in ['messages', 'needed_info', 'flags', 'responses', 'current_input', 'input_key_counter']:
                 # On pourrait vouloir garder d'autres choses comme les listes chargées
                 pass # Ne pas supprimer les listes, etc.
             elif key not in ['input_key_counter']: # Ne pas supprimer le compteur pour clé unique
                 st.session_state.pop(key)

        # Réinitialiser l'état pour une nouvelle conversation
        st.session_state.messages = [("<div class='bot-message'>Bonjour ! Je suis MentorBot, prêt à vous aider à trouver les groupes parfaits. Pour commencer, quel est le prénom de l'étudiant(e) ? 🧑‍🎓</div>", True)]
        st.session_state.responses = {}
        st.session_state.needed_info = {'student_name'} # Recommencer par le nom
        st.session_state.flags = {} # Reset flags
        st.session_state.current_input = ""
        # Réinitialiser les données calculées spécifiques
        st.session_state.available_forfaits = {}
        st.session_state.available_types_duree = {}
        st.session_state.selected_forfaits = {}
        st.session_state.selected_types_duree = {}
        st.session_state.subject_grades = {}
        st.session_state.course_choices = {}
        st.session_state.all_groups_for_selection = {}
        st.session_state.selected_groups_details = {}
        st.session_state.tariffs_by_group = {}
        st.session_state.total_tariff_base = 0
        st.session_state.total_after_auto_reduc = 0
        st.session_state.current_total_for_discount = 0
        st.session_state.matched_subjects = []
        st.rerun()

    st.write("---")
    st.write("**À propos**")
    st.write("Chatbot amélioré avec l'IA Gemini pour une expérience plus fluide.")

# Initialiser l'état de la session (MODIFIÉ)
if 'messages' not in st.session_state:
    st.session_state.messages = [("<div class='bot-message'>Bonjour ! Je suis MentorBot, prêt à vous aider à trouver les groupes parfaits. Pour commencer, quel est le prénom de l'étudiant(e) ? 🧑‍🎓</div>", True)]
    st.session_state.responses = {}
    st.session_state.needed_info = {'student_name'} # Info initiale nécessaire
    st.session_state.flags = {} # Pour états spécifiques (discount_requested, etc.)
    st.session_state.current_input = ""
    st.session_state.input_key_counter = 0 # Pour clés uniques de text_input
    # Initialiser les autres états si besoin
    st.session_state.available_forfaits = {}
    st.session_state.available_types_duree = {}
    st.session_state.selected_forfaits = {}
    st.session_state.selected_types_duree = {}
    st.session_state.subject_grades = {}
    st.session_state.course_choices = {}
    st.session_state.all_groups_for_selection = {}
    st.session_state.selected_groups_details = {}
    st.session_state.tariffs_by_group = {}
    st.session_state.total_tariff_base = 0
    st.session_state.total_after_auto_reduc = 0
    st.session_state.current_total_for_discount = 0
    st.session_state.matched_subjects = []


# Afficher les messages précédents dans un container scrollable
st.markdown("<div class='container' id='message-container'>", unsafe_allow_html=True)
for message, is_bot in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Utiliser JS pour scroller en bas (Optionnel mais améliore UX)
st.markdown("""
<script>
    const container = document.getElementById('message-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)


# Champ de saisie utilisateur (MODIFIÉ)
# Utiliser un compteur pour générer une clé unique à chaque re-run où l'input change
current_input_key = f"user_input_{st.session_state.input_key_counter}"

# Utiliser st.empty() pour potentiellement remplacer le champ après la fin
input_placeholder = st.empty()

# Vérifier si la conversation doit se terminer
conversation_active = True
if st.session_state.flags.get('final_summary_shown') and st.session_state.responses.get('another_case_choice') == False:
     conversation_active = False

if conversation_active:
     # Créer une fonction pour gérer la soumission via on_change ou bouton
     def submit_input():
         user_input = st.session_state[current_input_key]
         if user_input:
             st.session_state.current_input = user_input # Stocker la valeur soumise
             # Incrémenter compteur pour forcer nouveau widget au prochain tour si nécessaire
             st.session_state.input_key_counter += 1
             # Flag pour indiquer qu'on doit traiter l'input
             st.session_state.process_now = True
             # Effacer le champ après soumission (optionnel)
             # st.session_state[current_input_key] = ""


     # Afficher le champ de saisie
     input_placeholder.text_input(
         "Votre réponse :",
         key=current_input_key,
         on_change=submit_input, # Traiter quand l'utilisateur quitte le champ ou appuie Entrée
         placeholder="Écrivez votre message ici..." # Placeholder générique
     )

     # Traiter l'entrée si le flag process_now est activé
     if st.session_state.get("process_now", False):
         user_input_to_process = st.session_state.current_input
         st.session_state.process_now = False # Reset flag
         st.session_state.current_input = "" # Reset buffer

         if user_input_to_process:
              # Gérer le cas 'oui' pour recommencer
              if st.session_state.flags.get('final_summary_shown') and \
                 'another_case_choice' in st.session_state.responses and \
                 st.session_state.responses['another_case_choice'] == True:
                   # Logique de reset (similaire au bouton)
                   keys_to_clear = list(st.session_state.keys())
                   for key in keys_to_clear:
                       if key not in ['input_key_counter']: st.session_state.pop(key)
                   st.session_state.messages = [("<div class='bot-message'>Ok, c'est reparti ! Quel est le prénom du nouvel étudiant(e) ? 🧑‍🎓</div>", True)]
                   st.session_state.responses = {}
                   st.session_state.needed_info = {'student_name'}
                   st.session_state.flags = {}
                   st.session_state.current_input = ""
                   st.rerun()
              else:
                  # Appeler la fonction principale de traitement
                  with st.spinner("MentorBot réfléchit... 🤔"):
                       process_user_turn(user_input_to_process, st.session_state)
                  st.rerun() # Re-run pour afficher la nouvelle réponse du bot

else: # Conversation inactive
     input_placeholder.markdown("<div class='bot-message'>Conversation terminée. Cliquez sur 'Réinitialiser' pour recommencer.</div>", unsafe_allow_html=True)