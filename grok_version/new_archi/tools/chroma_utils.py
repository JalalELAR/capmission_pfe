# chatbot_project/tools/chroma_utils.py
import os
import chromadb
from sentence_transformers import SentenceTransformer



client = chromadb.PersistentClient(path="./chroma_db5")
collection_groupes = client.get_collection(name="groupes_vectorises9")
collection_seances = client.get_or_create_collection(name="seances_vectorises")
collection_combinaisons = client.get_or_create_collection(name="combinaisons_vectorises")
collection_students = client.get_or_create_collection(name="students_vectorises")
collection_niveaux = client.get_or_create_collection(name="niveaux")
resultats = collection_niveaux.get() 
metadatas = resultats['metadatas']  # liste de dicts, ex: [{"niveau": "Primaire"}, {"niveau": "Secondaire"}, ...]
# Extraire les valeurs du champ "niveau"
niveaux = [meta["niveau"] for meta in metadatas if "niveau" in meta]

all_groups = collection_groupes.get(include=["metadatas"])
available_levels = set(metadata.get('niveau', '').strip().lower() for metadata in all_groups['metadatas'])
available_subjects = set(metadata.get('matiere', '').strip().lower() for metadata in all_groups['metadatas'])
# Récupérer les valeurs uniques depuis ChromaDB
schools = set()
subjects = set()
centers = set()
teachers = set()
for metadata in all_groups['metadatas']:
    try:
        schools.add(metadata['ecole'].split(", ")[0])
        subjects.add(metadata['matiere'])
        centers.add(metadata['centre'])
        teachers.add(metadata['teacher'])
    except (KeyError, IndexError):
        continue

schools_list = sorted(list(schools))
levels_list = sorted(list(niveaux))
subjects_list = sorted(list(subjects))
centers_list = sorted(list(centers))
teachers_list = sorted(list(teachers))

model = SentenceTransformer("all-MiniLM-L6-v2")

