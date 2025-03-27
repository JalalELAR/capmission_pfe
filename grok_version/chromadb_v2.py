import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2 import Error

# Connexion à PostgreSQL 9.6
try:
    conn = psycopg2.connect(
        dbname="cm_db",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
    conn.set_client_encoding('UTF8')
    print("Connexion à la base de données réussie.")
except Error as e:
    print(f"Erreur lors de la connexion à PostgreSQL : {e}")
    exit()

# Requête SQL
query = """
SELECT 
    cc.id AS id_cours,
    cc."name" AS name_cours,
    to_char(cc.date_debut,'YYYY/MM/DD') AS date_debut,
    to_char(cc.date_fin,'YYYY/MM/DD') AS date_fin,
    cc.heure_debut,
    cc.heure_fin,
    cc.jour,
    cc2."name" AS centre, 
    CONCAT(tea.firstname, ' ', tea.lastname) AS teacher,
    cn."name" AS niveau,
    cm."name" AS matiere,
    CONCAT(st.firstname, ' ', st.lastname) AS student,
    ce."name" AS ecole,
    (SELECT COUNT(*) FROM mtm_cours_student mcs WHERE mcs.id_seance = cc.id) AS nb_students
FROM cm_student_seance_forfait ssf
LEFT JOIN cm_forfait_for_student ffs ON ffs.id = ssf.forfait
LEFT JOIN cm_tiers st ON st.id = ssf.student
LEFT JOIN cm_ecole ce ON ce.id = st.mto_student_ecole
LEFT JOIN cm_niveau cn ON cn.id = ffs.mto_forfait_niveau
LEFT JOIN cm_matiere cm ON cm.id = ffs.mto_forfait_matiere
LEFT JOIN cm_seance cs ON cs.id = ssf.seance
LEFT JOIN cm_cours cc ON cc.id = cs.seance_cours
LEFT JOIN cm_centre cc2 ON cc2.id = cc.centre  
LEFT JOIN cm_tiers tea ON tea.id = cc.teacher
WHERE cc.id IS NOT NULL
AND cc.deleted = FALSE
GROUP BY cc.id, tea.id, cn.id, cm.id, st.id, ce.id, cc2.id 
ORDER BY cc.datecreation DESC;
"""

# Charger les données dans un DataFrame
try:
    df = pd.read_sql_query(query, conn)
    print(f"Données chargées avec succès : {len(df)} lignes récupérées.")
except Error as e:
    print(f"Erreur lors de l'exécution de la requête SQL : {e}")
    conn.close()
    exit()
finally:
    conn.close()

# Étape 1 : Calculer le nombre d'étudiants par id_cours
students_per_group = df.groupby('id_cours')['student'].apply(lambda x: len(set(x))).to_dict()

# Étape 2 : Regrouper les écoles par id_cours (préserver les répétitions)
schools_per_group = df.groupby('id_cours')['ecole'].apply(lambda x: ", ".join(x.dropna())).to_dict()

# Étape 3 : Regrouper les étudiants par id_cours
students_by_group = df.groupby('id_cours')['student'].apply(lambda x: ", ".join(set(x.dropna()))).to_dict()

# Initialiser ChromaDB
client = chromadb.PersistentClient(path="./chroma_db5")
collection_name = "groupes_vectorises5"
try:
    client.delete_collection(collection_name)
except:
    pass
collection = client.create_collection(name=collection_name)

# Initialiser le modèle pour les embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Préparer les données pour ChromaDB
documents = []
metadatas = []
ids = []

# Regrouper par id_cours (un groupe = un centre)
for id_cours, group in df.groupby('id_cours'):
    # Vérifier que le groupe est associé à un seul centre
    unique_centres = group['centre'].unique()
    if len(unique_centres) != 1:
        print(f"Erreur : Le groupe {id_cours} est associé à plusieurs centres : {unique_centres}")
        continue
    centre = unique_centres[0]

    # Description du groupe
    description = (
        f"Niveau: {group['niveau'].iloc[0]}, "
        f"Matière: {group['matiere'].iloc[0]}, "
        f"Centre: {centre}, "
        f"Enseignant: {group['teacher'].iloc[0]}, "
        f"Écoles: {schools_per_group[id_cours]}"
    ).encode('utf-8').decode('utf-8')
    
    # Liste des étudiants pour ce groupe
    students = students_by_group.get(id_cours, "")
    
    # Nombre d'étudiants pour ce groupe
    num_students = students_per_group.get(id_cours, 0)
    
    # Nombre total d'étudiants (via nb_students)
    total_students = group['nb_students'].iloc[0]
    
    # Métadonnées
    metadata = {
        "id_cours": str(id_cours),
        "name_cours": str(group['name_cours'].iloc[0]).encode('utf-8').decode('utf-8'),
        "num_students": str(num_students),
        "total_students": str(total_students),
        "student": students.encode('utf-8').decode('utf-8'),
        "ecole": schools_per_group[id_cours].encode('utf-8').decode('utf-8'),
        "centre": str(centre).encode('utf-8').decode('utf-8'),
        "teacher": str(group['teacher'].iloc[0]).encode('utf-8').decode('utf-8'),
        "date_debut": str(group['date_debut'].iloc[0]),
        "date_fin": str(group['date_fin'].iloc[0]),
        "heure_debut": str(group['heure_debut'].iloc[0]),
        "heure_fin": str(group['heure_fin'].iloc[0]),
        "jour": str(group['jour'].iloc[0]),
        "niveau": str(group['niveau'].iloc[0]).encode('utf-8').decode('utf-8'),
        "matiere": str(group['matiere'].iloc[0]).encode('utf-8').decode('utf-8')
    }
    
    # Ajouter à ChromaDB
    documents.append(description)
    metadatas.append(metadata)
    ids.append(str(id_cours))  # ID basé uniquement sur id_cours

# Générer les embeddings
try:
    embeddings = model.encode(documents).tolist()
except Exception as e:
    print(f"Erreur lors de la génération des embeddings : {e}")
    exit()

# Insérer dans ChromaDB
try:
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Données insérées dans ChromaDB avec succès.")
except Exception as e:
    print(f"Erreur lors de l'insertion dans ChromaDB : {e}")
    exit()

print("Vectorisation terminée !")