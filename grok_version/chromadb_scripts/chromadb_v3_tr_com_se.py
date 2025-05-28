import chromadb
import psycopg2
from psycopg2.extras import RealDictCursor

# Connexion à ChromaDB
client = chromadb.PersistentClient(path="./chroma_db5")

# Connexion à PostgreSQL
conn = psycopg2.connect(
        dbname="cm_db",
        user="postgres",
        password="root",
        host="localhost",
        port="5432",
    cursor_factory=RealDictCursor
)
cursor = conn.cursor()

# 1. Vectorisation de la table `tarifs`
cursor.execute("""
    SELECT DISTINCT cc.id AS cours_id, cf.id AS forfait_id, co.tarifunitaire
    FROM cm_cours cc
    JOIN cm_forfait cf ON cc.offre = cf.id
    JOIN cm_offretemporelle co ON co.offregeneric_id = cf.id
    WHERE cc.deleted = FALSE AND cc.offre IS NOT NULL AND cf.deleted = FALSE AND co.deleted = FALSE
    ORDER BY cf.id, cc.id
""")
tarifs_data = cursor.fetchall()

collection_tarifs = client.get_or_create_collection(name="tarifs_vectorises")
ids_tarifs = [str(row['cours_id']) for row in tarifs_data]
metadatas_tarifs = [{"id_forfait": row['forfait_id'], "tarif_unitaire": float(row['tarifunitaire'])} for row in tarifs_data]
documents_tarifs = [f"id_forfait: {row['forfait_id']}, tarif_unitaire: {row['tarifunitaire']}" for row in tarifs_data]

collection_tarifs.upsert(
    ids=ids_tarifs,
    metadatas=metadatas_tarifs,
    documents=documents_tarifs
)
print("Vectorisation de `tarifs` terminée.")

# 2. Vectorisation de la table `combinaisons`
cursor.execute("""
    SELECT idcombinaison, idforfait, reduction
    FROM cm_combinaison_element
    WHERE deleted = FALSE
    ORDER BY idcombinaison
""")
combinaisons_data = cursor.fetchall()

collection_combinaisons = client.get_or_create_collection(name="combinaisons_vectorises")
ids_combinaisons = [f"{row['idcombinaison']}_{row['idforfait']}" for row in combinaisons_data]
metadatas_combinaisons = [{"id_combinaison": row['idcombinaison'], "id_forfait": row['idforfait'], "reduction": float(row['reduction'])} for row in combinaisons_data]
documents_combinaisons = [f"id_combinaison: {row['idcombinaison']}, id_forfait: {row['idforfait']}, reduction: {row['reduction']}" for row in combinaisons_data]

collection_combinaisons.upsert(
    ids=ids_combinaisons,
    metadatas=metadatas_combinaisons,
    documents=documents_combinaisons
)
print("Vectorisation de `combinaisons` terminée.")

# 3. Vectorisation de la table `seances`
# Réinitialiser la collection pour garantir des données fraîches
# Exécuter la requête
cursor.execute("""
    SELECT 
        id AS seance_id, 
        to_char(START, 'YYYY/MM/DD') AS date_seance, 
        seance_cours AS id_cours
    FROM cm_seance
    WHERE deleted = FALSE AND seance_cours IS NOT NULL AND START > '2024/01/01'
    """)
    
# Vérifier si des résultats existent
if cursor.rowcount == -1 or cursor.rowcount == 0:
    print("Aucun résultat retourné par la requête. Vérifiez la table cm_seance ou les conditions.")
    raise Exception("Requête vide ou non exécutée.")

# Réinitialiser la collection
try:
    client.delete_collection("seances_vectorises")
    print("Collection seances_vectorises réinitialisée.")
except:
    pass
collection_seances = client.create_collection(name="seances_vectorises")

MAX_BATCH_SIZE = 5461
total_seances = 0
seances_12734033 = 0

while True:
    seances_data = cursor.fetchmany(MAX_BATCH_SIZE)
    if not seances_data:
        break
    
    # Accéder aux colonnes par index (si tuples)
    ids_seances = [str(row[0]) for row in seances_data]  # seance_id
    metadatas_seances = [{"date_seance": row[1], "id_cours": str(row[2])} for row in seances_data]  # date_seance, id_cours
    documents_seances = [f"date_seance: {row[1]}, id_cours: {row[2]}" for row in seances_data]
    
    # Compter les séances pour 12734033
    seances_12734033_in_batch = sum(1 for row in seances_data if str(row[2]) == '12734033')
    seances_12734033 += seances_12734033_in_batch
    
    collection_seances.upsert(
        ids=ids_seances,
        metadatas=metadatas_seances,
        documents=documents_seances
    )
    total_seances += len(seances_data)
print(f"Vectorisé {total_seances} séances... Dont {seances_12734033_in_batch} pour id_cours 12734033 dans ce lot.")

print(f"Vectorisation terminée. Total séances : {total_seances}, Total pour 12734033 : {seances_12734033}")

# Vérification post-vectorisation
seances_data_check = collection_seances.get(include=["metadatas"])
relevant_seances = [m for m in seances_data_check['metadatas'] if m['id_cours'] == '12734033']
print(f"Séances pour 12734033 après vectorisation : {len(relevant_seances)}")
for seance in relevant_seances:
    print(f"Séance - ID: {seance.get('id', 'N/A')}, Date: {seance['date_seance']}, id_cours: {seance['id_cours']}")

# Fermer la connexion PostgreSQL
cursor.close()
conn.close()