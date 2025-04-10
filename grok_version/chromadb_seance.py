import psycopg2
from psycopg2 import Error
from psycopg2.extras import RealDictCursor
import chromadb

# Initialiser ChromaDB
client = chromadb.PersistentClient(path="./chroma_db5")

# Connexion à PostgreSQL
try:
    conn = psycopg2.connect(
        dbname="cm_db",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
    conn.set_client_encoding('UTF8')
    cursor = conn.cursor(cursor_factory=RealDictCursor)  # Utiliser RealDictCursor
    
    # Exécuter la requête
    print("Exécution de la requête SQL...")
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
        
        # Accéder aux colonnes par nom (RealDictCursor retourne des dictionnaires)
        ids_seances = [str(row['seance_id']) for row in seances_data]  # seance_id
        metadatas_seances = [{"date_seance": row['date_seance'], "id_cours": str(row['id_cours'])} for row in seances_data]
        documents_seances = [f"date_seance: {row['date_seance']}, id_cours: {row['id_cours']}" for row in seances_data]
        
        # Compter les séances pour 12734033
        seances_12734033_in_batch = sum(1 for row in seances_data if str(row['id_cours']) == '12734033')
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

except Error as e:
    print(f"Erreur lors de la vectorisation de `seances` : {e}")
except Exception as e:
    print(f"Erreur générale : {e}")
finally:
    cursor.close()
    conn.close()