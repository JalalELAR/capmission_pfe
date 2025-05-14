import psycopg2
from psycopg2.extras import RealDictCursor
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration PostgreSQL
DB_CONFIG = {
    "dbname": "cm_db",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}

# Configuration ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def vectoriser_niveaux():
    try:
        # 1. Connexion à PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 2. Extraction des données
        cursor.execute("SELECT name FROM cm_niveau cn;")
        resultats = cursor.fetchall()
        
        # 3. Préparation des embeddings
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for idx, niveau in enumerate(resultats):
            nom_niveau = niveau['name']
            
            documents.append(nom_niveau)
            metadatas.append({"niveau": nom_niveau})
            ids.append(f"niv_{idx+1}")
            embeddings.append(model.encode(nom_niveau).tolist())
        
        # 4. Stockage dans ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db5")
        collection = client.get_or_create_collection("niveaux")
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        print(f"{len(documents)} niveaux vectorisés avec succès !")
        
    except Exception as e:
        print(f"Erreur : {str(e)}")
    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()

# Installation requise (à exécuter une fois)
# pip install psycopg2-binary chromadb sentence-transformers

# Exécution
vectoriser_niveaux()

client = chromadb.PersistentClient(path="./chroma_db5")
collection = client.get_or_create_collection("niveaux")
# Récupérer tous les documents, métadonnées et ids
resultats = collection.get()

# Afficher les niveaux vectorisés
for idx, (doc, meta, id_) in enumerate(zip(resultats['documents'], resultats['metadatas'], resultats['ids'])):
    print(f"Niveau {idx+1} - ID: {id_} - Document: {doc} - Métadonnées: {meta}")