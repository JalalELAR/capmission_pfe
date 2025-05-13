import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
import os
from tqdm import tqdm

# Configuration de la base de données (à personnaliser)
db_config = {
    "host": "localhost",
    "port": "5432",
    "database": "cm_db",
    "user": "postgres",
    "password": "root"
}

# Requête SQL avec student_id, niveau et école
sql_query = """
SELECT 
    ct.id, 
    concat(ct.firstname, ' ', ct.lastname) AS student_name,
    cn."name" AS niveau,
    ce."name" AS ecole
FROM cm_tiers ct 
LEFT JOIN mtm_role_tiers mrt ON ct.id = mrt.id_tiers 
LEFT JOIN cm_role cr ON cr.id = mrt.id_role 
LEFT JOIN mtm_niveau_student mns ON ct.id = mns.id_tiers 
LEFT JOIN cm_niveau cn ON cn.id = mns.id_niveau 
LEFT JOIN cm_ecole ce ON ce.id = ct.mto_student_ecole 
WHERE cr."name" = 'ROLE_STUDENT';
"""

def get_students_from_db():
    """Exécute la requête SQL et retourne la liste des étudiants avec leurs IDs, niveaux et écoles."""
    try:
        # Connexion à la base de données
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Exécuter la requête
        cursor.execute(sql_query)
        students = [(str(row[0]), row[1].strip(), row[2], row[3]) for row in cursor.fetchall() if row[1].strip()]
        
        # Fermer la connexion
        cursor.close()
        conn.close()
        
        print(f"Récupéré {len(students)} étudiants depuis la base de données.")
        return students
    
    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return []

def vectorize_students(students):
    """Vectorise les noms des étudiants avec leurs niveaux et écoles, et les stocke dans ChromaDB par lots."""
    # Initialiser le modèle SentenceTransformer
    print("Chargement du modèle SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialiser ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db5")
    collection = client.get_or_create_collection(name="students_vectorises")
    
    # Vider la collection existante (optionnel)
    print("Vidage de la collection students_vectorises existante...")
    client.delete_collection(name="students_vectorises")
    collection = client.create_collection(name="students_vectorises")
    
    # Taille maximale du lot (conservatrice pour éviter les erreurs)
    batch_size = 5000
    
    # Préparer les données
    ids = []
    metadatas = []
    documents = []
    embeddings = []
    
    print("Vectorisation des étudiants avec leurs niveaux et écoles (valeurs nulles acceptées)...")
    for student_id, student_name, niveau, ecole in tqdm(students, desc="Vectorisation"):
        if not student_name:
            continue
        # Remplacer les valeurs nulles par "Inconnu" pour la vectorisation
        niveau_text = niveau if niveau is not None else "Inconnu"
        ecole_text = ecole if ecole is not None else "Inconnu"
        # Combiner le nom, le niveau et l'école pour la vectorisation
        document = f"{student_name} - Niveau Actuel : {niveau_text} - École : {ecole_text}"
        # Générer l'embedding
        embedding = model.encode(document, convert_to_numpy=True).tolist()
        # Remplacer les valeurs nulles dans les métadonnées pour ChromaDB
        metadata_niveau = niveau if niveau is not None else "Inconnu"
        metadata_ecole = ecole if ecole is not None else "Inconnu"
        # Ajouter aux listes
        ids.append(student_id)
        metadatas.append({"student_name": student_name, "niveau": metadata_niveau, "ecole": metadata_ecole})
        documents.append(document)
        embeddings.append(embedding)
    
    # Insérer par lots
    total_students = len(ids)
    if total_students == 0:
        print("Aucun étudiant à insérer.")
        return
    
    print(f"Insertion de {total_students} étudiants dans ChromaDB par lots de {batch_size}...")
    for i in tqdm(range(0, total_students, batch_size), desc="Insertion des lots"):
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        collection.add(
            ids=batch_ids,
            metadatas=batch_metadatas,
            documents=batch_documents,
            embeddings=batch_embeddings
        )
    
    print("Insertion terminée avec succès.")

def main():
    # Étape 1 : Récupérer les étudiants depuis la base de données
    students = get_students_from_db()
    
    if not students:
        print("Aucune donnée à vectoriser. Fin du programme.")
        return
    
    # Étape 2 : Vectoriser et stocker dans ChromaDB
    vectorize_students(students)
    
    # Vérification finale
    client = chromadb.PersistentClient(path="./chroma_db5")
    collection = client.get_collection(name="students_vectorises")
    print(f"Total d'étudiants dans students_vectorises : {collection.count()}")

if __name__ == "__main__":
    main()