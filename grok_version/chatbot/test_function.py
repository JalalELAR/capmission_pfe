import chromadb
from chromadb import Client
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
chroma_path = os.path.join(parent_dir, "chroma_db5")
client = chromadb.PersistentClient(path=chroma_path)
collection_students = client.get_or_create_collection(name="students_vectorises")

def search_students_by_name(student_name, collection_students):
    try:
        # Récupérer la collection
        collection = collection_students
        
        # Effectuer la requête pour trouver les étudiants avec le même nom
        results = collection.query(
            query_texts=[student_name],
            n_results=10,  # Limite à 10 résultats pour éviter une surcharge
            where={"student_name": student_name}  # Filtre exact sur le nom
        )
        
        # Vérifier s'il y a des résultats
        if results["ids"] and results["ids"][0]:
            students = []
            for id, metadata in zip(results["ids"][0], results["metadatas"][0]):
                student = {
                    "student_name": metadata.get("student_name", ""),
                    "niveau": metadata.get("niveau", ""),
                    "ecole": metadata.get("ecole", "")
                }
                students.append(student)
            return students
        return []
    
    except Exception as e:
        print(f"Erreur lors de la recherche des étudiants : {e}")
        return []

# Exemple d'utilisation
if __name__ == "__main__":
    # Simuler une recherche
    sample_students = search_students_by_name("Kenza Alami", collection_students)
    if not sample_students:
        print("Aucun étudiant trouvé avec ce nom.")
    else:
        print(f"Étudiants trouvés : {len(sample_students)}")
    for student in sample_students:
        print(student)