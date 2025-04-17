import chromadb

# Connexion à ChromaDB
client = chromadb.PersistentClient(path="./chroma_db5")
collection = client.get_collection(name="groupes_vectorises9")

# Récupérer toutes les métadonnées
all_groups = collection.get(include=["metadatas"])

# Vérifier les forfaits spécifiques
target_forfaits = ["12677992", "12678012"]
found = False
for metadata in all_groups['metadatas']:
    id_forfait = metadata.get('id_forfait')
    if id_forfait in target_forfaits:
        found = True
        print(f"Forfait trouvé : "
              f"id_cours={metadata['id_cours']}, "
              f"matière={metadata['matiere']}, "
              f"niveau={metadata['niveau']}, "
              f"id_forfait={id_forfait}, "
              f"nom_forfait={metadata.get('nom_forfait')}, "
              f"type_duree={metadata.get('type_duree')}, "
              f"type_duree_id={metadata.get('type_duree_id')}, "
              f"tarifunitaire={metadata.get('tarifunitaire')}, "
              f"duree_tarifs={metadata.get('duree_tarifs')}")

if not found:
    print(f"Aucun groupe trouvé pour les forfaits {target_forfaits}.")