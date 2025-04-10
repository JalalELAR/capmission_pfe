import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import random
from datetime import datetime, timedelta

client = chromadb.PersistentClient(path="../chroma_db5")
collection_groupes = client.get_collection(name="groupes_vectorises6")
group_data = collection_groupes.get(ids=["12740084"], include=["metadatas"])
print(group_data['metadatas'])