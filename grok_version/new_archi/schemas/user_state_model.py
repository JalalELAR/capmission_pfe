# chatbot_project/schemas/user_state_model.py
from typing import List, Optional
from pydantic import BaseModel

class GroupePref(BaseModel):
    type: Optional[str] = ""
    creneaux_dispos: List[str] = []

class MatiereInfo(BaseModel):
    nom: str
    note: Optional[float] = None
    enseignant: Optional[str] = None
    commentaire: Optional[str] = None

class UserState(BaseModel):
    user_id: str = ""
    nom: str = ""
    niveau: str = ""
    ecole: str = ""
    matières: List[MatiereInfo] = []
    choix_cours: str = ""
    disponibilités: List[str] = []
    groupe_pref: GroupePref = GroupePref()
