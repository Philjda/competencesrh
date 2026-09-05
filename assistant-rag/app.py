
"""
Assistant RAG générique
------------------------
Chatbot Gradio qui télécharge tes PDF (hébergés sur GitHub ou ailleurs),
puis répond aux questions des utilisateurs en s'appuyant uniquement sur
leur contenu (recherche sémantique + génération sous contrainte).

Prêt à être déposé sur Hugging Face Spaces (sdk: gradio).
"""

import io
import os
import numpy as np
import requests
import gradio as gr
import torch
import transformers
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

transformers.logging.set_verbosity_error()

# ========================================================================
# ⚙️  CONFIGURATION — à adapter à ton projet
# ========================================================================

# Nom et rôle de ton assistant (affichés dans l'interface et utilisés
# dans le prompt envoyé au modèle).
NOM_ASSISTANT = "Mon Assistant Documentaire"
ROLE = (
    "Tu es un assistant qui répond aux questions des utilisateurs de façon "
    "précise, utile et directe, en te basant uniquement sur les documents fournis."
)

# 👉 Mets ici l'URL "raw" de ton/tes PDF hébergé(s) sur GitHub.
# Sur GitHub : ouvre le fichier PDF → bouton "Raw" → copie l'URL.
# Elle ressemble à :
#   https://raw.githubusercontent.com/TON-UTILISATEUR/TON-REPO/main/mon-fichier.pdf
URLS_PDF = [
    "https://raw.githubusercontent.com/TON-UTILISATEUR/TON-REPO/main/mon-fichier.pdf",
]

# Message affiché si aucun passage pertinent n'est trouvé pour une question.
REPONSE_INCONNUE = "Je ne sais pas, il faudrait vérifier auprès d'un humain."

# Questions d'exemple affichées dans l'interface (facultatif).
EXEMPLES = [
    "Que contient ce document ?",
]

# Dossier optionnel du dépôt : place ici des PDF supplémentaires en local
# (ex: documents/annexe.pdf). Ils s'ajoutent à ceux de URLS_PDF ci-dessus.
DOSSIER_LOCAL = "documents"

# ========================================================================
# 1. Choix du modèle selon le matériel disponible
# ========================================================================
if torch.cuda.is_available():
    MODELE = "Qwen/Qwen2.5-1.5B-Instruct"
    DEVICE = 0
    print("🚀 GPU détecté : on utilise le moteur rapide.")
else:
    MODELE = "Qwen/Qwen2.5-0.5B-Instruct"
    DEVICE = -1
    print("🐢 Pas de GPU : on utilise le moteur léger.")

# ========================================================================
# 2. Téléchargement et extraction des PDF
# ========================================================================
TEXTES = {}

# 2a. PDF distants (les URL que tu as renseignées dans URLS_PDF)
for url in URLS_PDF:
    nom_fichier = url.rstrip("/").split("/")[-1] or url
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        pages = PdfReader(io.BytesIO(r.content)).pages
        TEXTES[nom_fichier] = "\n".join((p.extract_text() or "") for p in pages)
        print(f"✅ (distant) {nom_fichier:32s} {len(pages)} page(s) · {len(TEXTES[nom_fichier]):5d} caractères")
    except Exception as e:
        print(f"⚠️ Impossible de télécharger {url} : {e}")

# 2b. PDF locaux ajoutés dans le dépôt (dossier documents/)
if os.path.isdir(DOSSIER_LOCAL):
    for nom_fichier in sorted(os.listdir(DOSSIER_LOCAL)):
        if nom_fichier.lower().endswith(".pdf"):
            chemin = os.path.join(DOSSIER_LOCAL, nom_fichier)
            try:
                pages = PdfReader(chemin).pages
                TEXTES[nom_fichier] = "\n".join((p.extract_text() or "") for p in pages)
                print(f"✅ (local)   {nom_fichier:32s} {len(pages)} page(s) · {len(TEXTES[nom_fichier]):5d} caractères")
            except Exception as e:
                print(f"⚠️ Impossible de lire {nom_fichier} : {e}")

if not TEXTES:
    print("❌ Aucun document n'a pu être chargé. Vérifie tes URL dans URLS_PDF "
          "ou ajoute des PDF dans le dossier documents/.")

print(f"📚 {len(TEXTES)} document(s) chargé(s).")

# ========================================================================
# 3. Découpage en passages
# ========================================================================
def decouper(texte, taille=500, chevauchement=80):
    """Découpe un texte en passages d'environ `taille` caractères,
    en coupant de préférence en fin de phrase."""
    texte = " ".join(texte.split())
    passages, debut = [], 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            coupe = texte.rfind(". ", debut + taille // 2, fin)
            if coupe != -1:
                fin = coupe + 1
        passages.append(texte[debut:fin].strip())
        if fin >= len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)
    return [p for p in passages if p]


DOCUMENTS = []
for f, t in TEXTES.items():
    nom = f.replace(".pdf", "").replace("-", " ").replace("_", " ").title()
    for i, p in enumerate(decouper(t), 1):
        DOCUMENTS.append({"titre": f"{nom} · passage {i}", "texte": p})

print(f"✅ {len(DOCUMENTS)} passage(s) prêt(s) à être encodé(s).")

# ========================================================================
# 4. Encodage des passages (embeddings)
# ========================================================================
encodeur = None
vecteurs = None

if DOCUMENTS:
    print("Chargement du modèle d'embeddings...")
    encodeur = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    textes = [f"{d['titre']} - {d['texte']}" for d in DOCUMENTS]
    vecteurs = encodeur.encode(textes, normalize_embeddings=True)
    print(f"✅ {len(vecteurs)} passage(s) encodé(s) ({vecteurs.shape[1]} dimensions).")


def chercher(question, k=3, seuil=0.25):
    """Renvoie les k passages les plus proches de la question,
    en écartant ceux qui sont trop peu pertinents (seuil de similarité)."""
    if encodeur is None or vecteurs is None or len(DOCUMENTS) == 0:
        return []
    v_question = encodeur.encode(question, normalize_embeddings=True)
    similarites = vecteurs @ v_question
    indices = np.argsort(-similarites)[:k]
    resultats = [
        (DOCUMENTS[i]["titre"], DOCUMENTS[i]["texte"], float(similarites[i]))
        for i in indices
        if similarites[i] >= seuil
    ]
    return resultats

# ========================================================================
# 5. Chargement du modèle de génération
# ========================================================================
print("Chargement du modèle de génération (1 à 3 minutes la première fois)...")
generateur = pipeline("text-generation", model=MODELE, device=DEVICE)
generateur.tokenizer.clean_up_tokenization_spaces = False
generateur.model.generation_config.max_new_tokens = 180
generateur.model.generation_config.do_sample = False
generateur.model.generation_config.temperature = None
generateur.model.generation_config.top_p = None
generateur.model.generation_config.top_k = None
print("✅ Modèle chargé, l'assistant est prêt !")


def demander_au_modele(messages):
    sortie = generateur(messages)
    return sortie[0]["generated_text"][-1]["content"]


def repondre(question, k=3):
    """L'assistant complet : recherche + rédaction sous contrainte."""
    passages = chercher(question, k=k)

    if not passages:
        return REPONSE_INCONNUE, "aucune source pertinente trouvée"

    contexte = "\n\n".join(f"[{titre}]\n{texte}" for titre, texte, _ in passages)

    messages = [
        {
            "role": "system",
            "content": ROLE
            + " Tu réponds uniquement à partir des documents fournis. "
              f"Si la réponse n'y figure pas, réponds exactement : « {REPONSE_INCONNUE} »",
        },
        {"role": "user", "content": f"Documents :\n{contexte}\n\nQuestion : {question}"},
    ]

    reponse = demander_au_modele(messages)
    sources = ", ".join(t for t, _, _ in passages)
    return reponse, sources

# ========================================================================
# 6. Interface Gradio
# ========================================================================
def chat_fn(message, history):
    reponse, sources = repondre(message)
    return f"{reponse}\n\n📎 *Sources : {sources}*"


demo = gr.ChatInterface(
    fn=chat_fn,
    title=f"🤖 {NOM_ASSISTANT}",
    description=(
        "Posez une question : l'assistant répond uniquement à partir "
        "des documents qui lui ont été fournis."
    ),
    examples=EXEMPLES,
)

if __name__ == "__main__":
    demo.launch()
