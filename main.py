import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import io # Ajouté pour le téléchargement
import tempfile # Ajouté pour les fichiers temporaires

import pandas as pd
import PyPDF2
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# --- NOUVELLES IMPORTATIONS GOOGLE DRIVE ---
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
# --- FIN NOUVELLES IMPORTATIONS ---

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------
# Configuration & Client Initialization
# --------------------------------------

def load_api_key() -> str:
    """Charge la clé API GROQ depuis .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("La variable d'environnement 'GROQ_API_KEY' n'est pas définie")
        raise ValueError("La variable d'environnement 'GROQ_API_KEY' n'est pas définie")
    logger.info("Clé API GROQ chargée.")
    return api_key

# --- NOUVELLES FONCTIONS DE CONFIGURATION GOOGLE DRIVE ---
def load_gdrive_config() -> Tuple[Optional[str], Optional[str]]:
    """Charge le chemin des credentials et l'ID du dossier GDrive depuis .env."""
    load_dotenv()
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

    if not credentials_path:
        logger.warning("Variable d'environnement 'GOOGLE_APPLICATION_CREDENTIALS' non définie. Authentification GDrive échouera.")
    else:
        logger.info("Chemin des credentials GDrive chargé.")

    if not folder_id:
        logger.warning("Variable d'environnement 'GOOGLE_DRIVE_FOLDER_ID' non définie. Impossible de localiser le dossier GDrive.")
    else:
        logger.info("ID du dossier GDrive chargé.")

    return credentials_path, folder_id
# --- FIN NOUVELLES FONCTIONS DE CONFIGURATION ---

def create_groq_client(api_key: str) -> Groq:
    """Crée et retourne un client Groq."""
    return Groq(api_key=api_key)

# --- NOUVELLES FONCTIONS GOOGLE DRIVE ---
def get_drive_service(credentials_path: str) -> Optional[Any]:
    """Crée et retourne un service Google Drive authentifié."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly'] # Lecture seule suffisante
    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        logger.info("Service Google Drive créé avec succès.")
        return service
    except FileNotFoundError:
        logger.error(f"Fichier de credentials GDrive non trouvé à : {credentials_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la création du service Google Drive : {e}", exc_info=True)
    return None

def list_pdfs_in_folder(service: Any, folder_id: str) -> List[Dict[str, str]]:
    """Liste les fichiers PDF dans un dossier Google Drive spécifié."""
    pdfs = []
    page_token = None
    try:
        while True:
            response = service.files().list(
                q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false",
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()

            files = response.get('files', [])
            pdfs.extend(files)
            logger.info(f"{len(files)} fichiers PDF trouvés dans ce batch.")

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        logger.info(f"Total de {len(pdfs)} fichiers PDF trouvés dans le dossier {folder_id}.")
        return pdfs
    except HttpError as error:
        logger.error(f"Erreur HTTP lors de la recherche de fichiers PDF : {error}")
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de fichiers PDF : {e}", exc_info=True)
    return []

def download_pdf_to_temp(service: Any, file_id: str, file_name: str, temp_dir: str) -> Optional[str]:
    """Télécharge un PDF depuis Drive et le sauvegarde dans un fichier temporaire."""
    # Crée un chemin de fichier temporaire sûr
    temp_pdf_path = os.path.join(temp_dir, f"temp_{file_id}_{os.path.basename(file_name)}")
    logger.info(f"Tentative de téléchargement de {file_name} (ID: {file_id}) vers {temp_pdf_path}")
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logger.info(f"Téléchargement {file_name}: {int(status.progress() * 100)}%.")

        # Écrire les données téléchargées dans le fichier temporaire
        fh.seek(0)
        with open(temp_pdf_path, 'wb') as f:
            f.write(fh.read())

        logger.info(f"Fichier {file_name} téléchargé avec succès dans {temp_pdf_path}")
        return temp_pdf_path
    except HttpError as error:
        logger.error(f"Erreur HTTP lors du téléchargement de {file_name} (ID: {file_id}): {error}")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de {file_name} (ID: {file_id}): {e}", exc_info=True)
    # Nettoyer si le téléchargement a échoué après création du fichier
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)
    return None

# --- FIN NOUVELLES FONCTIONS GOOGLE DRIVE ---


# ----------------
# Extraction Texte (INCHANGÉ)
# ----------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un fichier PDF."""
    if not os.path.exists(pdf_path):
        logger.error(f"Le fichier temporaire {pdf_path} n'existe pas ou n'est pas accessible.")
        raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")

    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"Extraction de texte depuis {pdf_path} ({num_pages} pages)")
            for i, page in enumerate(reader.pages):
                page_txt = page.extract_text() or ""
                text.append(page_txt)
                if (i + 1) % 10 == 0 or (i + 1) == num_pages: # Log progress
                     logger.info(f"Page {i+1}/{num_pages} extraite de {os.path.basename(pdf_path)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du PDF {pdf_path}: {e}", exc_info=True)
        raise

    logger.info(f"Extraction de texte terminée pour {pdf_path}")
    return "\n".join(text)


def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les espaces multiples."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------
# Étape 1 : Segmentation (INCHANGÉ)
# ----------------------


def segment_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Découpe le texte en segments d'environ chunk_size mots avec chevauchement,
    en tentant de préserver les limites de phrases.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments = []
    current_segment = []
    current_size = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_size = len(sentence_words)

        if not sentence_words: # Skip empty sentences
            continue

        # Check if adding the sentence exceeds the chunk size significantly
        if current_size > 0 and current_size + sentence_size > chunk_size:
            # Add the current segment
            segments.append(" ".join(current_segment))

            # Determine overlap words based on the overlap size or sentence boundaries
            overlap_word_count = 0
            words_to_keep = []
            temp_segment_words = " ".join(current_segment).split()
            # Go back word by word until overlap size is reached or start is hit
            for i in range(len(temp_segment_words) - 1, -1, -1):
                 words_to_keep.insert(0, temp_segment_words[i])
                 overlap_word_count +=1
                 if overlap_word_count >= overlap:
                      # Check if we are in the middle of a sentence based on last word
                      if not temp_segment_words[i].endswith(('.', '!', '?')):
                          # Try to extend to the beginning of the sentence
                          potential_start = i - 1
                          while potential_start >= 0 and not temp_segment_words[potential_start].endswith(('.', '!', '?')):
                              words_to_keep.insert(0, temp_segment_words[potential_start])
                              potential_start -= 1
                      break # Stop once overlap is sufficient

            # Start new segment with overlap and the current sentence
            current_segment = words_to_keep + sentence_words
            current_size = len(current_segment)

        else:
            # Add sentence to the current segment
            current_segment.extend(sentence_words)
            current_size += sentence_size

    # Add the last segment if it exists
    if current_segment:
        segments.append(" ".join(current_segment))

    logger.info(f"Texte segmenté en {len(segments)} segments")
    # Log first few words of each segment for verification
    for i, seg in enumerate(segments[:3]):
        logger.debug(f"Segment {i+1} début: {' '.join(seg.split()[:10])}...")
    if len(segments) > 3:
         logger.debug("...")
         seg = segments[-1]
         logger.debug(f"Segment {len(segments)} début: {' '.join(seg.split()[:10])}...")

    return segments


# -------------------------------------
# Étape 2 & 3 : Analyse CoT + Codification (INCHANGÉ)
# -------------------------------------


def extract_json_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extrait du JSON valide depuis un texte potentiellement mélangé."""
    # Essayer d'abord avec une recherche stricte de tableau JSON
    json_array_match = re.search(r"^\s*(\[.*\])\s*$", text, re.DOTALL)
    if json_array_match:
        json_str = json_array_match.group(1)
        try:
            # Valider avec le module json
            parsed = json.loads(json_str)
            if isinstance(parsed, list): # Assurer que c'est une liste
                logger.debug("JSON array extrait avec regex stricte.")
                return parsed
        except json.JSONDecodeError:
             logger.warning(f"Regex a trouvé un pattern JSON array, mais le parsing a échoué: {json_str[:100]}...")
             pass # Continuer avec d'autres méthodes

    # Rechercher un pattern JSON (array) dans le texte, même s'il y a du texte autour
    json_pattern = r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\})*\s*\]" # Recherche un array d'objets
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                logger.debug("JSON array extrait avec regex large.")
                return parsed
        except json.JSONDecodeError:
            logger.warning(f"Regex a trouvé un pattern JSON array potentiel, mais le parsing a échoué: {json_str[:100]}...")
            pass

    # Essayer d'extraire entre les premiers '[' et les derniers ']' comme fallback
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        potential_json = text[start : end + 1]
        try:
            parsed = json.loads(potential_json)
            # Vérifier si c'est une liste d'objets (ou une liste vide)
            if isinstance(parsed, list):
                 is_list_of_dicts = all(isinstance(item, dict) for item in parsed)
                 if is_list_of_dicts or not parsed: # Accepter liste vide ou liste de dicts
                    logger.debug("JSON array extrait entre crochets [ ... ].")
                    return parsed
                 else:
                     logger.warning("JSON extrait entre crochets n'est pas une liste de dictionnaires.")
            else:
                 logger.warning("JSON extrait entre crochets n'est pas une liste.")

        except json.JSONDecodeError:
            logger.warning(f"Tentative d'extraction JSON entre crochets [ ... ] a échoué pour: {potential_json[:100]}...")
            pass

    logger.warning("Aucun JSON valide (array de dicts) n'a pu être extrait du texte.")
    return None


def analyze_and_code_segment(
    client: Groq, model: str, segment: str, max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Effectue une analyse CoT puis propose jusqu'à 3 codes (JSON).
    Inclut un mécanisme de réessai en cas d'échec.
    """
    # Construction du prompt avec exemples few-shot et instructions pour extraits longs
    prompt = (
        "Tu es un sociologue du numérique spécialisé en analyse qualitative.\n\n"
        "Étape 1: Analyse ce segment d'entretien en pensant à voix haute pour identifier les idées principales, "
        "y compris les idées implicites, les nuances ou l'ironie. (Ne pas inclure cette pensée dans la réponse finale JSON)\n\n"
        "Étape 2: Propose jusqu'à 3 codes sociologiques brefs (1-5 mots) accompagnés d'un extrait pertinent. "
        "Chaque code doit refléter un concept sociologique significatif.\n\n"
        "IMPORTANT: Pour chaque code, tu dois sélectionner un extrait LONG (au moins 15-20 mots OU une phrase complète significative) "
        "qui illustre bien le concept. Évite absolument les extraits trop courts (moins de 10 mots). "
        "L'extrait doit inclure suffisamment de contexte pour être compréhensible seul.\n\n"
        "Voici des exemples de thèmes et codes appropriés avec des extraits LONGS:\n\n"
        "EXEMPLE 1:\n"
        "Nom du thème: Un usage stratégique et pédagogique de l'IA générative\n"
        "Définition: Ce thème désigne l'intégration réfléchie de l'IA générative, en particulier de ChatGPT, dans les pratiques "
        "académiques des étudiant·es comme outil d'aide à la rédaction, à la compréhension et à la structuration des idées. "
        "L'usage est perçu comme complémentaire au travail intellectuel, mobilisé de manière ciblée pour dépasser des obstacles.\n"
        "Exemple verbatim: « le raisonnement que je vais présenter dans un travail, je vais plutôt le faire moi-même et j'ai plutôt "
        "utilisé ChatGpt pour m'aider pour des choses externes. Genre des biographies, des lectures, tout ça. Parce que je trouve "
        "que le raisonnement il n'est pas assez fort. »\n"
        'Codes possibles: [{ "code": "Complémentarité cognitive", "excerpt": "le raisonnement que je vais présenter dans un travail, je vais plutôt le faire moi-même et j\'ai plutôt utilisé ChatGpt pour m\'aider pour des choses externes" }, '
        '{ "code": "Usage ciblé et délimité", "excerpt": "utilisé ChatGpt pour m\'aider pour des choses externes. Genre des biographies, des lectures, tout ça. Parce que je trouve que le raisonnement il n\'est pas assez fort" }]\n\n'
        "EXEMPLE 2:\n"
        "Nom du thème: Un rapport critique à la légitimité académique menant à des tensions éthiques\n"
        "Définition: Ce thème renvoie à la manière dont les étudiant·es naviguent dans un espace institutionnel marqué par l'incertitude "
        "normative et le flou des règles entourant l'usage de l'IA générative. Entre méfiance vis-à-vis du plagiat et conscience des risques "
        "de transgression, les individus développent des stratégies d'auto-régulation.\n"
        "Exemple verbatim: « ça me brise le cœur de me dire, « bon ce mot est trop beau, il faut que je le mette quelque chose de plus humain, quoi. » »\n"
        'Codes possibles: [{ "code": "Anxiété normative", "excerpt": "ça me brise le cœur de me dire, « bon ce mot est trop beau, il faut que je le mette quelque chose de plus humain, quoi. »" }, '
        '{ "code": "Camouflage des traces IA", "excerpt": "ce mot est trop beau, il faut que je mette quelque chose de plus humain, quoi. » Pour que ça fasse plus comme si c\'était moi qui l\'avais écrit" }]\n\n'
        "EXEMPLE 3:\n"
        "Nom du thème: Une transformation du rapport au savoir et à l'écriture\n"
        "Définition: Ce thème interroge les effets de l'IA générative sur les manières de concevoir, produire et s'approprier le savoir dans "
        "le cadre universitaire. L'écriture n'est plus seulement une performance solitaire mais devient un processus dialogique, co-construit "
        "avec la machine.\n"
        "Exemple verbatim: « ça change les standards. Avant, on devait tout faire nous-mêmes, du coup au bout d'un moment t'as fait un travail "
        "pendant 3 semaines, au bout d'un moment t'as envie de le donner, même si c'est pas parfait, les mots utilisés sont pas parfaits, etc. "
        "Et là, vu qu'on a accès à un contenu de qualité relativement vite, je pense qu'il y a des standards où je me dis que ce n'est pas "
        "exactement ce que je veux. »\n"
        'Codes possibles: [{ "code": "Évolution des standards", "excerpt": "ça change les standards. Avant, on devait tout faire nous-mêmes, du coup au bout d\'un moment t\'as fait un travail pendant 3 semaines" }, '
        '{ "code": "Exigence accrue", "excerpt": "ça rend le truc plus exigeant parce qu\'on a accès à un contenu de qualité relativement vite" }, '
        '{ "code": "Temporalité transformée", "excerpt": "on a accès à un contenu de qualité relativement vite, je pense qu\'il y a des standards où je me dis que ce n\'est pas exactement ce que je veux" }]\n\n'
        f"Segment à analyser:\n'''\n{segment}\n'''\n\n"
        "Réponds UNIQUEMENT en format JSON avec un tableau (liste) contenant les codes et extraits, suivant ce modèle précis:\n"
        '[{"code": "concept sociologique", "excerpt": "extrait pertinent d\'au moins 15-20 mots ou une phrase complète"}, ...]\n'
        "Ne fournis AUCUN texte avant ou après le tableau JSON."
    )

    messages = [{"role": "user", "content": prompt}]
    segment_id_for_log = f"Segment starting with: '{segment[:50]}...'" # For logging

    for attempt in range(max_retries + 1):
        logger.info(f"Analyse CoT - Tentative {attempt+1}/{max_retries+1} pour {segment_id_for_log}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024, # Augmenté pour JSON + potentielle pensée CoT interne
                stream=False,
            )
            # Utiliser .strip() pour enlever les espaces/lignes vides avant/après
            raw_response_text = resp.choices[0].message.content.strip()
            logger.debug(f"Réponse brute reçue (tentative {attempt+1}):\n{raw_response_text}\n---")


            # Tentative d'extraction du JSON
            codes = extract_json_from_text(raw_response_text)

            if codes is not None: # Accepte une liste vide si le LLM n'a rien trouvé
                 # Vérification de la longueur des extraits
                valid_codes = []
                issues_found = False
                if not codes: # Si la liste est vide
                    logger.info(f"Analyse CoT: Le modèle n'a retourné aucun code pour {segment_id_for_log}. Accepté.")
                    return []

                for i, code in enumerate(codes):
                    if not isinstance(code, dict):
                         logger.warning(f"Item {i} dans la liste JSON n'est pas un dictionnaire: {code}")
                         issues_found = True
                         continue # Ignorer cet item
                    code_text = code.get("code")
                    excerpt = code.get("excerpt")

                    if not code_text or not excerpt:
                        logger.warning(f"Code JSON incomplet trouvé (manque 'code' ou 'excerpt'): {code}")
                        issues_found = True
                        continue # Ignorer ce code

                    excerpt_words = excerpt.split()
                    if len(excerpt_words) >= 10:
                        valid_codes.append(code)
                    else:
                        logger.warning(f"Extrait trop court détecté (< 10 mots) pour le code '{code_text}': '{excerpt}'. Tentative de correction échouée ou non tentée par le modèle.")
                        # On ne tente pas de l'étendre ici, on le signale juste. Le Juge s'en chargera.
                        # Ou on pourrait décider de l'ignorer ici, mais le garder pour le Juge est peut-être mieux.
                        valid_codes.append(code) # Garder même si court, pour que le Juge le voie.
                        issues_found = True # Marquer qu'il y a eu un problème potentiel

                # Si on a extrait quelque chose et qu'il n'y a pas eu d'erreur majeure de format
                # On retourne les codes (même ceux potentiellement courts, le Juge filtrera)
                if valid_codes or not issues_found: # Si on a des codes valides OU si on a extrait une liste vide sans erreur
                     logger.info(f"Analyse CoT réussie (tentative {attempt+1}). {len(valid_codes)} codes extraits pour {segment_id_for_log}.")
                     return valid_codes
                # Si on a eu des problèmes (codes incomplets, etc.) mais qu'on a quand même extrait *quelque chose*,
                # on passe à la tentative suivante si possible.
                elif attempt < max_retries:
                     logger.warning(f"Problèmes détectés dans le JSON (codes incomplets ou extraits courts signalés). Tentative {attempt+2} demandée.")
                     # Préparer le message pour la nouvelle tentative
                     fix_prompt = (
                         "Ta réponse précédente n'était pas entièrement satisfaisante (peut-être JSON invalide, incomplet, ou extraits trop courts signalés). "
                         "Assure-toi de fournir UNIQUEMENT un tableau JSON valide contenant des objets avec les clés 'code' et 'excerpt'. "
                         "Les EXTRAITS doivent être LONGS (minimum 15-20 mots ou une phrase complète). "
                         "Format attendu: "
                         '[{"code": "concept sociologique", "excerpt": "extrait long et pertinent"}, ...]\n'
                         "N'inclus aucune explication, juste le JSON."
                     )
                     messages = [
                         {"role": "user", "content": prompt}, # Rappeler le contexte initial
                         {"role": "assistant", "content": raw_response_text}, # Montrer la réponse précédente
                         {"role": "user", "content": fix_prompt}, # Demander la correction
                     ]
                # else: on a eu des soucis et c'était la dernière tentative

            # Si extract_json_from_text retourne None OU si on arrive ici après échec de validation
            elif attempt < max_retries:
                logger.warning(f"Échec de l'extraction JSON ou JSON invalide (tentative {attempt+1}). Tentative {attempt+2} demandée.")
                fix_prompt = (
                    "Ta réponse précédente n'a pas pu être interprétée comme un JSON valide ou était incorrecte. "
                    "Réponds avec un tableau JSON valide au format: "
                    '[{"code": "concept sociologique", "excerpt": "extrait long et pertinent"}, ...]\n'
                    "N'inclus aucune explication ou texte en dehors du JSON. Assure-toi que les extraits soient longs (15-20 mots min)."
                )
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": raw_response_text},
                    {"role": "user", "content": fix_prompt},
                ]
            # else: c'était la dernière tentative et extract_json a retourné None

        except Exception as e:
            logger.error(
                f"Erreur inattendue lors de l'analyse du segment (tentative {attempt+1}) pour {segment_id_for_log}: {e}", exc_info=True
            )
            if attempt == max_retries:
                logger.error(f"Échec final de l'analyse pour {segment_id_for_log} après plusieurs tentatives.")
                return [] # Retourner une liste vide en cas d'échec final

    # Si on sort de la boucle sans succès
    logger.error(
        f"Impossible d'obtenir un JSON valide après {max_retries + 1} tentatives pour {segment_id_for_log}. Retour d'une liste vide."
    )
    return []


# -----------------------------
# Étape 4 : Validation (juge) (INCHANGÉ)
# -----------------------------


def judge_codes(
    client: Groq, model: str, segment: str, codes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Valide ou corrige chaque code proposé par l'agent.
    Retourne une liste de codes validés/améliorés.
    S'assure que les extraits sont suffisamment longs (au moins 10 mots).
    """
    if not codes:
        return []

    segment_id_for_log = f"Segment starting with: '{segment[:50]}...'" # For logging
    logger.info(f"Validation Juge: Démarrage pour {len(codes)} codes de {segment_id_for_log}.")

    # Convertir les codes reçus en JSON pour le prompt
    try:
        codes_json = json.dumps(codes, ensure_ascii=False, indent=2)
    except TypeError:
        logger.error("Erreur lors de la conversion des codes en JSON pour le prompt du juge.")
        return codes # Retourner les codes originaux en cas d'erreur de sérialisation

    # Construction du prompt avec exemples few-shot pour la validation et insistance sur extraits longs
    prompt = (
        "Tu es un expert en sociologie évaluant des codes d'analyse qualitative.\n\n"
        f"Voici un segment d'entretien:\n'''\n{segment}\n'''\n\n"
        f"Et voici les codes proposés:\n{codes_json}\n\n"
        "CRITÈRE IMPORTANT: Pour chaque code :\n"
        "1. Évalue la pertinence sociologique du 'code'.\n"
        "2. Vérifie que l'extrait ('excerpt') est LONG (minimum 10 mots, idéalement 15-20 ou une phrase complète) et illustre bien le code.\n"
        "3. Si un extrait est trop court ou mal choisi, tu DOIS l'étendre ou le corriger en cherchant dans le segment original pour fournir un meilleur contexte.\n"
        "4. Si un code n'est pas pertinent ou ne peut être justifié par un extrait approprié, supprime-le.\n\n"
        "Voici des exemples de bons codes sociologiques et de leurs extraits pertinents LONGS pour t'inspirer:\n\n"
        # ... (les exemples restent les mêmes) ...
        "EXEMPLE 1 - Concernant l'usage de l'IA dans les travaux académiques:\n"
        '- Code: "Complémentarité cognitive" avec extrait: "le raisonnement que je vais présenter dans un travail, je vais plutôt le faire moi-même et j\'ai plutôt utilisé ChatGpt pour m\'aider pour des choses externes"\n'
        '- Code: "Usage ciblé et délimité" avec extrait: "utilisé ChatGpt pour m\'aider pour des choses externes. Genre des biographies, des lectures, tout ça. Parce que je trouve que le raisonnement il n\'est pas assez fort"\n\n'
        "EXEMPLE 2 - Concernant les tensions éthiques liées à l'IA:\n"
        '- Code: "Anxiété normative" avec extrait: "ça me brise le cœur de me dire, « bon ce mot est trop beau, il faut que je le mette quelque chose de plus humain, quoi. » Pour que ça fasse plus comme si c\'était moi qui l\'avais écrit"\n'
        '- Code: "Camouflage des traces IA" avec extrait: "ce mot est trop beau, il faut que je mette quelque chose de plus humain, quoi. » Pour que ça fasse plus comme si c\'était moi qui l\'avais écrit et pas ChatGPT"\n\n'
        "EXEMPLE 3 - Concernant la transformation du rapport à l'écriture:\n"
        '- Code: "Évolution des standards" avec extrait: "ça change les standards. Avant, on devait tout faire nous-mêmes, du coup au bout d\'un moment t\'as fait un travail pendant 3 semaines"\n'
        '- Code: "Exigence accrue" avec extrait: "ça rend le truc plus exigeant parce qu\'on a accès à un contenu de qualité relativement vite, je pense qu\'il y a des standards plus élevés"\n'
        '- Code: "Temporalité transformée" avec extrait: "on a accès à un contenu de qualité relativement vite, je pense qu\'il y a des standards où je me dis que ce n\'est pas exactement ce que je veux"\n\n'
        "Retourne UNIQUEMENT un tableau JSON avec les codes finaux (validés, améliorés ou supprimés), en suivant ce format:\n"
        '[{"code": "code validé ou amélioré", "excerpt": "extrait pertinent LONG (min 10 mots)"}, ...]\n\n'
        "Assure-toi que chaque objet dans le tableau a bien les clés 'code' et 'excerpt' et que les extraits respectent la longueur minimale. Ne retourne RIEN d'autre que le JSON."
    )

    max_retries_judge = 1 # Moins de tentatives pour le juge, il doit être plus direct
    for attempt in range(max_retries_judge + 1):
        logger.info(f"Validation Juge - Tentative {attempt+1}/{max_retries_judge+1} pour {segment_id_for_log}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1536,  # Augmenté pour permettre des extraits longs et potentiellement plus de codes
                stream=False,
            )
            content = resp.choices[0].message.content.strip()
            logger.debug(f"Réponse brute reçue du Juge (tentative {attempt+1}):\n{content}\n---")

            # Extraction du JSON de la réponse du juge
            validated_codes = extract_json_from_text(content)

            if validated_codes is not None: # Accepte une liste vide si le juge supprime tout
                final_codes = []
                issues_found = False
                if not validated_codes:
                     logger.info(f"Validation Juge: Le juge a supprimé tous les codes pour {segment_id_for_log}.")
                     return []

                for i, code in enumerate(validated_codes):
                     if not isinstance(code, dict):
                         logger.warning(f"Item {i} du Juge n'est pas un dictionnaire: {code}")
                         issues_found = True
                         continue
                     code_text = code.get("code")
                     excerpt = code.get("excerpt")

                     if not code_text or not excerpt:
                         logger.warning(f"Code du Juge incomplet trouvé (manque 'code' ou 'excerpt'): {code}")
                         issues_found = True
                         continue

                     excerpt_words = excerpt.split()
                     if len(excerpt_words) >= 10:
                         final_codes.append(code)
                     else:
                         # Le juge n'a pas respecté la consigne de longueur minimale !
                         logger.warning(f"Juge a retourné un extrait trop court (< 10 mots) pour code '{code_text}': '{excerpt}'. On essaie de l'étendre manuellement.")
                         issues_found = True
                         # Tentative manuelle d'extension si l'extrait court est dans le segment
                         extended_excerpt = None
                         if excerpt in segment:
                              # Essayer de trouver la phrase complète contenant l'extrait court
                              # Split par ponctuations de fin de phrase, en conservant les délimiteurs
                              sentences = re.split(r'([.!?])\s*', segment)
                              # Reconstruire les phrases avec leur ponctuation
                              full_sentences = []
                              if sentences:
                                   current_sentence = sentences[0]
                                   for j in range(1, len(sentences), 2):
                                        if j+1 < len(sentences):
                                             current_sentence += (sentences[j] or '') + (sentences[j+1] or '')
                                             full_sentences.append(current_sentence.strip())
                                             current_sentence = "" # Start new sentence conceptually
                                        else: # Handle last part if no trailing punctuation
                                             current_sentence += (sentences[j] or '')
                                             full_sentences.append(current_sentence.strip())
                                   # Add the last part if not empty and not added
                                   if current_sentence and current_sentence.strip() and current_sentence.strip() not in full_sentences:
                                       full_sentences.append(current_sentence.strip())


                              for sentence in full_sentences:
                                   if excerpt in sentence and len(sentence.split()) >= 10:
                                        extended_excerpt = sentence
                                        logger.info(f"Extension manuelle réussie pour l'extrait court: '{extended_excerpt}'")
                                        break
                         if extended_excerpt:
                             code["excerpt"] = extended_excerpt # Mettre à jour l'extrait dans le dictionnaire
                             final_codes.append(code)
                         else:
                             logger.warning(f"Impossible d'étendre manuellement l'extrait court fourni par le juge pour '{code_text}'. Ce code sera ignoré.")
                             # Ne pas ajouter ce code si l'extension échoue

                # Si on a extrait quelque chose et qu'il n'y a pas eu d'erreur majeure de format, OU si on a une liste vide valide
                if final_codes or not issues_found:
                     logger.info(f"Validation Juge réussie (tentative {attempt+1}). {len(final_codes)} codes validés/améliorés pour {segment_id_for_log}.")
                     return final_codes
                # Si on a eu des problèmes et qu'on a encore des tentatives
                elif attempt < max_retries_judge:
                     logger.warning(f"Problèmes détectés dans la réponse du Juge (codes incomplets, extraits trop courts non corrigés). Nouvelle tentative demandée.")
                     # Le prompt de retry pourrait être similaire à celui de l'analyseur
                     fix_prompt = (
                        "Ta réponse précédente contenait des erreurs (JSON invalide, codes incomplets, ou extraits encore trop courts).\n"
                        "Corrige ta réponse pour fournir UNIQUEMENT un tableau JSON valide.\n"
                        "Chaque élément doit avoir 'code' et 'excerpt'.\n"
                        "Chaque 'excerpt' DOIT avoir au moins 10 mots (idéalement 15-20 ou phrase complète).\n"
                        'Format: [{"code": "...", "excerpt": "extrait LONG"}, ...]'
                     )
                     # Note: On ne renvoie pas la réponse erronée au juge pour éviter confusion, on réitère juste la demande.
                     # Ceci est différent de l'approche pour l'analyseur CoT.
                     messages = [{"role": "user", "content": prompt}] # On redonne le prompt initial complet
                # else: dernière tentative échouée

            # Si extract_json_from_text retourne None OU si on arrive ici après échec de validation/correction
            elif attempt < max_retries_judge:
                logger.warning(f"Échec de l'extraction JSON de la réponse du Juge (tentative {attempt+1}). Nouvelle tentative demandée.")
                messages = [{"role": "user", "content": prompt}] # On redonne le prompt initial complet

        except Exception as e:
            logger.error(f"Erreur inattendue lors de la validation Juge (tentative {attempt+1}) pour {segment_id_for_log}: {e}", exc_info=True)
            if attempt == max_retries_judge:
                 logger.error(f"Échec final de la validation Juge pour {segment_id_for_log}. Retour des codes originaux.")
                 return codes # Retourner les codes originaux en cas d'échec grave du juge

    # Si on sort de la boucle sans succès après les tentatives
    logger.error(f"Impossible d'obtenir une réponse valide du Juge après {max_retries_judge + 1} tentatives pour {segment_id_for_log}. Retour des codes originaux.")
    return codes # Fallback: retourner les codes originaux si le juge échoue complètement


# --------------------------------
# Étape 5 : Clustering des codes (LIMITÉ À 10 THÈMES MAX) (INCHANGÉ)
# --------------------------------


def cluster_codes_limited(
    all_codes: List[Dict[str, Any]],
    max_clusters: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    """
    Regroupe les codes par similarité sémantique en limitant à max_clusters thèmes.
    Retourne un dict cluster_id -> list de codes et un mapping code -> cluster_id.
    """
    if not all_codes:
        logger.warning("Clustering: Reçu une liste de codes vide. Retour de résultats vides.")
        return {}, {}

    # Extraire uniquement les textes des codes
    # Utiliser un set pour dédoublonner les textes de code avant l'embedding
    unique_code_texts = list(set(code["code"] for code in all_codes))
    code_text_to_index = {text: i for i, text in enumerate(unique_code_texts)}
    logger.info(f"Clustering: {len(all_codes)} codes reçus, {len(unique_code_texts)} textes de code uniques à traiter.")


    if not unique_code_texts:
        logger.warning("Clustering: Aucun texte de code unique trouvé après filtrage. Retour de résultats vides.")
        return {}, {}

    try:
        # Générer les embeddings pour les codes uniques
        logger.info(f"Génération des embeddings pour {len(unique_code_texts)} codes uniques avec '{embedding_model}'...")
        embedder = SentenceTransformer(embedding_model)
        embeddings = embedder.encode(unique_code_texts, show_progress_bar=True)
        logger.info("Embeddings générés.")

        # Déterminer le nombre de clusters à utiliser
        # On ne peut pas avoir plus de clusters que de points de données (codes uniques)
        n_clusters_effective = min(max_clusters, len(unique_code_texts))
        if n_clusters_effective < 1:
             logger.error("Clustering: Le nombre effectif de clusters est inférieur à 1. Impossible de clusteriser.")
             return {}, {} # Ou gérer ce cas différemment

        logger.info(f"Clustering hiérarchique demandé avec n_clusters = {n_clusters_effective} (max_clusters={max_clusters}, unique_codes={len(unique_code_texts)})")

        # Clustering hiérarchique agglomératif
        # 'ward' minimise la variance au sein de chaque cluster
        # 'cosine' est souvent bon pour les embeddings textuels, mais AgglomerativeClustering avec 'ward' utilise 'euclidean' par défaut.
        # Testons avec 'ward' et 'euclidean' (défaut). Si les résultats sont mauvais, 'average' et 'cosine' pourraient être une alternative.
        clustering = AgglomerativeClustering(n_clusters=n_clusters_effective, metric='euclidean', linkage='ward')
        # Les labels correspondent à l'ordre des unique_code_texts
        labels = clustering.fit_predict(embeddings)
        logger.info("Clustering terminé.")

        # Créer un mapping du texte de code unique vers son label de cluster
        code_text_to_cluster_label = {text: int(labels[i]) for i, text in enumerate(unique_code_texts)}

        # Organiser tous les codes originaux (y compris les doublons) en clusters
        clusters: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_clusters_effective)}
        code_to_cluster_map: Dict[str, int] = {} # Pour le retour final (mapping code -> cluster id)

        processed_code_texts_in_map = set() # Pour éviter les doublons dans code_to_cluster_map si le même code apparaît plusieurs fois

        for code_obj in all_codes:
            code_text = code_obj["code"]
            # Trouver le label de cluster pour ce texte de code
            cluster_label = code_text_to_cluster_label.get(code_text)
            if cluster_label is not None:
                clusters[cluster_label].append(code_obj)
                # Ajouter au mapping de retour seulement si pas déjà fait pour ce texte de code
                if code_text not in processed_code_texts_in_map:
                    code_to_cluster_map[code_text] = cluster_label
                    processed_code_texts_in_map.add(code_text)
            else:
                # Ne devrait pas arriver si tous les codes sont dans unique_code_texts
                 logger.warning(f"Code '{code_text}' non trouvé dans le mapping de clustering après traitement. Ignoré.")


        # Filtrer les clusters vides (peu probable avec Agglomerative Clustering mais par sécurité)
        final_clusters = {k: v for k, v in clusters.items() if v}
        num_final_clusters = len(final_clusters)

        logger.info(
            f"Clustering terminé: {num_final_clusters} thèmes (clusters non vides) identifiés (limité à {max_clusters})."
        )
        if num_final_clusters < n_clusters_effective:
             logger.warning(f"Moins de clusters non vides ({num_final_clusters}) que demandé ({n_clusters_effective}).")


        return final_clusters, code_to_cluster_map

    except ImportError as ie:
         logger.error(f"Erreur d'importation liée au clustering (SentenceTransformer ou Scikit-learn): {ie}. Assurez-vous que les bibliothèques sont installées.")
         raise
    except Exception as e:
        logger.error(f"Erreur grave lors du clustering: {e}", exc_info=True)
        # Fallback très simple: créer max_clusters groupes arbitraires si tout échoue
        logger.warning("Tentative de fallback de clustering simple.")
        simple_clusters = {}
        simple_code_to_cluster = {}
        if not all_codes: return {}, {}

        # Regroupement simplifié pour fallback
        num_codes = len(all_codes)
        codes_per_cluster = max(1, (num_codes + max_clusters - 1) // max_clusters) # Arrondi supérieur

        current_code_index = 0
        fallback_cluster_id = 0
        processed_codes_fallback = set() # Pour le map

        while current_code_index < num_codes and fallback_cluster_id < max_clusters:
            cluster_codes = all_codes[current_code_index : current_code_index + codes_per_cluster]
            if cluster_codes:
                simple_clusters[fallback_cluster_id] = cluster_codes
                for code in cluster_codes:
                    code_text = code['code']
                    if code_text not in processed_codes_fallback:
                         simple_code_to_cluster[code_text] = fallback_cluster_id
                         processed_codes_fallback.add(code_text)

                fallback_cluster_id += 1
                current_code_index += len(cluster_codes)
            else:
                 break # Should not happen if logic is right

        logger.warning(f"Fallback clustering créé avec {len(simple_clusters)} groupes.")
        return simple_clusters, simple_code_to_cluster


# -------------------------------
# Étape 6 : Labelling des thèmes (INCHANGÉ)
# -------------------------------


def label_themes(
    client: Groq, model: str, clusters: Dict[int, List[Dict[str, Any]]]
) -> Dict[int, str]:
    """
    Pour chaque cluster, demande au LLM un titre de thème sociologique concis.
    """
    theme_labels: Dict[int, str] = {}
    if not clusters:
        logger.warning("Labelling: Reçu un dictionnaire de clusters vide.")
        return {}

    logger.info(f"Démarrage du labelling pour {len(clusters)} clusters.")

    # Exemple few-shot pour le labelling
    examples = [
        "Usage stratégique IA", # Raccourci pour le prompt
        "Tensions éthiques IA",
        "Transformation savoir/écriture",
    ]

    # Limiter le nombre d'exemples dans le prompt pour économiser les tokens si beaucoup de clusters
    MAX_ITEMS_PER_CLUSTER_FOR_PROMPT = 5

    for cluster_id, codes in clusters.items():
        if not codes:
            logger.warning(f"Cluster {cluster_id} est vide, impossible de le labelliser.")
            continue

        # Extraire le texte des codes et quelques excerpts pour le contexte
        # Prendre un échantillon si le cluster est très grand
        sample_codes = codes[:MAX_ITEMS_PER_CLUSTER_FOR_PROMPT]
        # Créer une représentation textuelle simple pour le prompt
        code_items_repr = []
        for c in sample_codes:
             # Tronquer les extraits longs pour le prompt
             excerpt_preview = c.get('excerpt', '')
             if len(excerpt_preview.split()) > 25:
                 excerpt_preview = ' '.join(excerpt_preview.split()[:25]) + '...'
             code_items_repr.append(f"- Code: \"{c.get('code', 'N/A')}\" (Ex: \"{excerpt_preview}\")")

        codes_representation = "\n".join(code_items_repr)
        if len(codes) > MAX_ITEMS_PER_CLUSTER_FOR_PROMPT:
             codes_representation += f"\n... (et {len(codes) - MAX_ITEMS_PER_CLUSTER_FOR_PROMPT} autres codes)"

        prompt = (
            "Tu es un sociologue synthétisant des résultats d'analyse qualitative.\n\n"
            f"Voici un groupe de codes apparentés issus d'un cluster (Thème ID: {cluster_id}):\n{codes_representation}\n\n"
            "Propose un titre de thème sociologique TRÈS CONCIS (2-5 mots maximum) qui capture l'essence commune de ces codes.\n"
            "Le titre doit être pertinent par rapport aux codes et extraits fournis.\n\n"
            "Voici des exemples de BONS titres de thèmes CONCIS:\n"
            f'- "{examples[0]}"\n'
            f'- "{examples[1]}"\n'
            f'- "{examples[2]}"\n\n'
            'Réponds UNIQUEMENT en format JSON: {"theme": "Ton titre concis ici"}\n'
            "Ne fournis AUCUN autre texte."
        )

        theme_name = f"Thème non labellisé {cluster_id+1}" # Fallback name
        try:
            logger.info(f"Labelling du cluster {cluster_id} ({len(codes)} codes)...")
            resp = client.chat.completions.create(
                model=model, # Utiliser le même modèle que pour l'analyse CoT ou un modèle rapide ? Testons avec CoT.
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=128, # Devrait être suffisant pour {"theme": "..."}
                stream=False,
                # response_format={"type": "json_object"}, # Si le modèle supporte le mode JSON strict
            )
            content = resp.choices[0].message.content.strip()
            logger.debug(f"Réponse brute reçue pour labelling cluster {cluster_id}:\n{content}\n---")

            # Extraction du JSON - plus robuste
            match = re.search(r'\{\s*"theme"\s*:\s*"([^"]+)"\s*\}', content)
            if match:
                theme_name = match.group(1).strip()
                # Vérifier la concision (optionnel mais recommandé par le prompt)
                if len(theme_name.split()) > 6: # Permettre un mot de plus que demandé
                    logger.warning(f"Label pour cluster {cluster_id} est long: '{theme_name}'. Raccourcissement suggéré.")
                    # On pourrait essayer de le raccourcir ici ou juste l'accepter. Acceptons-le pour l'instant.
            else:
                # Tentative d'extraction JSON complète si regex échoue
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "theme" in data and isinstance(data["theme"], str):
                        theme_name = data["theme"].strip()
                        logger.info(f"Label extrait via parsing JSON pour cluster {cluster_id}: '{theme_name}'")
                        if len(theme_name.split()) > 6:
                           logger.warning(f"Label pour cluster {cluster_id} est long: '{theme_name}'.")
                    else:
                        logger.warning(f"Réponse JSON pour cluster {cluster_id} n'a pas le format attendu {{'theme': '...'}}. Contenu: {content}")
                        # Essayer de prendre le contenu brut comme label si c'est une chaîne courte? Risqué. Utiliser fallback.
                except json.JSONDecodeError:
                    logger.warning(f"Réponse pour cluster {cluster_id} n'est pas un JSON valide ou n'a pas pu être extraite. Contenu: {content}")
                    # Essayer de récupérer le texte s'il est court et ressemble à un label?
                    if len(content.split()) <= 6 and not content.startswith("{") and not content.startswith("["):
                         theme_name = content
                         logger.info(f"Utilisation du texte brut comme label (fallback) pour cluster {cluster_id}: '{theme_name}'")
                    # Sinon, le fallback 'Thème non labellisé...' sera utilisé.


        except Exception as e:
            logger.error(f"Erreur lors du labelling du cluster {cluster_id}: {e}", exc_info=True)
            # Le fallback 'Thème non labellisé...' sera utilisé

        logger.info(f"Cluster {cluster_id} labellisé comme: '{theme_name}'")
        theme_labels[cluster_id] = theme_name

    logger.info("Labelling des thèmes terminé.")
    return theme_labels


# -----------------------------------
# Étape 7 : Second-level clustering (pour regrouper si > 10 thèmes) (INCHANGÉ)
# -----------------------------------


def meta_cluster_themes(
    client: Groq,
    model: str,
    clusters: Dict[int, List[Dict[str, Any]]],
    theme_map: Dict[int, str],
    target_count: int = 10,
) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Si nécessaire, regroupe les thèmes en méta-thèmes pour atteindre le nombre cible.
    Retourne un dict meta_theme_id -> nom du méta-thème et un mapping original_cluster_id -> meta_theme_id.
    """
    num_initial_themes = len(clusters)
    if num_initial_themes <= target_count:
        logger.info(f"Méta-clustering non requis ({num_initial_themes} thèmes <= cible de {target_count}).")
        # Retourner les thèmes comme leurs propres méta-thèmes
        # Le mapping meta_theme_id -> nom
        meta_theme_names = {cluster_id: theme_map.get(cluster_id, f"Thème {cluster_id}")
                           for cluster_id in clusters.keys()}
        # Le mapping original_cluster_id -> meta_theme_id (qui est lui-même ici)
        theme_to_meta_assignment = {cluster_id: cluster_id for cluster_id in clusters.keys()}
        return meta_theme_names, theme_to_meta_assignment

    logger.info(f"Méta-clustering requis: {num_initial_themes} thèmes > cible de {target_count}. Démarrage du regroupement.")

    # Pour chaque thème, créer une description synthétique
    theme_summaries = {}
    MAX_CODES_PER_THEME_FOR_PROMPT = 5 # Limiter pour le prompt de méta-clustering
    for cluster_id, codes in clusters.items():
        theme_name = theme_map.get(cluster_id, f"Thème {cluster_id}")
        # Prendre les codes les plus fréquents ou les premiers N codes comme représentatifs
        representative_codes = [c['code'] for c in codes[:MAX_CODES_PER_THEME_FOR_PROMPT]]
        codes_str = ", ".join(f'"{code}"' for code in representative_codes)
        if len(codes) > MAX_CODES_PER_THEME_FOR_PROMPT:
            codes_str += ", ..."
        # Utiliser l'ID original du cluster comme clé dans le prompt
        theme_summaries[str(cluster_id)] = ( # Utiliser str(cluster_id) car JSON keys must be strings
            f"Thème ID {cluster_id} ('{theme_name}'): contient des codes comme [{codes_str}]"
        )

    # Demander au LLM de regrouper les thèmes avec exemples few-shot
    # Convertir les résumés en une chaîne formatée pour le prompt
    themes_description_str = "\n".join(f"- {desc}" for desc in theme_summaries.values())

    prompt = (
        "Tu es un expert en sociologie chargé de regrouper des thèmes d'analyse qualitative similaires.\n\n"
        f"Voici {num_initial_themes} thèmes identifiés (avec leurs ID et quelques codes exemples):\n{themes_description_str}\n\n"
        f"Regroupe ces thèmes en EXACTEMENT {target_count} méta-thèmes maximum. Chaque thème original (identifié par son ID) doit être assigné "
        f"à un seul méta-thème plus large.\n\n"
        "Pour chaque méta-thème créé, propose un nom concis (2-6 mots).\n\n"
        "Voici un exemple de regroupement thématique en sociologie du numérique:\n\n"
        '- Méta-thème "Pratiques numériques et apprentissage" pourrait inclure les thèmes originaux avec ID 3 ("Usage pédagogique des outils"), 7 ("Appropriation des technologies") et 12 ("Stratégies d\'adaptation numérique").\n'
        '- Méta-thème "Tensions éthiques et normatives" pourrait inclure les thèmes ID 1 ("Anxiété face aux règles"), 5 ("Légitimité académique contestée") et 9 ("Stratégies de contournement").\n'
        # ... (peut-être ajouter un 3ème exemple) ...
        "\nRéponds UNIQUEMENT en format JSON avec le format suivant:\n"
        '{\n'
        '  "meta_themes": {\n'
        '    "0": "Nom du Méta-thème 1",\n'
        '    "1": "Nom du Méta-thème 2",\n'
        f'    ...\n'
        f'    "{target_count-1}": "Nom du Méta-thème {target_count}"\n' # Assurer qu'il y a target_count clés
        '  },\n'
        '  "assignments": {\n'
        '    "original_theme_id_1": meta_theme_id_for_1, \n' # ex: "3": 0
        '    "original_theme_id_2": meta_theme_id_for_2, \n' # ex: "7": 0
        '    "original_theme_id_5": meta_theme_id_for_5, \n' # ex: "1": 1
        '    ...\n'
        '    "last_original_theme_id": its_meta_theme_id \n' # ex: "12": 0
        '  }\n'
        '}\n\n'
        "Où 'meta_themes' est un dictionnaire des méta-thèmes numérotés de 0 à {target_count-1}. "
        "Et 'assignments' indique pour chaque ID de thème original (clé en chaîne de caractères) l'ID numérique (0 à {target_count-1}) du méta-thème auquel il est assigné.\n"
        "Assure-toi que CHAQUE thème original listé ci-dessus a une entrée dans 'assignments' et que les ID de méta-thème assignés sont valides (entre 0 et {target_count-1})."
    )

    final_meta_theme_names: Dict[int, str] = {}
    final_theme_to_meta_assignment: Dict[int, int] = {}

    try:
        logger.info("Envoi de la requête de méta-clustering au LLM...")
        resp = client.chat.completions.create(
            model=model, # Utiliser le même modèle puissant
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Un peu plus de créativité pour le regroupement et nommage
            max_tokens=2048, # Augmenté car le prompt est long et la réponse peut l'être aussi
            stream=False,
            # response_format={"type": "json_object"}, # Si supporté
        )
        content = resp.choices[0].message.content.strip()
        logger.debug(f"Réponse brute reçue pour méta-clustering:\n{content}\n---")


        # Extraction du JSON de la réponse
        # Utiliser une regex pour extraire le bloc JSON principal, plus robuste aux commentaires
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                raw_meta_themes = data.get("meta_themes")
                raw_assignments = data.get("assignments")

                if isinstance(raw_meta_themes, dict) and isinstance(raw_assignments, dict):
                    # Valider et convertir les meta_themes
                    valid_meta_themes = {}
                    all_meta_ids = set()
                    for k, v in raw_meta_themes.items():
                        try:
                            meta_id = int(k)
                            if 0 <= meta_id < target_count and isinstance(v, str) and v.strip():
                                valid_meta_themes[meta_id] = v.strip()
                                all_meta_ids.add(meta_id)
                            else:
                                logger.warning(f"Méta-thème invalide dans la réponse: key={k}, value={v}. Ignoré.")
                        except ValueError:
                             logger.warning(f"Clé de méta-thème non entière: {k}. Ignoré.")

                    # S'assurer qu'on a le bon nombre de méta-thèmes, sinon ajouter des placeholders
                    if len(valid_meta_themes) < target_count:
                         logger.warning(f"Le LLM a fourni {len(valid_meta_themes)} méta-thèmes, mais {target_count} étaient attendus. Ajout de placeholders.")
                         for i in range(target_count):
                              if i not in valid_meta_themes:
                                   valid_meta_themes[i] = f"Méta-thème {i+1} (Généré)"
                                   all_meta_ids.add(i)

                    # Valider et convertir les assignments
                    valid_assignments = {}
                    assigned_original_themes = set()
                    original_theme_ids_str = set(theme_summaries.keys()) # Les ID des thèmes qu'on a envoyés (en str)

                    for k, v in raw_assignments.items():
                         if k not in original_theme_ids_str:
                              logger.warning(f"Assignment reçu pour un ID de thème original inconnu: {k}. Ignoré.")
                              continue
                         try:
                             original_id = int(k) # Convertir la clé en int pour le mapping final
                             meta_id = int(v)
                             if meta_id in all_meta_ids: # Vérifier si l'ID de méta-thème assigné est valide
                                 valid_assignments[original_id] = meta_id
                                 assigned_original_themes.add(k)
                             else:
                                 logger.warning(f"Assignment invalide: thème original {k} assigné au méta-thème inexistant {v}. Ignoré.")
                         except (ValueError, TypeError):
                             logger.warning(f"Assignment invalide (non entier): key={k}, value={v}. Ignoré.")

                    # Vérifier si tous les thèmes originaux ont été assignés
                    missing_assignments = original_theme_ids_str - assigned_original_themes
                    if missing_assignments:
                        logger.warning(f"Certains thèmes originaux n'ont pas été assignés par le LLM: {missing_assignments}. Tentative d'assignation au premier méta-thème (0).")
                        fallback_meta_id = 0 # Assigner au premier méta-thème par défaut
                        if fallback_meta_id not in valid_meta_themes: # S'assurer que le méta-thème 0 existe
                             # Si même 0 n'existe pas (très improbable), prendre le premier dispo
                             fallback_meta_id = min(valid_meta_themes.keys()) if valid_meta_themes else 0
                             if fallback_meta_id not in valid_meta_themes and fallback_meta_id == 0: # Créer le 0 s'il manque et qu'on en a besoin
                                  valid_meta_themes[0] = "Méta-thème 1 (Généré)"


                        for missing_id_str in missing_assignments:
                             try:
                                valid_assignments[int(missing_id_str)] = fallback_meta_id
                             except ValueError: pass # Ignorer si l'ID n'est pas un int (ne devrait pas arriver)


                    # Si tout semble OK
                    if valid_meta_themes and valid_assignments:
                         logger.info(f"Méta-clustering réussi via LLM. {len(valid_meta_themes)} méta-thèmes définis.")
                         final_meta_theme_names = valid_meta_themes
                         final_theme_to_meta_assignment = valid_assignments
                         return final_meta_theme_names, final_theme_to_meta_assignment
                    else:
                         logger.error("Méta-clustering: JSON reçu mais invalide ou incomplet après validation.")

                else:
                    logger.error("Méta-clustering: La structure JSON ('meta_themes' ou 'assignments') est incorrecte ou manquante.")

            except json.JSONDecodeError as jde:
                logger.error(f"Méta-clustering: Erreur de décodage JSON lors du traitement de la réponse: {jde}")
        else:
            logger.error("Méta-clustering: Aucun bloc JSON trouvé dans la réponse du LLM.")

    except Exception as e:
        logger.error(f"Erreur grave lors du méta-clustering LLM: {e}", exc_info=True)

    # --- Fallback si le LLM échoue ---
    logger.warning("Méta-clustering LLM a échoué. Utilisation d'un regroupement simple comme fallback.")
    simple_meta_names = {}
    simple_assignments = {}

    # Créer un regroupement basique des thèmes
    original_theme_ids = sorted(clusters.keys())
    num_themes_to_group = len(original_theme_ids)
    # Répartir aussi équitablement que possible
    base_size = num_themes_to_group // target_count
    remainder = num_themes_to_group % target_count

    current_theme_index = 0
    for meta_id in range(target_count):
        size = base_size + (1 if meta_id < remainder else 0)
        simple_meta_names[meta_id] = f"Groupe Thématique {meta_id+1}"
        # Assigner les thèmes originaux à ce méta-thème
        for i in range(size):
            if current_theme_index < num_themes_to_group:
                original_id = original_theme_ids[current_theme_index]
                simple_assignments[original_id] = meta_id
                current_theme_index += 1
            else: break # Should not happen with correct calculation

    # S'assurer que tous les thèmes sont assignés (sécurité)
    assigned_count = len(simple_assignments)
    if assigned_count != num_themes_to_group:
         logger.error(f"Erreur dans le fallback de méta-clustering: {assigned_count} assignés vs {num_themes_to_group} attendus.")
         # Tenter de corriger en assignant les manquants au dernier groupe? Risqué.
         # Ou juste logguer l'erreur.

    logger.info(f"Fallback de méta-clustering terminé avec {len(simple_meta_names)} groupes.")
    return simple_meta_names, simple_assignments


# -----------------------------------
# Étape 8 : Compilation du tableau final (INCHANGÉ)
# -----------------------------------


def compile_results_with_meta(
    all_codes: List[Dict[str, Any]],
    code_to_cluster: Dict[str, int], # Mapping: code_text -> original_cluster_id
    theme_map: Dict[int, str],       # Mapping: original_cluster_id -> theme_name
    meta_theme_map: Dict[int, str],  # Mapping: meta_theme_id -> meta_theme_name
    theme_to_meta: Dict[int, int],   # Mapping: original_cluster_id -> meta_theme_id
) -> pd.DataFrame:
    """
    Génère un DataFrame pandas avec colonnes: Méta-thème, Thème, Code, Extrait
    """
    rows = []
    logger.info("Compilation des résultats finaux en DataFrame.")
    logger.debug(f"Mappings reçus: code->cluster: {len(code_to_cluster)} items, cluster->theme: {len(theme_map)} items, meta->name: {len(meta_theme_map)} items, theme->meta: {len(theme_to_meta)} items.")


    missing_assignments_count = 0
    codes_without_cluster = 0

    for code_dict in all_codes:
        code_text = code_dict.get("code")
        excerpt = code_dict.get("excerpt")

        if not code_text:
             logger.warning(f"Code manquant dans l'objet: {code_dict}. Ignoré.")
             continue

        # 1. Trouver l'ID du cluster original pour ce texte de code
        original_cluster_id = code_to_cluster.get(code_text, -1) # Utiliser -1 comme indicateur d'absence

        if original_cluster_id == -1:
            codes_without_cluster +=1
            theme_name = "Thème non assigné"
            meta_theme_name = "Méta-thème non assigné"
        else:
            # 2. Trouver le nom du thème original
            theme_name = theme_map.get(original_cluster_id, f"Thème Inconnu ({original_cluster_id})")

            # 3. Trouver l'ID du méta-thème pour ce thème original
            meta_theme_id = theme_to_meta.get(original_cluster_id, -1) # Utiliser -1 comme indicateur

            if meta_theme_id == -1:
                missing_assignments_count += 1
                meta_theme_name = "Méta-thème non assigné"
                 # Log seulement la première fois pour éviter le spam
                if missing_assignments_count == 1:
                    logger.warning(f"Le thème original ID {original_cluster_id} ('{theme_name}') n'a pas d'assignation à un méta-thème dans theme_to_meta map. Vérifiez l'étape de méta-clustering.")
            else:
                # 4. Trouver le nom du méta-thème
                meta_theme_name = meta_theme_map.get(meta_theme_id, f"Méta-thème Inconnu ({meta_theme_id})")

        rows.append(
            {"Méta-thème": meta_theme_name, "Thème": theme_name, "Code": code_text, "Extrait": excerpt}
        )

    if codes_without_cluster > 0:
         logger.warning(f"{codes_without_cluster} codes n'ont pas pu être associés à un cluster original (problème potentiel dans code_to_cluster map).")
    if missing_assignments_count > 0:
         logger.warning(f"Il y a eu {missing_assignments_count} instances où un thème original n'a pas pu être lié à un méta-thème (problème potentiel dans theme_to_meta map).")


    # Créer le DataFrame
    if not rows:
         logger.warning("Aucune donnée à compiler dans le DataFrame.")
         # Retourner un DataFrame vide avec les colonnes attendues
         return pd.DataFrame(columns=["Méta-thème", "Thème", "Code", "Extrait"])

    df = pd.DataFrame(rows)

    # Trier par méta-thème puis par thème pour une meilleure lisibilité
    # Gérer les cas où les noms sont des placeholders comme "Non assigné"
    # En les triant peut-être à la fin? Ou au début? Mettons-les au début.
    df['Méta-thème'] = pd.Categorical(df['Méta-thème'], ordered=True)
    df['Thème'] = pd.Categorical(df['Thème'], ordered=True)

    # Trier. pandas trie les catégories dans l'ordre où elles apparaissent par défaut,
    # sauf si on définit explicitement un ordre. Essayons le tri par défaut d'abord.
    try:
         df = df.sort_values(by=["Méta-thème", "Thème", "Code"])
    except Exception as sort_e:
         logger.warning(f"Échec du tri du DataFrame: {sort_e}. Le DataFrame non trié sera retourné.")


    logger.info(f"Compilation terminée. DataFrame créé avec {len(df)} lignes.")
    return df


# -----------------------
# Pipeline Orchestrateur (INCHANGÉ)
# -----------------------

def run_pipeline(
    pdf_path: str, output_csv_path: str, max_themes: int = 10
):
    """
    Exécute le pipeline complet d'analyse qualitative pour UN SEUL PDF.
    """
    pdf_basename = os.path.basename(pdf_path)
    try:
        # Initialisation (la clé API est déjà chargée globalement ou passée)
        logger.info(f"--- Démarrage Pipeline pour: {pdf_basename} ---")
        api_key = load_api_key() # Recharger au cas où, bien que déjà fait avant
        client = create_groq_client(api_key)

        # Extraction et préparation du texte
        logger.info(f"[{pdf_basename}] Étape 1/8: Extraction du texte")
        raw = extract_text_from_pdf(pdf_path)
        text = clean_text(raw)
        if not text:
             logger.warning(f"[{pdf_basename}] Aucun texte extrait ou texte vide. Arrêt du pipeline pour ce fichier.")
             return None # Retourner None pour indiquer l'échec

        # Segmentation
        logger.info(f"[{pdf_basename}] Étape 2/8: Segmentation du texte")
        segments = segment_text(text)
        if not segments:
             logger.warning(f"[{pdf_basename}] Aucune segment généré. Arrêt du pipeline pour ce fichier.")
             return None

        # Configuration des modèles (peut-être les rendre configurables via args/env?)
        model_cot = "llama3-70b-8192" # Modèle puissant pour analyse/génération initiale
        model_judge = "llama3-8b-8192" # Modèle rapide et efficace pour validation/correction

        # Analyse, Codification et Validation
        logger.info(f"[{pdf_basename}] Étape 3 & 4/8: Analyse CoT, Codification & Validation Juge ({len(segments)} segments)")
        all_codes = []
        processed_segments = 0
        for i, seg in enumerate(segments):
            # Log progress more dynamically
            if (i + 1) % 5 == 0 or i == 0 or i == len(segments) - 1:
                 logger.info(f"[{pdf_basename}] Traitement segment {i+1}/{len(segments)}...")
            codes = analyze_and_code_segment(client, model_cot, seg)
            if codes:
                # Passer les codes au juge même s'ils sont vides, au cas où le juge ferait qqc? Non, inutile.
                validated = judge_codes(client, model_judge, seg, codes)
                if validated: # Ajouter seulement s'il reste des codes après jugement
                    all_codes.extend(validated)
            processed_segments += 1


        logger.info(f"[{pdf_basename}] {len(all_codes)} codes valides générés après analyse de {processed_segments} segments.")

        if not all_codes:
            logger.warning(f"[{pdf_basename}] Aucun code généré ou validé pour ce fichier. Sauvegarde d'un CSV vide.")
             # Créer un DataFrame vide avec les bonnes colonnes et le sauvegarder
            empty_df = pd.DataFrame(columns=["Méta-thème", "Thème", "Code", "Extrait"])
            empty_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"[{pdf_basename}] Fichier CSV vide sauvegardé dans {output_csv_path}")
            return empty_df # Retourner le DF vide

        # Clustering limité à max_themes
        logger.info(f"[{pdf_basename}] Étape 5/8: Clustering des {len(all_codes)} codes (max {max_themes} thèmes)")
        clusters, code_to_cluster = cluster_codes_limited(all_codes, max_themes)
        if not clusters:
             logger.warning(f"[{pdf_basename}] Clustering n'a retourné aucun cluster. Tentative de compilation sans thèmes.")
             # Fournir des mappings vides pour que compile_results fonctionne
             theme_map = {}
             meta_theme_map = {0: "Méta-thème non défini"}
             theme_to_meta = {} # Vide, car pas de thèmes
        else:
            # Labelling des thèmes
            logger.info(f"[{pdf_basename}] Étape 6/8: Labelling des {len(clusters)} thèmes")
            theme_map = label_themes(client, model_cot, clusters) # Utiliser le modèle CoT pour labelliser aussi

            # Méta-clustering si nécessaire
            num_clusters_found = len(clusters)
            if num_clusters_found > max_themes:
                logger.info(f"[{pdf_basename}] Étape 7/8: Méta-clustering requis ({num_clusters_found} > {max_themes})")
                meta_theme_map, theme_to_meta = meta_cluster_themes(
                    client, model_cot, clusters, theme_map, max_themes
                )
            else:
                logger.info(f"[{pdf_basename}] Étape 7/8: Méta-clustering non requis ({num_clusters_found} <= {max_themes})")
                # Pas besoin de méta-clustering, les thèmes sont les méta-thèmes
                meta_theme_map = {cid: name for cid, name in theme_map.items()}
                theme_to_meta = {cid: cid for cid in clusters.keys()}

        # Compilation des résultats
        logger.info(f"[{pdf_basename}] Étape 8/8: Compilation des résultats")
        df = compile_results_with_meta(
            all_codes, code_to_cluster, theme_map, meta_theme_map, theme_to_meta
        )

        # Sauvegarde
        # S'assurer que le dossier de sortie existe
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Dossier de sortie créé: {output_dir}")

        # Utiliser utf-8-sig pour inclure le BOM, améliore la compatibilité Excel avec les accents
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"[{pdf_basename}] Résultats finaux sauvegardés dans {output_csv_path}")

        # Affichage des statistiques pour ce fichier
        if not df.empty:
             meta_themes_count = df["Méta-thème"].nunique()
             themes_count = df["Thème"].nunique()
             codes_count = len(df)

             logger.info(f"[{pdf_basename}] Analyse terminée:")
             logger.info(f"  - {meta_themes_count} méta-thèmes")
             logger.info(f"  - {themes_count} thèmes")
             logger.info(f"  - {codes_count} codes")
        else:
             logger.info(f"[{pdf_basename}] Analyse terminée, mais aucun résultat à afficher (DataFrame vide).")


        print(f"\n--- Analyse terminée pour {pdf_basename} ---")
        print(f"Résultats sauvegardés dans: {output_csv_path}")
        print("-" * (len(pdf_basename) + 28))


        return df

    except FileNotFoundError as fnf_error:
         logger.error(f"[{pdf_basename}] ERREUR FATALE: Fichier PDF non trouvé à {pdf_path}. Assurez-vous que le chemin est correct et que le fichier existe. Erreur: {fnf_error}")
         print(f"\nERREUR: Fichier PDF non trouvé pour {pdf_basename}. Voir logs.")
         return None # Indiquer l'échec
    except Exception as e:
        logger.error(f"[{pdf_basename}] ERREUR dans le pipeline principal pour {pdf_path}: {e}", exc_info=True)
        print(f"\nERREUR pendant le traitement de {pdf_basename}. Voir logs pour détails.")
        # Optionnel: sauvegarder un fichier d'erreur?
        return None # Indiquer l'échec


# --- NOUVEL ORCHESTRATEUR POUR GOOGLE DRIVE ---
def process_drive_folder(drive_service: Any, folder_id: str, output_dir: str, max_themes: int):
    """
    Orchestre le traitement de tous les PDF d'un dossier Google Drive.
    """
    logger.info(f"--- Démarrage du traitement du dossier Google Drive ID: {folder_id} ---")
    logger.info(f"Les résultats CSV seront sauvegardés dans: {output_dir}")

    # S'assurer que le dossier de sortie existe
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Dossier de sortie créé: {output_dir}")
        except OSError as e:
            logger.error(f"Impossible de créer le dossier de sortie {output_dir}: {e}")
            return

    # Lister les PDF dans le dossier
    pdf_files = list_pdfs_in_folder(drive_service, folder_id)

    if not pdf_files:
        logger.warning(f"Aucun fichier PDF trouvé dans le dossier Google Drive {folder_id}.")
        print("Aucun PDF trouvé dans le dossier spécifié.")
        return

    total_files = len(pdf_files)
    processed_count = 0
    error_count = 0

    # Utiliser un dossier temporaire pour les téléchargements
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Utilisation du dossier temporaire: {temp_dir}")
        for i, pdf_file in enumerate(pdf_files):
            file_id = pdf_file['id']
            file_name = pdf_file['name']
            logger.info(f"--- Traitement Fichier {i+1}/{total_files}: {file_name} (ID: {file_id}) ---")

            # Construire le nom du fichier CSV de sortie
            base_name = os.path.splitext(file_name)[0]
            # Nettoyer un peu le nom de base pour éviter les caractères invalides dans les noms de fichiers
            safe_base_name = re.sub(r'[\\/*?:"<>|]', '_', base_name)
            output_csv_name = f"{safe_base_name}_codification.csv"
            output_csv_path = os.path.join(output_dir, output_csv_name)

            # Vérifier si le fichier existe déjà pour potentiellement le sauter? Non, traitons tout.

            # Télécharger le PDF dans le dossier temporaire
            temp_pdf_path = download_pdf_to_temp(drive_service, file_id, file_name, temp_dir)

            if temp_pdf_path:
                try:
                    # Exécuter le pipeline d'analyse sur le fichier téléchargé
                    result_df = run_pipeline(temp_pdf_path, output_csv_path, max_themes)
                    if result_df is not None: # run_pipeline retourne None en cas d'échec majeur
                        processed_count += 1
                        logger.info(f"Traitement de {file_name} terminé avec succès.")
                    else:
                         error_count += 1
                         logger.error(f"Échec du pipeline pour {file_name}.")

                except Exception as e:
                    error_count += 1
                    logger.error(f"Erreur imprévue lors de l'appel à run_pipeline pour {file_name}: {e}", exc_info=True)
                    print(f"Erreur imprévue lors du traitement de {file_name}. Voir logs.")
                finally:
                    # Nettoyer le fichier temporaire spécifique (même si TemporaryDirectory le fera à la fin)
                    # Cela libère de l'espace disque plus tôt pour les gros fichiers.
                    if os.path.exists(temp_pdf_path):
                        try:
                            os.remove(temp_pdf_path)
                            logger.debug(f"Fichier temporaire {temp_pdf_path} supprimé.")
                        except OSError as e:
                             logger.warning(f"Impossible de supprimer le fichier temporaire {temp_pdf_path}: {e}")
            else:
                error_count += 1
                logger.error(f"Échec du téléchargement pour {file_name}. Fichier sauté.")
                print(f"Échec du téléchargement de {file_name}. Fichier sauté.")

            print("-" * 40) # Séparateur entre les fichiers

    # Fin du traitement
    logger.info("--- Traitement du dossier Google Drive terminé ---")
    logger.info(f"Fichiers PDF trouvés: {total_files}")
    logger.info(f"Fichiers traités avec succès: {processed_count}")
    logger.info(f"Fichiers échoués (téléchargement ou pipeline): {error_count}")
    print("\n--- Traitement terminé ---")
    print(f"Total de fichiers PDF trouvés : {total_files}")
    print(f"Fichiers traités avec succès : {processed_count}")
    print(f"Fichiers en échec         : {error_count}")
    print(f"Les résultats CSV se trouvent dans : {output_dir}")
# --- FIN NOUVEL ORCHESTRATEUR ---


# --------------
# Entrée principale (Modifiée pour Google Drive)
# --------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse qualitative sociologie du numérique depuis Google Drive"
    )
    # Arguments pour Google Drive (optionnels si .env est utilisé)
    parser.add_argument(
        "--gdrive-folder-id",
        help="ID du dossier Google Drive contenant les PDF (alternative à .env)",
        default=None
    )
    parser.add_argument(
        "--gdrive-creds",
        help="Chemin vers le fichier JSON des credentials du compte de service (alternative à .env)",
        default=None
    )
    # Argument obligatoire pour le dossier de sortie
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Dossier où enregistrer les fichiers CSV résultants",
    )
    parser.add_argument(
        "--max-themes",
        type=int,
        default=10,
        help="Nombre maximum de thèmes finaux à générer par PDF (défaut: 10)"
    )
    args = parser.parse_args()

    # Charger la configuration GDrive depuis .env ou arguments
    creds_path_env, folder_id_env = load_gdrive_config()

    # Donner la priorité aux arguments de la ligne de commande sur .env
    gdrive_creds_path = args.gdrive_creds if args.gdrive_creds else creds_path_env
    gdrive_folder_id = args.gdrive_folder_id if args.gdrive_folder_id else folder_id_env

    # Vérifications
    if not gdrive_creds_path:
        logger.error("Chemin des credentials Google Drive non fourni via argument --gdrive-creds ou variable d'environnement GOOGLE_APPLICATION_CREDENTIALS.")
        print("ERREUR: Chemin des credentials Google Drive manquant.")
        exit(1)
    if not gdrive_folder_id:
        logger.error("ID du dossier Google Drive non fourni via argument --gdrive-folder-id ou variable d'environnement GOOGLE_DRIVE_FOLDER_ID.")
        print("ERREUR: ID du dossier Google Drive manquant.")
        exit(1)
    if not args.output_dir:
         logger.error("Le dossier de sortie --output-dir est requis.")
         print("ERREUR: Argument --output-dir manquant.")
         exit(1)


    # Créer le service Drive
    drive_service = get_drive_service(gdrive_creds_path)

    if drive_service:
        # Lancer le traitement du dossier
        process_drive_folder(drive_service, gdrive_folder_id, args.output_dir, args.max_themes)
    else:
        logger.error("Impossible d'initialiser le service Google Drive. Vérifiez les credentials et la connexion.")
        print("ERREUR: Impossible d'initialiser le service Google Drive.")
        exit(1)

    logger.info("Script terminé.")