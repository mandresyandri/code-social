import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import PyPDF2
from dotenv import load_dotenv
from groq import Groq
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        raise ValueError("La variable d'environnement 'GROQ_API_KEY' n'est pas définie")
    return api_key


def create_groq_client(api_key: str) -> Groq:
    """Crée et retourne un client Groq."""
    return Groq(api_key=api_key)

# ----------------
# Extraction Texte
# ----------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un fichier PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
    
    text = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_txt = page.extract_text() or ''
                text.append(page_txt)
                logger.info(f"Page {i+1}/{len(reader.pages)} extraite")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du PDF: {e}")
        raise
        
    return "\n".join(text)


def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les espaces multiples."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ----------------------
# Étape 1 : Segmentation
# ----------------------

def segment_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Découpe le texte en segments d'environ chunk_size mots avec chevauchement,
    en tentant de préserver les limites de phrases.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    segments = []
    current_segment = []
    current_size = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_size = len(sentence_words)
        
        if current_size + sentence_size > chunk_size and current_segment:
            # Ajouter le segment actuel à la liste
            segments.append(" ".join(current_segment))
            
            # Conserver les dernières phrases pour un chevauchement
            overlap_words = min(overlap, current_size)
            words_to_keep = " ".join(current_segment).split()[-overlap_words:]
            current_segment = words_to_keep + sentence_words
            current_size = len(current_segment)
        else:
            current_segment.extend(sentence_words)
            current_size += sentence_size
    
    # Ajouter le dernier segment s'il n'est pas vide
    if current_segment:
        segments.append(" ".join(current_segment))
    
    logger.info(f"Texte segmenté en {len(segments)} segments")
    return segments

# -------------------------------------
# Étape 2 & 3 : Analyse CoT + Codification
# -------------------------------------

def extract_json_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extrait du JSON valide depuis un texte potentiellement mélangé."""
    # Rechercher un pattern JSON dans le texte
    json_pattern = r'\[\s*{.*}\s*\]'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Essayer d'extraire entre crochets
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return None


def analyze_and_code_segment(client: Groq, model: str, segment: str, max_retries: int = 2) -> List[Dict[str, Any]]:
    """
    Effectue une analyse CoT puis propose jusqu'à 3 codes (JSON).
    Inclut un mécanisme de réessai en cas d'échec.
    """
    prompt = (
        "Tu es un sociologue du numérique spécialisé en analyse qualitative.\n\n"
        "Étape 1: Analyse ce segment d'entretien en pensant à voix haute pour identifier les idées principales, "
        "y compris les idées implicites, les nuances ou l'ironie.\n\n"
        "Étape 2: Propose jusqu'à 3 codes sociologiques brefs (1-5 mots) accompagnés d'un extrait pertinent. "
        "Chaque code doit refléter un concept sociologique significatif.\n\n"
        f"Segment à analyser:\n'''\n{segment}\n'''\n\n"
        "Tu dois répondre UNIQUEMENT en format JSON suivant ce modèle précis:\n"
        "[{\"code\": \"concept sociologique\", \"excerpt\": \"extrait pertinent\"}, ...]\n"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
                stream=False
            )
            text = resp.choices[0].message.content.strip()
            
            # Tentative d'extraction du JSON
            codes = extract_json_from_text(text)
            if codes:
                return codes
            
            # Si échec d'extraction, demander une reformulation
            if attempt < max_retries:
                fix_prompt = (
                    "Ta réponse précédente n'est pas un JSON valide. "
                    "Réponds UNIQUEMENT avec un tableau JSON au format: "
                    "[{\"code\": \"concept sociologique\", \"excerpt\": \"extrait pertinent\"}, ...]\n"
                    "N'inclus aucune explication, juste le JSON."
                )
                messages = [{"role": "user", "content": prompt}, 
                           {"role": "assistant", "content": text},
                           {"role": "user", "content": fix_prompt}]
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse du segment (tentative {attempt+1}): {e}")
            if attempt == max_retries:
                logger.error("Échec de l'analyse après plusieurs tentatives")
                return []
    
    logger.warning("Impossible d'obtenir un JSON valide, retour d'une liste vide")
    return []

# -----------------------------
# Étape 4 : Validation (juge)
# -----------------------------

def judge_codes(client: Groq, model: str, segment: str, codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Valide ou corrige chaque code proposé par l'agent.
    Retourne une liste de codes validés/améliorés.
    """
    if not codes:
        return []
    
    codes_json = json.dumps(codes, ensure_ascii=False)
    
    prompt = (
        "Tu es un expert en sociologie évaluant des codes d'analyse qualitative.\n\n"
        f"Voici un segment d'entretien:\n'''\n{segment}\n'''\n\n"
        f"Et voici les codes proposés:\n{codes_json}\n\n"
        "Pour chaque code, évalue sa pertinence et sa précision. "
        "Puis retourne un tableau JSON avec les codes validés ou améliorés, en suivant ce format:\n"
        "[{\"code\": \"code validé ou amélioré\", \"excerpt\": \"extrait pertinent\"}, ...]\n\n"
        "Ton tableau JSON doit être le SEUL élément de ta réponse."
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
            stream=False
        )
        content = resp.choices[0].message.content.strip()
        
        # Extraction du JSON
        validated = extract_json_from_text(content)
        if validated:
            return validated
        
        # Si échec, deuxième tentative avec prompt simplifié
        fix_prompt = (
            "Réformate ta réponse précédente en JSON valide uniquement. "
            "Format attendu: [{\"code\": \"...\", \"excerpt\": \"...\"}, ...]\n"
            "Aucun texte supplémentaire."
        )
        
        fix_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {"role": "user", "content": fix_prompt}
            ],
            temperature=0,
            max_tokens=512
        )
        
        fixed = fix_resp.choices[0].message.content.strip()
        validated = extract_json_from_text(fixed)
        
        return validated if validated else codes  # Fallback aux codes originaux
    
    except Exception as e:
        logger.error(f"Erreur lors de la validation des codes: {e}")
        return codes  # En cas d'erreur, conserver les codes originaux

# --------------------------------
# Étape 5 : Clustering des codes (LIMITÉ À 10 THÈMES MAX)
# --------------------------------

def cluster_codes_limited(all_codes: List[Dict[str, Any]], max_clusters: int = 10, embedding_model: str = 'all-MiniLM-L6-v2') -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    """
    Regroupe les codes par similarité sémantique en limitant à max_clusters thèmes.
    Retourne un dict cluster_id -> list de codes et un mapping code -> cluster_id.
    """
    if not all_codes:
        return {}, {}
    
    # Extraire uniquement les textes des codes
    code_texts = [code['code'] for code in all_codes]
    
    try:
        # Générer les embeddings
        embedder = SentenceTransformer(embedding_model)
        embeddings = embedder.encode(code_texts)
        
        # Déterminer le nombre de clusters à utiliser
        n_clusters = min(max_clusters, len(all_codes))
        
        # Clustering hiérarchique avec nombre fixe de clusters
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings)
        
        # Organiser en clusters
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        code_to_cluster: Dict[str, int] = {}
        
        for code_obj, label in zip(all_codes, labels):
            code_text = code_obj['code']
            clusters.setdefault(int(label), []).append(code_obj)
            code_to_cluster[code_text] = int(label)
        
        logger.info(f"Clustering terminé: {len(clusters)} thèmes identifiés (limité à {max_clusters})")
        
        return clusters, code_to_cluster
    
    except Exception as e:
        logger.error(f"Erreur lors du clustering: {e}")
        # Fallback: regrouper en max_clusters de manière basique si nécessaire
        simple_clusters = {}
        simple_code_to_cluster = {}
        
        # Regroupement simplifié pour fallback
        chunk_size = max(1, len(all_codes) // max_clusters)
        for i, code in enumerate(all_codes):
            cluster_id = min(i // chunk_size, max_clusters - 1)
            simple_clusters.setdefault(cluster_id, []).append(code)
            simple_code_to_cluster[code['code']] = cluster_id
            
        return simple_clusters, simple_code_to_cluster

# -------------------------------
# Étape 6 : Labelling des thèmes
# -------------------------------

def label_themes(client: Groq, model: str, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
    """
    Pour chaque cluster, demande au LLM un titre de thème sociologique.
    """
    theme_labels: Dict[int, str] = {}
    
    for cluster_id, codes in clusters.items():
        # Extraire le texte des codes et les excerpts pour le contexte
        code_items = [{"code": c['code'], "excerpt": c['excerpt']} for c in codes]
        codes_json = json.dumps(code_items, ensure_ascii=False)
        
        prompt = (
            "Tu es un sociologue synthétisant des résultats d'analyse qualitative.\n\n"
            f"Voici un groupe de codes apparentés avec leurs extraits:\n{codes_json}\n\n"
            "Propose un titre de thème sociologique concis (2-4 mots) qui capture l'essence commune de ces codes.\n\n"
            "Réponds uniquement en JSON: {\"theme\": \"...\"}"
        )
        
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=128,
                stream=False
            )
            content = resp.choices[0].message.content.strip()
            
            # Extraction du JSON
            match = re.search(r'\{\s*"theme"\s*:\s*"([^"]+)"\s*\}', content)
            if match:
                theme_labels[cluster_id] = match.group(1)
            else:
                # Tentative d'extraction JSON complète
                try:
                    data = json.loads(content)
                    if "theme" in data:
                        theme_labels[cluster_id] = data["theme"]
                    else:
                        theme_labels[cluster_id] = f"Thème {cluster_id+1}"
                except:
                    theme_labels[cluster_id] = f"Thème {cluster_id+1}"
        
        except Exception as e:
            logger.error(f"Erreur lors du labelling du cluster {cluster_id}: {e}")
            theme_labels[cluster_id] = f"Thème {cluster_id+1}"
    
    return theme_labels

# -----------------------------------
# Étape 7 : Second-level clustering (pour regrouper si > 10 thèmes)
# -----------------------------------

def meta_cluster_themes(client: Groq, model: str, clusters: Dict[int, List[Dict[str, Any]]], 
                        theme_map: Dict[int, str], target_count: int = 10) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Si nécessaire, regroupe les thèmes en méta-thèmes pour atteindre le nombre cible.
    Retourne un dict cluster_id -> nom du méta-thème et un mapping theme_id -> meta_theme_id.
    """
    if len(clusters) <= target_count:
        # Pas besoin de regroupement supplémentaire
        return theme_map, {k: k for k in clusters.keys()}
    
    # Pour chaque thème, créer une description complète
    theme_descriptions = {}
    for cluster_id, codes in clusters.items():
        theme_name = theme_map.get(cluster_id, f"Thème {cluster_id}")
        codes_text = "; ".join([f"{c['code']}: {c['excerpt']}" for c in codes[:3]])  # Limiter pour éviter trop de tokens
        theme_descriptions[cluster_id] = f"Thème '{theme_name}' avec codes: {codes_text}"
    
    # Demander au LLM de regrouper les thèmes
    themes_json = json.dumps(theme_descriptions, ensure_ascii=False)
    prompt = (
        "Tu es un expert en sociologie chargé de regrouper des thèmes similaires.\n\n"
        f"Voici {len(theme_descriptions)} thèmes issus d'une analyse qualitative:\n{themes_json}\n\n"
        f"Regroupe ces thèmes en {target_count} méta-thèmes maximum. Chaque thème original doit être assigné "
        f"à un seul méta-thème plus large.\n\n"
        "Réponds uniquement en JSON avec le format suivant:\n"
        "{\"meta_themes\": {\"0\": \"Nom du méta-thème 1\", \"1\": \"Nom du méta-thème 2\", ...}, "
        "\"assignments\": {\"0\": 1, \"2\": 0, ...}}\n\n"
        "Où 'meta_themes' est un dictionnaire des méta-thèmes numérotés, et 'assignments' indique pour chaque "
        "thème original (clé) le numéro du méta-thème auquel il est assigné (valeur)."
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            stream=False
        )
        content = resp.choices[0].message.content.strip()
        
        # Extraction du JSON
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            try:
                data = json.loads(json_str)
                meta_themes = {int(k): v for k, v in data.get("meta_themes", {}).items()}
                assignments = {int(k): int(v) for k, v in data.get("assignments", {}).items()}
                
                # Vérifier que tous les thèmes sont assignés
                for theme_id in clusters.keys():
                    if theme_id not in assignments:
                        # Assigner à un méta-thème existant ou créer un nouveau
                        if meta_themes:
                            # Trouver le méta-thème avec le moins d'assignations
                            meta_counts = {mt: sum(1 for t, m in assignments.items() if m == mt) for mt in meta_themes}
                            min_meta = min(meta_counts, key=meta_counts.get)
                            assignments[theme_id] = min_meta
                        else:
                            # Créer un premier méta-thème
                            meta_themes[0] = f"Méta-thème 1"
                            assignments[theme_id] = 0
                
                return meta_themes, assignments
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON lors du méta-clustering")
    except Exception as e:
        logger.error(f"Erreur lors du méta-clustering: {e}")
    
    # Fallback : regroupement simple
    simple_meta = {}
    simple_assignments = {}
    
    # Créer un regroupement basique des thèmes
    chunk_size = max(1, len(clusters) // target_count)
    for i, theme_id in enumerate(sorted(clusters.keys())):
        meta_id = min(i // chunk_size, target_count - 1)
        if meta_id not in simple_meta:
            simple_meta[meta_id] = f"Groupe thématique {meta_id+1}"
        simple_assignments[theme_id] = meta_id
    
    return simple_meta, simple_assignments

# -----------------------------------
# Étape 8 : Compilation du tableau final
# -----------------------------------

def compile_results_with_meta(all_codes: List[Dict[str, Any]], 
                             code_to_cluster: Dict[str, int],
                             theme_map: Dict[int, str],
                             meta_theme_map: Dict[int, str],
                             theme_to_meta: Dict[int, int]) -> pd.DataFrame:
    """
    Génère un DataFrame pandas avec colonnes: Méta-thème, Thème, Code, Extrait
    """
    rows = []
    
    for code_dict in all_codes:
        code = code_dict['code']
        excerpt = code_dict['excerpt']
        
        cluster_id = code_to_cluster.get(code, -1)
        theme = theme_map.get(cluster_id, 'À catégoriser')
        
        meta_id = theme_to_meta.get(cluster_id, -1)
        meta_theme = meta_theme_map.get(meta_id, 'Non classé')
        
        rows.append({
            'Thème': theme, 
            'Code': code, 
            'Extrait': excerpt
        })
    
    # Créer le DataFrame
    df = pd.DataFrame(rows)
    
    # Trier par méta-thème puis par thème
    df = df.sort_values(['Méta-thème', 'Thème'])
    
    return df

# -----------------------
# Pipeline Orchestrateur
# -----------------------

def run_pipeline(pdf_path: str, output_path: str = 'resultats_analyse.csv', max_themes: int = 10):
    """
    Exécute le pipeline complet d'analyse qualitative avec limite de thèmes.
    """
    try:
        # Initialisation
        logger.info(f"Démarrage du pipeline d'analyse qualitative pour {pdf_path}")
        api_key = load_api_key()
        client = create_groq_client(api_key)
        
        # Extraction et préparation du texte
        logger.info("Extraction du texte depuis le PDF")
        raw = extract_text_from_pdf(pdf_path)
        text = clean_text(raw)
        
        # Segmentation
        logger.info("Segmentation du texte")
        segments = segment_text(text)
        
        # Configuration des modèles
        model_cot = 'meta-llama/llama-4-maverick-17b-128e-instruct'  # Pour l'analyse CoT
        model_judge = 'llama3-8b-8192'  # Pour la validation
        
        # Analyse et codification
        logger.info("Analyse et codification des segments")
        all_codes = []
        
        for i, seg in enumerate(segments):
            logger.info(f"Traitement du segment {i+1}/{len(segments)}")
            codes = analyze_and_code_segment(client, model_cot, seg)
            if codes:
                validated = judge_codes(client, model_judge, seg, codes)
                all_codes.extend(validated)
        
        logger.info(f"Total de {len(all_codes)} codes générés")
        
        # Clustering limité à max_themes
        logger.info(f"Regroupement des codes en {max_themes} thèmes maximum")
        clusters, code_to_cluster = cluster_codes_limited(all_codes, max_themes)
        
        # Labelling des thèmes
        logger.info("Attribution de labels aux thèmes")
        theme_map = label_themes(client, model_cot, clusters)
        
        # Si encore trop de thèmes, faire un méta-clustering
        if len(clusters) > max_themes:
            logger.info(f"Second regroupement pour atteindre {max_themes} thèmes maximum")
            meta_theme_map, theme_to_meta = meta_cluster_themes(
                client, model_cot, clusters, theme_map, max_themes)
        else:
            # Pas besoin de méta-clustering
            meta_theme_map = {k: v for k, v in theme_map.items()}
            theme_to_meta = {k: k for k in clusters.keys()}
        
        # Compilation des résultats
        logger.info("Compilation des résultats")
        df = compile_results_with_meta(all_codes, code_to_cluster, theme_map, meta_theme_map, theme_to_meta)
        
        # Sauvegarde
        df.to_csv(output_path, index=False)
        logger.info(f"Résultats sauvegardés dans {output_path}")
        
        # Affichage des statistiques
        # meta_themes_count = len(set(df['Méta-thème']))
        themes_count = len(set(df['Thème']))
        codes_count = len(df)
        
        print(f"\nRésultats de l'analyse qualitative:")
        # print(f"- {meta_themes_count} méta-thèmes")
        print(f"- {themes_count} thèmes")
        print(f"- {codes_count} codes")
        
        # Afficher un aperçu des méta-thèmes et du nombre de codes associés
        meta_summary = df.groupby('Méta-thème').count()[['Code']].rename(columns={'Code': 'Nombre de codes'})
        print("\nRépartition des codes par méta-thème:")
        print(meta_summary.to_markdown())
        
        # Afficher le tableau complet (limité à 20 premières lignes pour la console)
        print("\nAperçu des résultats:")
        print(df.head().to_markdown(index=False))
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur dans le pipeline: {e}", exc_info=True)
        raise

# --------------
# Entrée principale
# --------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline d'analyse qualitative sociologie du numérique")
    parser.add_argument('pdf_path', help="Chemin vers le PDF d'entretien")
    parser.add_argument('--output', default='resultats_analyse.csv', help="Chemin pour enregistrer les résultats")
    parser.add_argument('--max-themes', type=int, default=10, help="Nombre maximum de thèmes à générer")
    args = parser.parse_args()
    
    run_pipeline(args.pdf_path, args.output, args.max_themes)
