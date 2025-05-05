import json  # Explicitly import json
import logging
import os
import sys
import time  # Import time for sleep
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from groq import APIStatusError, Groq, RateLimitError  # Import specific errors

# --- Configuration ---
CSV_FILE_PATH = "resultats_agent.csv"
REPORT_OUTPUT_FILE = "rapport_analyse_ia.txt"
GROQ_MODEL = "llama3-70b-8192"  # Or 'llama3-8b-8192' for faster, potentially less detailed results
TEMPERATURE = 0.7  # Slightly increased for more creative synthesis

# Retry configuration
MAX_RETRIES = 5  # Nombre maximum de tentatives pour un appel API
INITIAL_DELAY = 1  # Délai initial en secondes avant le premier retry

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- API Key and Client Functions ---
def load_api_key() -> str:
    """Charge la clé API GROQ depuis .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error(
            "La variable d'environnement 'GROQ_API_KEY' n'est pas définie dans le fichier .env"
        )
        raise ValueError("La variable d'environnement 'GROQ_API_KEY' n'est pas définie")
    logging.info("Clé API GROQ chargée.")
    return api_key


def create_groq_client(api_key: str) -> Groq:
    """Crée et retourne un client Groq."""
    try:
        client = Groq(api_key=api_key)
        logging.info("Client Groq créé avec succès.")
        return client
    except Exception as e:
        logging.error(f"Erreur lors de la création du client Groq : {e}")
        raise


# --- Data Handling Functions ---
def load_data(filepath: str) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV dans un DataFrame Pandas."""
    try:
        df = pd.read_csv(filepath)
        logging.info(
            f"Données chargées avec succès depuis {filepath}. Shape: {df.shape}"
        )
        # Basic validation
        expected_columns = ["Méta-thème", "Thème", "Code", "Extrait"]
        if not all(col in df.columns for col in expected_columns):
            logging.warning(
                f"Colonnes attendues non trouvées. Colonnes présentes: {list(df.columns)}"
            )
            # Attempt to proceed if key columns exist, otherwise raise error if critical ones missing
            if "Méta-thème" not in df.columns or "Extrait" not in df.columns:
                raise ValueError(
                    "Le fichier CSV doit contenir au moins les colonnes 'Méta-thème' et 'Extrait'."
                )
        if df.empty:
            logging.warning("Le fichier CSV est vide.")
            raise ValueError("Le fichier CSV est vide.")
        return df
    except FileNotFoundError:
        logging.error(f"Erreur: Le fichier {filepath} n'a pas été trouvé.")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Erreur: Le fichier {filepath} est vide.")
        raise
    except Exception as e:
        logging.error(f"Erreur inattendue lors du chargement de {filepath}: {e}")
        raise


def extract_structured_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convertit le DataFrame en liste d'objets structurés.
    Étape "Parser" de la méthodologie.
    """
    structured_data = []

    for _, row in df.iterrows():
        entry = {
            "meta": row.get("Méta-thème", "N/A"),
            "theme": row.get("Thème", "N/A"),
            "code": row.get("Code", "N/A"),
            "excerpt": row.get("Extrait", "N/A"),
        }
        structured_data.append(entry)

    logging.info(f"Données structurées extraites: {len(structured_data)} entrées")
    return structured_data


def plan_sections(structured_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Identifie les méta-thèmes uniques et génère un plan de sections initial.
    Étape "Planner" de la méthodologie.
    """
    sections = {}
    meta_themes = set(item["meta"] for item in structured_data)

    for meta in meta_themes:
        # Extraire les données pour ce méta-thème
        meta_data = [item for item in structured_data if item["meta"] == meta]
        themes = set(item["theme"] for item in meta_data)
        codes = set(item["code"] for item in meta_data)

        sections[meta] = {
            "title": meta,  # Le titre sera raffiné par l'IA
            "guiding_question": "",  # Sera généré par l'IA
            "themes": list(themes),
            "codes": list(codes),
            "excerpts": [item["excerpt"] for item in meta_data],
            "data": meta_data,
        }

    logging.info(f"Plan des sections créé: {len(sections)} méta-thèmes identifiés")
    return sections


# --- Extended Analysis Functions ---


def generate_planning_with_groq(
    client: Groq, sections: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Utilise Groq pour générer des titres et questions-guides pour chaque section.
    Inclut retry et gestion des erreurs API.
    Partie de l'étape "Planner".
    """
    updated_sections = sections.copy()

    system_prompt = """
    Tu es un sociologue expert en méthodologie qualitative.
    Tu dois générer un titre concis (2-5 mots) et une question-guide analytique pour une section d'analyse,
    en te basant sur les thèmes, codes et extraits fournis.
    La réponse doit être STRICTEMENT au format JSON.
    """

    for meta, section in updated_sections.items():
        themes_str = ", ".join(section["themes"])
        codes_str = ", ".join(section["codes"])
        excerpts_sample = section["excerpts"][
            :3
        ]  # Limite à quelques extraits pour éviter les prompts trop longs
        excerpts_str = "\n".join([f"- {e}" for e in excerpts_sample])

        # --- PROMPT ADAPTÉ POUR JSON ---
        user_prompt = f"""
Méta-thème: {meta}

Thèmes associés: {themes_str}

Codes associés: {codes_str}

Quelques extraits représentatifs:
{excerpts_str}

Pour cette section d'analyse, génère un titre concis et percutant ainsi qu'une question-guide analytique.
Réponds STRICTEMENT au format JSON comme ceci:
{{
  "title": "Titre concis",
  "guiding_question": "Question analytique?"
}}
"""
        # --- FIN PROMPT ADAPTÉ ---

        planning_data = None
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=GROQ_MODEL,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )

                response_content = chat_completion.choices[0].message.content

                # --- VALIDATION JSON ---
                planning_data = json.loads(response_content)
                # Additional check: ensure expected keys are present
                if "title" in planning_data and "guiding_question" in planning_data:
                    logging.info(
                        f"Planification générée pour '{meta}' (Attempt {attempt + 1}): '{planning_data.get('title', 'N/A')}'"
                    )
                    break  # Exit retry loop on success
                else:
                    last_error = "Réponse JSON valide mais structure inattendue"
                    logging.warning(
                        f"Planification JSON inattendue pour '{meta}' (Attempt {attempt + 1}): {response_content}"
                    )

            except RateLimitError as e:
                delay = INITIAL_DELAY * (2**attempt)
                logging.warning(
                    f"Rate limit atteint lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
                )
                last_error = e
                time.sleep(delay)
            except APIStatusError as e:
                # Capture other API errors (like 400)
                logging.error(
                    f"Erreur API lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
                )
                last_error = e
                # For 400 errors due to prompt, retrying the same prompt won't help.
                # For others, maybe a single retry? Let's break for now as per plan's idea of fixing upstream.
                break  # Do not retry other API status errors automatically
            except json.JSONDecodeError as e:
                logging.error(
                    f"Erreur de décodage JSON lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
                )
                last_error = e
                break  # JSON decode error is usually a model issue, retrying might not help
            except Exception as e:
                logging.error(
                    f"Erreur inattendue lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                last_error = e
                break  # Catch other unexpected errors

        if (
            planning_data
            and "title" in planning_data
            and "guiding_question" in planning_data
        ):
            updated_sections[meta]["title"] = planning_data["title"]
            updated_sections[meta]["guiding_question"] = planning_data[
                "guiding_question"
            ]
        else:
            logging.error(
                f"Échec final de la planification pour '{meta}' après {MAX_RETRIES} tentatives. Utilisation des valeurs par défaut. Dernière erreur: {last_error}"
            )
            # Keep default values if the API call fails after retries

    return updated_sections


def generate_section_content_with_groq(
    client: Groq, section: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Utilise Groq pour générer le contenu d'une section avec CoT et paragraphe initial.
    Inclut retry et gestion des erreurs API.
    Étape "Thinker" de la méthodologie.
    """
    meta = section.get(
        "title", section.get("meta", "Section inconnue")
    )  # Use refined title if available
    question = section.get("guiding_question", "")
    themes_str = ", ".join(section["themes"])
    codes_str = ", ".join(section["codes"])

    # Sélectionner tous les extraits pour cette section
    excerpts_str = "\n".join([f"- {e}" for e in section["excerpts"]])

    system_prompt = """
    Tu es un sociologue expert en analyse qualitative. Ta tâche est de générer:
    1. Une brève chaîne de pensée (CoT) qui relie logiquement les codes et extraits fournis
    2. Un paragraphe fluide (100-150 mots) qui synthétise ces données et intègre au moins un extrait complet

    Ton analyse doit être nuancée, rigoureuse et basée sur les données fournies.
    La réponse doit être STRICTEMENT au format JSON.
    """

    # --- PROMPT ADAPTÉ POUR JSON ---
    user_prompt = f"""
Section: {meta}
Question analytique: {question}

Thèmes: {themes_str}

Codes: {codes_str}

Extraits disponibles:
{excerpts_str}

Génère une chaîne de pensée et un paragraphe synthétique.
Réponds STRICTEMENT au format JSON comme ceci:
{{
  "chain_of_thought": "Ta réflexion analytique (40-60 mots)",
  "paragraph": "Ton paragraphe synthétique (100-150 mots incluant au moins un extrait complet entre guillemets)"
}}
"""
    # --- FIN PROMPT ADAPTÉ ---

    content_data = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=GROQ_MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content

            # --- VALIDATION JSON ---
            content_data = json.loads(response_content)
            # Additional check: ensure expected keys are present
            if "chain_of_thought" in content_data and "paragraph" in content_data:
                logging.info(
                    f"Contenu initial généré pour la section '{meta}' (Attempt {attempt + 1})"
                )
                break  # Exit retry loop on success
            else:
                last_error = "Réponse JSON valide mais structure inattendue"
                logging.warning(
                    f"Contenu initial JSON inattendu pour '{meta}' (Attempt {attempt + 1}): {response_content}"
                )

        except RateLimitError as e:
            delay = INITIAL_DELAY * (2**attempt)
            logging.warning(
                f"Rate limit atteint lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
            )
            last_error = e
            time.sleep(delay)
        except APIStatusError as e:
            logging.error(
                f"Erreur API lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
            )
            last_error = e
            break  # Do not retry other API status errors automatically
        except json.JSONDecodeError as e:
            logging.error(
                f"Erreur de décodage JSON lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
            )
            last_error = e
            break  # JSON decode error is usually a model issue, retrying might not help
        except Exception as e:
            logging.error(
                f"Erreur inattendue lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            last_error = e
            break  # Catch other unexpected errors

    if (
        content_data
        and "chain_of_thought" in content_data
        and "paragraph" in content_data
    ):
        section["chain_of_thought"] = content_data["chain_of_thought"]
        section["initial_paragraph"] = content_data["paragraph"]
    else:
        logging.error(
            f"Échec final de la génération de contenu pour '{meta}' après {MAX_RETRIES} tentatives. Utilisation des valeurs par défaut. Dernière erreur: {last_error}"
        )
        section["chain_of_thought"] = "Analyse non disponible en raison d'une erreur."
        section["initial_paragraph"] = "Contenu non disponible en raison d'une erreur."

    return section


def review_and_improve_with_groq(
    client: Groq, section: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Utilise Groq pour critiquer et améliorer le paragraphe initial avec retry et gestion des erreurs API.
    Étape "Critic" de la méthodologie.
    """
    meta = section.get(
        "title", section.get("meta", "Section inconnue")
    )  # Use refined title if available
    paragraph = section.get("initial_paragraph", "")

    # Vérifier si le paragraphe initial existe et est significatif, sinon créer un fallback pour la révision
    if not paragraph or len(paragraph) < 30:
        logging.warning(
            f"Paragraphe initial insuffisant pour '{meta}', création d'un paragraphe de secours pour la révision."
        )
        if section["excerpts"]:
            sample_excerpt = section["excerpts"][0]
            paragraph_for_review = f"L'analyse des témoignages sur {meta} révèle des perspectives intéressantes. Comme l'illustre cet extrait: \"{sample_excerpt}\", les étudiants développent une approche pragmatique de l'IA dans ce contexte."
        else:
            paragraph_for_review = f"L'analyse relative à {meta} met en évidence des pratiques spécifiques des étudiants concernant l'usage de l'IA dans leur parcours académique, entre gain d'efficacité et questionnements éthiques."
    else:
        paragraph_for_review = paragraph  # Use the generated paragraph

    # Sélectionner soigneusement les extraits pertinents pour le prompt de révision
    relevant_excerpts = []
    for excerpt in section["excerpts"]:
        if len(excerpt) > 15:  # Extraits significatifs
            relevant_excerpts.append(excerpt)
        if (
            len(relevant_excerpts) >= 3
        ):  # Limiter à 3 extraits pertinents dans le prompt
            break

    # S'il n'y a pas assez d'extraits pertinents (>15 chars), prendre les premiers disponibles (jusqu'à 3)
    if len(relevant_excerpts) < 2 and len(section["excerpts"]) > 0:
        # Avoid including the same short excerpts multiple times if they are already in relevant_excerpts
        for excerpt in section["excerpts"][:3]:
            if excerpt not in relevant_excerpts:
                relevant_excerpts.append(excerpt)

    excerpts_str = "\n".join([f"- {e}" for e in relevant_excerpts])

    system_prompt = """
    Tu es un réviseur critique expert en méthodologie qualitative. Ta tâche est d'examiner un paragraphe d'analyse,
    d'identifier 1-2 points à améliorer (fidélité aux données, clarté, nuance) et de proposer une version améliorée.

    Ton paragraphe amélioré DOIT:
    1. Faire 100-150 mots
    2. Intégrer AU MOINS un extrait complet entre guillemets doubles ("...") (préférablement parmi les extraits fournis ou ceux que tu juges pertinents)
    3. Maintenir la cohérence thématique avec le titre de la section
    4. Être fidèle aux données et éviter toute surinterprétation
    5. Répondre STRICTEMENT au format JSON.
    """

    # --- PROMPT ADAPTÉ POUR JSON ---
    user_prompt = f"""
Titre de section: {meta}

Paragraphe à évaluer:
"{paragraph_for_review}"

Extraits originaux (pour référence):
{excerpts_str}

1. Identifie 1 ou 2 points d'amélioration précis (fidélité, clarté, nuance)
2. Propose une version améliorée du paragraphe (100-150 mots) qui intègre AU MOINS un extrait complet entre guillemets doubles.

Réponds STRICTEMENT au format JSON comme ceci:
{{
  "critique": ["Point d'amélioration 1", "Point d'amélioration 2 (optionnel)"],
  "improved_paragraph": "Paragraphe amélioré (100-150 mots avec au moins un extrait entre guillemets)"
}}
"""
    # --- FIN PROMPT ADAPTÉ ---

    review_data = None
    last_error = None

    # MAX_RETRIES est déjà défini globalement
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=GROQ_MODEL,
                temperature=0.6,  # Légèrement inférieur pour une critique plus factuelle
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content

            # --- VALIDATION JSON ---
            review_data = json.loads(response_content)
            improved_paragraph = review_data.get("improved_paragraph", "")

            # Vérification supplémentaire de qualité: s'assurer qu'il y a au moins un extrait entre guillemets doubles et que le paragraphe est significatif
            if improved_paragraph and len(improved_paragraph) > 50:
                # Check for double quotes explicitly
                if '"' in improved_paragraph:
                    logging.info(
                        f"Révision effectuée avec succès pour la section '{meta}' (Attempt {attempt + 1})"
                    )
                    break  # Exit retry loop on success
                else:
                    last_error = "Paragraphe amélioré valide mais pas d'extrait entre guillemets doubles"
                    logging.warning(
                        f"Révision OK mais sans extrait double-guillemet pour '{meta}' (Attempt {attempt + 1})"
                    )
            else:
                last_error = "Paragraphe amélioré insuffisant ou manquant"
                logging.warning(
                    f"Paragraphe amélioré insuffisant/manquant pour '{meta}' (Attempt {attempt + 1})"
                )

        except RateLimitError as e:
            delay = INITIAL_DELAY * (2**attempt)
            logging.warning(
                f"Rate limit atteint lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
            )
            last_error = e
            time.sleep(delay)
        except APIStatusError as e:
            logging.error(
                f"Erreur API lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
            )
            last_error = e
            break  # Do not retry other API status errors automatically
        except json.JSONDecodeError as e:
            logging.error(
                f"Erreur de décodage JSON lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
            )
            last_error = e
            break  # JSON decode error is usually a model issue, retrying might not help
        except Exception as e:
            logging.error(
                f"Erreur inattendue lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            last_error = e
            break  # Catch other unexpected errors

    if (
        review_data
        and "improved_paragraph" in review_data
        and len(review_data["improved_paragraph"]) > 50
    ):
        # Final check for quote even if model didn't include it in first try, add one if possible
        final_paragraph = review_data["improved_paragraph"]
        if '"' not in final_paragraph and relevant_excerpts:
            best_excerpt = max(
                relevant_excerpts, key=len
            )  # Choose the longest available excerpt
            if len(best_excerpt) > 15:  # Only add if excerpt is significant
                final_paragraph = (
                    final_paragraph.strip()
                    + f' Comme l\'illustre cet extrait : "{best_excerpt}".'
                )
                logging.warning(
                    f"Ajout manuel d'un extrait entre guillemets pour '{meta}'."
                )

        section["critique"] = review_data.get("critique", ["Critique non disponible."])
        section["final_paragraph"] = (
            final_paragraph.strip()
        )  # Ensure no leading/trailing whitespace

    else:
        logging.error(
            f"Échec final de la révision pour '{meta}' après {MAX_RETRIES} tentatives. Génération d'un paragraphe de secours. Dernière erreur: {last_error}"
        )
        # If all attempts failed, create a fallback improved paragraph manually
        try:
            # Attempt to include a relevant excerpt in the fallback
            if relevant_excerpts:
                best_excerpt = max(
                    relevant_excerpts, key=len
                )  # Choose the longest available excerpt
                if len(best_excerpt) > 15:  # Only add if excerpt is significant
                    section["final_paragraph"] = (
                        f"L'analyse des données concernant {meta} révèle des perspectives importantes sur l'usage de l'IA par les étudiants. Comme l'illustre cet extrait significatif : \"{best_excerpt}\", les participants développent des stratégies adaptatives face à ces technologies. Cette dimension s'inscrit dans une réflexion plus large sur l'évolution des pratiques académiques et l'autonomie intellectuelle à l'ère numérique."
                    )
                else:
                    # Fallback to initial paragraph if no good excerpts
                    section["final_paragraph"] = paragraph_for_review.strip()
            else:
                # Fallback to initial paragraph if no excerpts
                section["final_paragraph"] = paragraph_for_review.strip()

            section["critique"] = [
                "Amélioration automatique suite à une erreur technique."
            ]
            logging.info(f"Paragraphe de secours généré pour la section '{meta}'")
        except Exception:
            # Ultimate fallback: use the initial paragraph without modification
            section["final_paragraph"] = paragraph_for_review.strip()
            section["critique"] = [
                "Critique et amélioration non disponibles en raison d'erreurs successives."
            ]
            logging.warning(
                f"Utilisation du paragraphe initial ({len(paragraph_for_review.strip())} chars) pour '{meta}' suite à des erreurs critiques"
            )

    return section


def generate_introduction_conclusion_with_groq(
    client: Groq, sections: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Génère l'introduction et la conclusion de la synthèse avec retry et gestion des erreurs API.
    Partie de l'étape "Synthesizer".
    """
    # Extraire les titres des sections (raffinés par l'IA si la planification a réussi)
    section_titles = [section.get("title", meta) for meta, section in sections.items()]

    # Extraire des extraits significatifs pour chaque section (pour enrichir l'intro/conclusion)
    all_excerpts = []
    for section in sections.values():
        section_excerpts = []
        # S'assurer que les extraits ne sont pas trop longs pour le prompt
        for excerpt in section["excerpts"]:
            if len(excerpt) > 40:  # Limiter les extraits trop longs pour le prompt
                section_excerpts.append(
                    excerpt[:200].rsplit(" ", 1)[0] + "..."
                )  # Truncate nicely
            elif len(excerpt) > 15:
                section_excerpts.append(excerpt)

            if (
                len(section_excerpts) >= 2
            ):  # Maximum 2 extraits pertinents par section dans le prompt
                break
        all_excerpts.extend(section_excerpts)

    sample_excerpts = all_excerpts[
        :10
    ]  # Limiter le nombre total d'extraits dans le prompt
    excerpts_str = "\n".join([f"- {e}" for e in sample_excerpts])
    sections_str = ", ".join(section_titles)

    # Créer une introduction par défaut en cas d'échec de l'API
    default_intro = f"""
L'avènement de l'intelligence artificielle, notamment à travers des outils comme ChatGPT, transforme profondément les pratiques des étudiants dans l'enseignement supérieur. Cette synthèse explore les usages, perceptions et implications de ces technologies dans le contexte académique. À travers l'analyse d'entretiens qualitatifs, nous avons identifié plusieurs thématiques centrales, structurées autour des sections suivantes : {sections_str}. Ces dimensions permettent de comprendre comment les étudiants intègrent l'IA dans leurs pratiques quotidiennes, entre gain d'efficacité et questionnements éthiques, et les défis que cela pose pour l'avenir de l'enseignement supérieur.
"""

    # Créer une conclusion par défaut en cas d'échec de l'API
    default_conclusion = f"""
Face à l'intégration croissante de l'IA dans les pratiques académiques, les perspectives se concentrent sur l'adaptation nécessaire de l'enseignement supérieur. Cela inclut la refonte des méthodes d'évaluation pour mieux valoriser la pensée critique et l'analyse plutôt que la simple restitution, la formation des étudiants à une utilisation éthique et discernée des outils d'IA, et une réflexion continue sur la manière de favoriser une collaboration fructueuse entre l'intelligence humaine et artificielle dans le processus d'apprentissage et de recherche. Ces ajustements sont essentiels pour préparer les étudiants aux défis et opportunités de l'ère numérique.
"""

    system_prompt = """
    Tu es un sociologue expert en analyse qualitative. Ta tâche est de générer:
    1. Une introduction captivante pour une synthèse narrative (100-150 mots)
    2. Une conclusion "Perspectives" concise et tournée vers l'avenir (environ 50-80 mots, 3-4 phrases)

    Ces textes doivent encadrer une analyse des usages de l'IA (notamment ChatGPT) par des étudiants dans leur parcours académique.

    L'introduction doit:
    - Présenter le contexte de l'étude et l'importance du sujet (IA dans l'enseignement supérieur)
    - Mentionner brièvement la méthodologie qualitative utilisée (analyse d'entretiens)
    - Présenter les principaux thèmes ou sections qui seront abordés (utiliser la liste fournie)
    - Indiquer l'objectif général de la synthèse.

    La conclusion "Perspectives" doit:
    - Proposer 3-4 pistes de réflexion, implications ou recommandations concrètes basées sur une analyse des usages (formation, évaluation, complémentarité homme-machine, etc.)
    - Être orientée vers des recommandations pratiques pour l'avenir de l'enseignement supérieur face à l'IA.
    - Ne pas simplement résumer l'analyse déjà présentée dans les sections.
    - Répondre STRICTEMENT au format JSON.
    """

    # --- PROMPT ADAPTÉ POUR JSON ---
    user_prompt = f"""
La synthèse sociologique explore l'usage de l'IA par les étudiants et comprend les sections d'analyse suivantes: {sections_str}

Voici des extraits représentatifs d'entretiens avec des étudiants pour contexte:
{excerpts_str}

Génère:
1. Une introduction complète (100-150 mots) pour cette synthèse.
2. Une conclusion "Perspectives" concise (environ 50-80 mots) qui propose des pistes pour l'avenir de l'enseignement supérieur face à l'IA.

Réponds STRICTEMENT au format JSON comme ceci:
{{
  "introduction": "Texte d'introduction (100-150 mots)",
  "conclusion": "Texte de conclusion (environ 50-80 mots)"
}}
"""
    # --- FIN PROMPT ADAPTÉ ---

    bookends_data = None
    last_error = None

    # MAX_RETRIES est déjà défini globalement
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=GROQ_MODEL,
                temperature=0.6,  # Température réduite pour plus de cohérence
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content

            # --- VALIDATION JSON ---
            bookends_data = json.loads(response_content)

            # Additional check: ensure expected keys are present and content is significant
            if (
                bookends_data
                and "introduction" in bookends_data
                and "conclusion" in bookends_data
                and len(bookends_data.get("introduction", "")) > 50
                and len(bookends_data.get("conclusion", "")) > 30
            ):
                logging.info(
                    f"Introduction et conclusion générées avec succès (Attempt {attempt + 1})"
                )
                return (
                    bookends_data["introduction"].strip(),
                    bookends_data["conclusion"].strip(),
                )
            else:
                last_error = "Réponse JSON valide mais structure ou contenu insuffisant"
                logging.warning(
                    f"Intro/Conclusion JSON inattendue ou insuffisante (Attempt {attempt + 1}): {response_content}"
                )

        except RateLimitError as e:
            delay = INITIAL_DELAY * (2**attempt)
            logging.warning(
                f"Rate limit atteint lors de la génération Intro/Conclusion (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
            )
            last_error = e
            time.sleep(delay)
        except APIStatusError as e:
            logging.error(
                f"Erreur API lors de la génération Intro/Conclusion (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
            )
            last_error = e
            break  # Do not retry other API status errors automatically
        except json.JSONDecodeError as e:
            logging.error(
                f"Erreur de décodage JSON lors de la génération Intro/Conclusion (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
            )
            last_error = e
            # Keep the manual extraction attempt from original code as a fallback here
            try:
                # Attempt manual extraction if JSON parsing fails
                if (
                    response_content
                    and "introduction" in response_content.lower()
                    and "conclusion" in response_content.lower()
                ):
                    parts = response_content.split("conclusion", 1)
                    intro_part = (
                        parts[0].lower().split("introduction", 1)[1]
                    )  # Split again on introduction after splitting on conclusion
                    conclu_part = parts[1]

                    # Basic cleaning
                    intro_clean = intro_part.strip().strip('":{} \n').strip()
                    conclu_clean = conclu_part.strip().strip('":{} \n').strip()

                    if len(intro_clean) > 50 and len(conclu_clean) > 30:
                        logging.warning(
                            f"Récupération manuelle partielle Intro/Conclusion après échec JSON pour '{meta}'."
                        )
                        return intro_clean, conclu_clean
                else:
                    logging.warning(
                        f"Échec de la récupération manuelle après erreur JSON pour '{meta}'."
                    )
            except Exception as manual_e:
                logging.warning(
                    f"Erreur lors de la tentative de récupération manuelle: {manual_e}"
                )

            break  # Break retry loop after JSON error and manual recovery attempt

        except Exception as e:
            logging.error(
                f"Erreur inattendue lors de la génération de l'introduction et de la conclusion (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            last_error = e
            break  # Catch other unexpected errors

    # If all attempts failed or critical errors occurred
    logging.error(
        f"Impossible de générer l'introduction et la conclusion après {MAX_RETRIES} tentatives. Utilisation des textes par défaut. Dernière erreur: {last_error}"
    )
    return default_intro.strip(), default_conclusion.strip()


def assemble_final_report(
    introduction: str, section_contents: Dict[str, Dict[str, Any]], conclusion: str
) -> str:
    """
    Assemble les différentes parties en un rapport cohérent.
    Étape "Synthesizer" finale de la méthodologie.
    """
    report = "# SYNTHÈSE SOCIOLOGIQUE : L'USAGE DE L'IA CHEZ LES ÉTUDIANTS\n\n"

    # Add introduction
    report += "## Introduction\n\n"
    report += introduction + "\n\n"

    # Add each section with its title and content
    for meta_theme_key, section in section_contents.items():
        # Use the potentially refined title, fallback to original meta key
        title_to_use = section.get("title", meta_theme_key)
        report += f"## {title_to_use}\n\n"
        # Use the final_paragraph from revision, fallback to initial if revision failed, ultimate fallback to a default
        final_paragraph = section.get(
            "final_paragraph",
            section.get(
                "initial_paragraph", "Contenu non disponible en raison d'une erreur."
            ),
        )
        report += final_paragraph + "\n\n"

    # Add conclusion
    report += "## Perspectives\n\n"
    report += conclusion + "\n\n"

    return report


# --- Main Execution ---
def main():
    """Fonction principale pour orchestrer l'analyse qualitative approfondie."""
    logging.info("--- Début du processus d'analyse qualitative ---")

    try:
        api_key = load_api_key()
        groq_client = create_groq_client(api_key)
    except (ValueError, Exception) as e:
        logging.error(
            f"Impossible de continuer sans client API Groq. Arrêt du script. Erreur: {e}"
        )
        sys.exit(1)

    try:
        # Étape 1: Parser - Charger et structurer les données
        dataframe = load_data(CSV_FILE_PATH)
        structured_data = extract_structured_data(dataframe)

        # Étape 2: Planner - Identifier les sections et planifier la structure
        # The planning step now includes API calls with retries and error handling
        sections_with_planning = generate_planning_with_groq(
            groq_client, plan_sections(structured_data)
        )

        # Étape 5 (anticipé): Générer l'introduction et la conclusion d'abord
        # This step now includes API calls with retries and error handling
        intro, conclu = generate_introduction_conclusion_with_groq(
            groq_client, sections_with_planning
        )

        # Étape 3 & 4: Thinker & Critic - Générer et améliorer le contenu section par section
        successful_sections = (
            {}
        )  # Collect sections that completed processing (even with fallbacks)
        for (
            meta_key,
            section,
        ) in sections_with_planning.items():  # Iterate over sections with planning info
            try:
                # Thinker: Générer le contenu initial avec CoT (includes retry/error handling)
                section_with_content = generate_section_content_with_groq(
                    groq_client, section
                )

                # Critic: Réviser et améliorer (includes retry/error handling)
                section_reviewed = review_and_improve_with_groq(
                    groq_client, section_with_content
                )

                # Store the result, whether it was successful API call or fallback
                successful_sections[meta_key] = section_reviewed

                # Logging for success or fallback
                if section_reviewed.get("final_paragraph", "").strip():
                    logging.info(
                        f"Section '{section_reviewed.get('title', meta_key)}' traitée (succès API ou fallback)."
                    )
                else:
                    logging.warning(
                        f"Section '{section_reviewed.get('title', meta_key)}' traitée, mais contenu final vide."
                    )

            except Exception as e:
                # This catch block should ideally not be hit if individual functions handle their errors and fallbacks
                logging.critical(
                    f"Erreur critique non gérée lors du traitement de la section '{meta_key}': {e}"
                )
                import traceback

                traceback.print_exc()
                # Add a basic fallback for this section if something went catastrophically wrong
                section["final_paragraph"] = (
                    f"Traitement de cette section ({section.get('title', meta_key)}) échoué en raison d'une erreur critique."
                )
                section["critique"] = ["Traitement échoué."]
                successful_sections[meta_key] = section

        # If no sections were processed (highly unlikely with current fallbacks)
        if not successful_sections:
            logging.error(
                "Aucune section n'a pu être traitée. Génération d'un rapport minimal."
            )
            # Fallback to a minimal hardcoded report
            final_report = """# SYNTHÈSE SOCIOLOGIQUE : L'USAGE DE L'IA CHEZ LES ÉTUDIANTS

## Introduction

Le processus d'analyse a rencontré des difficultés. Voici un rapport minimal.

## Analyse non disponible

Le contenu détaillé des sections n'a pas pu être généré en raison d'erreurs.

## Perspectives

Les perspectives n'ont pas pu être générées.
"""
        else:
            # Étape 5 (suite): Assembler le rapport final
            final_report = assemble_final_report(intro, successful_sections, conclu)

        # Afficher et sauvegarder le rapport
        print("\n--- SYNTHÈSE NARRATIVE GÉNÉRÉE ---\n")
        print(final_report)

        # Sauvegarde dans un fichier
        try:
            with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(final_report)
            logging.info(f"Synthèse sauvegardée avec succès dans {REPORT_OUTPUT_FILE}")
        except IOError as e:
            logging.error(
                f"Impossible de sauvegarder la synthèse dans {REPORT_OUTPUT_FILE}: {e}"
            )

    except Exception as e:
        logging.critical(f"Erreur fatale lors du processus principal: {e}")
        import traceback

        traceback.print_exc()
        # This catches errors before section processing or if data loading fails.
        # Generate a minimal error report.
        error_report_content = f"""# SYNTHÈSE SOCIOLOGIQUE : Échec de l'analyse

## Erreur Critique

Le processus d'analyse qualitative a rencontré une erreur fatale et n'a pas pu se terminer correctement.

**Détail de l'erreur:**
{str(e)}

Veuillez vérifier les logs et le fichier 'resultats_agent.csv'.
"""
        print("\n--- RAPPORT D'ERREUR MINIMAL ---\n")
        print(error_report_content)
        try:
            with open("rapport_erreur_ia.txt", "w", encoding="utf-8") as f:
                f.write(error_report_content)
            logging.info(
                "Rapport d'erreur minimal sauvegardé dans rapport_erreur_ia.txt"
            )
        except:
            logging.error("Impossible de sauvegarder le rapport d'erreur minimal.")

    logging.info("--- Fin du processus d'analyse qualitative ---")


if __name__ == "__main__":
    main()
