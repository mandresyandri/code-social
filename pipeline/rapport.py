import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import logging
import os
import sys

# --- Configuration ---
CSV_FILE_PATH = 'resultats_agent.csv'
REPORT_OUTPUT_FILE = 'rapport_analyse_ia.txt' # Optional: file to save the report
GROQ_MODEL = 'llama3-70b-8192' # Or 'llama3-8b-8192' for faster, potentially less detailed results
TEMPERATURE = 0.6 # Controls creativity vs factualness (0=deterministic, 1=max creativity)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Key and Client Functions ---
def load_api_key() -> str:
    """Charge la clé API GROQ depuis .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("La variable d'environnement 'GROQ_API_KEY' n'est pas définie dans le fichier .env")
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
def load_data(filepath: str) -> pd.DataFrame | None:
    """Charge les données depuis un fichier CSV dans un DataFrame Pandas."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Données chargées avec succès depuis {filepath}. Shape: {df.shape}")
        # Basic validation
        expected_columns = ['Méta-thème', 'Thème', 'Code', 'Extrait']
        if not all(col in df.columns for col in expected_columns):
            logging.warning(f"Colonnes attendues non trouvées. Colonnes présentes: {list(df.columns)}")
            # Attempt to proceed if key columns exist, otherwise raise error if critical ones missing
            if 'Méta-thème' not in df.columns or 'Extrait' not in df.columns:
                 raise ValueError("Le fichier CSV doit contenir au moins les colonnes 'Méta-thème' et 'Extrait'.")
        if df.empty:
            logging.warning("Le fichier CSV est vide.")
            return None
        return df
    except FileNotFoundError:
        logging.error(f"Erreur: Le fichier {filepath} n'a pas été trouvé.")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Erreur: Le fichier {filepath} est vide.")
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue lors du chargement de {filepath}: {e}")
        return None

def format_data_for_prompt(df: pd.DataFrame) -> str:
    """Formate les données du DataFrame en une chaîne structurée pour le prompt LLM."""
    if df is None or df.empty:
        return "Aucune donnée à traiter."

    formatted_string = "Voici les données extraites d'entretiens avec des étudiants sur leur usage de l'IA (notamment ChatGPT) dans leurs études, organisées par méta-thèmes, thèmes et codes :\n\n"

    # Group by 'Méta-thème' for better structure
    for meta_theme, group in df.groupby('Méta-thème'):
        formatted_string += f"== Méta-thème : {meta_theme} ==\n"
        # Within each meta-theme, group by 'Thème' (optional, could also list directly)
        for theme, theme_group in group.groupby('Thème'):
             formatted_string += f"  -- Thème : {theme} --\n"
             for _, row in theme_group.iterrows():
                 code = row.get('Code', 'N/A') # Handle potential missing column
                 extrait = row.get('Extrait', 'N/A')
                 formatted_string += f"    Code: {code}\n"
                 formatted_string += f"    Extrait: \"{extrait}\"\n\n" # Add newline for readability
        formatted_string += "\n" # Add space between meta-themes

    return formatted_string

# --- Report Generation Function ---
def generate_report_with_groq(client: Groq, formatted_data: str) -> str | None:
    """Génère un rapport de synthèse en utilisant le client Groq et les données formatées."""
    if not formatted_data or formatted_data == "Aucune donnée à traiter.":
        logging.warning("Aucune donnée fournie pour la génération du rapport.")
        return "Impossible de générer le rapport : aucune donnée d'entrée."

    system_prompt = """
    Tu es un assistant de recherche spécialisé en analyse qualitative.
    Ta tâche est de rédiger un rapport de synthèse basé sur les extraits d'entretiens fournis.
    Ces extraits proviennent d'étudiants discutant de leur utilisation de l'intelligence artificielle (IA), principalement ChatGPT, dans leur parcours académique.
    Le rapport doit :
    1.  Identifier les principaux méta-thèmes et thèmes émergents des données.
    2.  Synthétiser les points clés pour chaque thème majeur, en s'appuyant sur les codes et les extraits fournis comme illustrations (sans forcément les citer tous textuellement, mais en capturant leur essence).
    3.  Mettre en évidence les tensions, les nuances, les stratégies d'usage et les perceptions des étudiants concernant l'IA.
    4.  Adopter un ton objectif et analytique, typique d'un rapport de recherche.
    5.  Structurer le rapport de manière claire et logique (par exemple, par méta-thème).
    Ne te contente pas de lister les extraits, mais analyse et synthétise l'information qu'ils contiennent.
    """

    user_prompt = f"""
    Voici les données d'entretiens à analyser :
    ```
    {formatted_data}
    ```
    Rédige maintenant le rapport de synthèse basé sur ces données, en suivant les instructions fournies.
    """

    logging.info(f"Envoi de la requête à l'API Groq (Modèle: {GROQ_MODEL})...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            # max_tokens=... # Optionnel: Limiter la longueur de la réponse
            # top_p=...     # Optionnel: Autre paramètre de contrôle de la génération
            # stop=...      # Optionnel: Séquences pour arrêter la génération
            # stream=False # Mettre à True pour une réponse en streaming
        )
        report = chat_completion.choices[0].message.content
        logging.info("Rapport généré avec succès par l'API Groq.")
        return report
    except Exception as e:
        logging.error(f"Erreur lors de l'appel à l'API Groq : {e}")
        return None

# --- Main Execution ---
def main():
    """Fonction principale pour orchestrer le chargement, le formatage et la génération du rapport."""
    logging.info("--- Début du processus de génération de rapport ---")

    try:
        api_key = load_api_key()
        groq_client = create_groq_client(api_key)
    except (ValueError, Exception) as e:
        logging.error(f"Impossible de continuer sans client API Groq. Arrêt du script. Erreur: {e}")
        sys.exit(1) # Exit script if API setup fails

    dataframe = load_data(CSV_FILE_PATH)
    if dataframe is None or dataframe.empty:
        logging.error(f"Aucune donnée valide n'a pu être chargée depuis {CSV_FILE_PATH}. Arrêt.")
        sys.exit(1) # Exit script if data loading fails

    formatted_prompt_data = format_data_for_prompt(dataframe)
    if not formatted_prompt_data or formatted_prompt_data == "Aucune donnée à traiter.":
         logging.error("Le formatage des données n'a produit aucune information utile. Arrêt.")
         sys.exit(1)


    generated_report = generate_report_with_groq(groq_client, formatted_prompt_data)

    if generated_report:
        print("\n--- RAPPORT GÉNÉRÉ ---\n")
        print(generated_report)

        # Optionnel : Sauvegarder le rapport dans un fichier
        try:
            with open(REPORT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("--- RAPPORT D'ANALYSE SUR L'USAGE DE L'IA PAR LES ÉTUDIANTS ---\n\n")
                f.write(f"Source des données : {CSV_FILE_PATH}\n")
                f.write(f"Modèle IA utilisé : {GROQ_MODEL}\n")
                f.write(f"Date de génération : {pd.Timestamp.now()}\n")
                f.write("-" * 50 + "\n\n")
                f.write(generated_report)
            logging.info(f"Rapport sauvegardé avec succès dans {REPORT_OUTPUT_FILE}")
        except IOError as e:
            logging.error(f"Impossible de sauvegarder le rapport dans {REPORT_OUTPUT_FILE}: {e}")
    else:
        logging.error("La génération du rapport a échoué.")
        sys.exit(1) # Exit with error status if report generation failed

    logging.info("--- Fin du processus ---")

if __name__ == "__main__":
    main()