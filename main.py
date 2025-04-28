import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import re
from typing import Dict, List, Any, Optional


def load_api_key():
    """Charge la clé API depuis les variables d'environnement."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("La variable d'environnement 'GROQ_API_KEY' n'est pas définie dans .env")
    return groq_api_key


def create_groq_client(api_key):
    """Crée une instance du client Groq."""
    return Groq(api_key=api_key)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un fichier PDF."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte: {e}")
    return text


def clean_text(text: str) -> str:
    """Nettoie le texte extrait."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def generate_summary_prompt(content: str, **kwargs) -> str:
    """Génère un prompt pour résumer le contenu."""
    instructions = kwargs.get('instructions', 
                          "Résume ce document en identifiant les points principaux.")
    focus = kwargs.get('focus', '')
    
    prompt = instructions
    if focus:
        prompt += f" Concentre-toi particulièrement sur: {focus}."
    
    prompt += f"\n\nContenu du document:\n{content}"
    return prompt


def generate_completion(client, model, messages, **kwargs):
    """Génère une complétion en utilisant le client Groq."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get('temperature', 0.7),
        max_completion_tokens=kwargs.get('max_tokens', 1024),
        top_p=kwargs.get('top_p', 1),
        stream=kwargs.get('stream', True),
        stop=kwargs.get('stop', None),
    )


def print_completion(completion):
    """Affiche la complétion générée par le modèle."""
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


def summarize_pdf(pdf_path: str, model: str = "meta-llama/llama-4-maverick-17b-128e-instruct", **kwargs):
    """Résume le contenu d'un fichier PDF avec des paramètres personnalisables."""
    # Validation du fichier
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier PDF n'existe pas: {pdf_path}")
    
    api_key = load_api_key()
    client = create_groq_client(api_key)
    
    # Extraction et nettoyage du texte
    pdf_content = extract_text_from_pdf(pdf_path)
    cleaned_content = clean_text(pdf_content)
    
    # Tronquer si nécessaire
    max_length = kwargs.get('max_content_length', 10000)
    if len(cleaned_content) > max_length:
        cleaned_content = cleaned_content[:max_length] + "...[texte tronqué]"
    
    # Génération du prompt et des messages
    prompt = generate_summary_prompt(cleaned_content, **kwargs)
    messages = [{"role": "user", "content": prompt}]
    
    # Génération de la réponse
    return generate_completion(client, model, messages, **kwargs)


def main():
    """Point d'entrée principal."""
    try:
        pdf_path = input("Chemin vers le fichier PDF: ")
        
        # Options personnalisées
        kwargs = {}
        custom_instructions = input("Instructions spécifiques (optionnel): ")
        if custom_instructions:
            kwargs['instructions'] = custom_instructions
            
        focus_areas = input("Éléments à mettre en avant (optionnel): ")
        if focus_areas:
            kwargs['focus'] = focus_areas
        
        # Génération et affichage du résumé
        completion = summarize_pdf(pdf_path, **kwargs)
        print("\n--- Résumé du document ---\n")
        print_completion(completion)
        
    except Exception as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()
