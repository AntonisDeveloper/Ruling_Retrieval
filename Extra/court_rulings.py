import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import time
from datetime import datetime

def translate_text(text, source_lang, target_lang='en'):
    """
    Translate text from source language to target language
    """
    if pd.isna(text) or text == '':
        return ''
    
    try:
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def load_and_embed_rulings(csv_path, output_json_path):
    """
    Load court rulings from CSV, embed them using Sentence-BERT, and save to JSON
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to readable date
    df['date_readable'] = pd.to_datetime(df['date'].astype(float), unit='ms').dt.strftime('%Y-%m-%d')
    
    # Add translated columns
    print("Translating law areas...")
    df['law_area_en'] = df.apply(lambda row: translate_text(row['law_area'], row['language']), axis=1)
    
    print("Translating full texts...")
    df['full_text_en'] = df.apply(lambda row: translate_text(row['full_text'], row['language']), axis=1)
    
    # Save the updated CSV with translations
    df.to_csv(csv_path.replace('.csv', '_with_translations.csv'), index=False)
    print("Saved CSV with translations")
    
    # Initialize the Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get the text content (using translated text)
    texts = df['full_text_en'].tolist()
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts)
    
    # Create a dictionary with rulings and their embeddings
    rulings_data = {
        'rulings': [
            {
                'decision_id': row['decision_id'],
                'date': row['date_readable'],
                'year': str(row['year']),
                'text': text,
                'text_original': original_text,
                'law_area': law_area,
                'law_area_en': law_area_en,
                'language': lang,
                'num_tokens': str(num_tokens),
                'embedding': embedding.tolist()
            }
            for _, row in df.iterrows()
            for text, original_text, law_area, law_area_en, lang, num_tokens, embedding in [
                (row['full_text_en'], row['full_text'], row['law_area'], 
                 row['law_area_en'], row['language'], row['full_text_num_tokens_bert'],
                 embeddings[df.index.get_loc(row.name)])
            ]
        ]
    }
    
    # Save to JSON
    print("Saving embeddings to JSON...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(rulings_data, f, ensure_ascii=False, indent=2)
    
    return rulings_data

def find_similar_rulings(query_text, rulings_data, top_k=5):
    """
    Find the top-k most similar rulings to the query text
    """
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get query embedding
    query_embedding = model.encode([query_text])[0]
    
    # Get all ruling embeddings
    ruling_embeddings = np.array([ruling['embedding'] for ruling in rulings_data['rulings']])
    
    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], ruling_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return top-k similar rulings with their similarity scores
    similar_rulings = []
    for idx in top_indices:
        similar_rulings.append({
            'decision_id': rulings_data['rulings'][idx]['decision_id'],
            'date': rulings_data['rulings'][idx]['date'],
            'year': rulings_data['rulings'][idx]['year'],
            'text': rulings_data['rulings'][idx]['text'],
            'text_original': rulings_data['rulings'][idx]['text_original'],
            'law_area': rulings_data['rulings'][idx]['law_area'],
            'law_area_en': rulings_data['rulings'][idx]['law_area_en'],
            'language': rulings_data['rulings'][idx]['language'],
            'num_tokens': rulings_data['rulings'][idx]['num_tokens'],
            'similarity_score': float(similarities[idx])
        })
    
    return similar_rulings

def main():
    # Paths
    csv_path = 'rulings.csv'  # Update this with your CSV file path
    json_path = 'court_rulings_embeddings.json'
    
    # Load and embed rulings
    print("Loading and embedding court rulings...")
    rulings_data = load_and_embed_rulings(csv_path, json_path)
    print(f"Embeddings saved to {json_path}")
    
    # Example query
    query = "Example query text"  # Replace with your query
    print(f"\nFinding similar rulings for query: {query}")
    similar_rulings = find_similar_rulings(query, rulings_data)
    
    # Print results
    print("\nTop 5 similar rulings:")
    for i, ruling in enumerate(similar_rulings, 1):
        print(f"\n{i}. Similarity Score: {ruling['similarity_score']:.4f}")
        print(f"Decision ID: {ruling['decision_id']}")
        print(f"Date: {ruling['date']}")
        print(f"Year: {ruling['year']}")
        print(f"Language: {ruling['language']}")
        print(f"Law Area (Original): {ruling['law_area']}")
        print(f"Law Area (English): {ruling['law_area_en']}")
        print(f"Number of Tokens: {ruling['num_tokens']}")
        print(f"Text (English): {ruling['text'][:200]}...")
        print(f"Text (Original): {ruling['text_original'][:200]}...")

if __name__ == "__main__":
    main() 