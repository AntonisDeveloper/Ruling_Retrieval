import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_embed_rulings(csv_path, output_json_path):
    """
    Load court rulings from CSV, embed them using Sentence-BERT, and save to JSON
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize the Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get the text content (assuming there's a 'text' column)
    texts = df['text'].tolist()
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Create a dictionary with rulings and their embeddings
    rulings_data = {
        'rulings': [
            {
                'id': str(idx),
                'text': text,
                'embedding': embedding.tolist()
            }
            for idx, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
    }
    
    # Save to JSON
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
            'id': rulings_data['rulings'][idx]['id'],
            'text': rulings_data['rulings'][idx]['text'],
            'similarity_score': float(similarities[idx])
        })
    
    return similar_rulings

def main():
    # Paths
    csv_path = 'court_rulings.csv'  # Update this with your CSV file path
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
        print(f"Ruling ID: {ruling['id']}")
        print(f"Text: {ruling['text'][:200]}...")  # Print first 200 characters

if __name__ == "__main__":
    main() 