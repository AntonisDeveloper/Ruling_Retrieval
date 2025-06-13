import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import json
import pickle
import heapq
from collections import defaultdict, Counter
import ast
from tqdm import tqdm
import re

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, 'preprocessed_data')

class RulingsRetriever:
    def __init__(self, data_path: str):
        """Initialize the retriever with the data path."""
        print(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        
        # Convert string representation of lists to actual lists
        self.df['articles_mentioned'] = self.df['articles_mentioned'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )
        self.df['full_article_references'] = self.df['full_article_references'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )
        
        # Load precomputed embeddings
        embeddings_path = os.path.join(PREPROCESSED_DIR, 'embeddings.npy')
        print(f"Loading embeddings from: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        
        # Load precomputed graph
        graph_path = os.path.join(PREPROCESSED_DIR, 'graph_file.gpickle')
        print(f"Loading graph from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Compile regex pattern for article matching
        self.article_pattern = re.compile(r'Art\.\s*(\d+[a-z]?)\s+([A-Z]+)', re.IGNORECASE)
        
    def normalize_article(self, article):
        """Normalize article string to standard format."""
        match = self.article_pattern.search(article)
        if match:
            number, law = match.groups()
            return f"Art. {number} {law.upper()}"
        return article.strip()
        
    def vector_search(self, description, top_k=10):
        """Search for similar rulings using vector similarity."""
        if self.embeddings is None:
            print("Error: Embeddings not available")
            return []
        
        print("Performing vector search...")
        # Embed the query description
        query_embedding = self.model.encode(description)
        
        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get indices of top k similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return decision_ids and similarity scores
        results = []
        for idx in top_indices:
            decision_id = self.df.iloc[idx]['decision_id']
            similarity = similarities[idx]
            results.append((decision_id, similarity))
        
        return results
    
    def keyword_search(self, keywords, articles, top_k=10):
        """Search for rulings based on matching keywords and articles."""
        print("Performing keyword search...")
        
        # Normalize input articles
        normalized_articles = [self.normalize_article(art) for art in articles]
        
        # Count exact keyword matches in each document
        keyword_matches = []
        for _, row in self.df.iterrows():
            text = row['full_text_en'].lower() if pd.notna(row['full_text_en']) else ''
            matches = sum(1 for kw in keywords if kw.lower() in text)
            keyword_matches.append(matches)
        
        # Count article matches
        article_matches = []
        for _, row in self.df.iterrows():
            doc_articles = [self.normalize_article(art) for art in row['articles_mentioned']]
            matches = len(set(normalized_articles) & set(doc_articles))
            article_matches.append(matches)
        
        # Combine scores (60% articles, 40% keywords)
        max_keywords = max(keyword_matches) if keyword_matches and max(keyword_matches) > 0 else 1
        max_articles = max(article_matches) if article_matches and max(article_matches) > 0 else 1
        
        combined_scores = np.array([
            0.4 * (k/max_keywords) + 0.6 * (a/max_articles)
            for k, a in zip(keyword_matches, article_matches)
        ])
        
        # Get indices of top k documents
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        # Return decision_ids and scores
        results = []
        for idx in top_indices:
            decision_id = self.df.iloc[idx]['decision_id']
            score = combined_scores[idx]
            results.append((decision_id, score))
        
        return results
    
    def graph_search(self, description, articles, law_area, court, top_k=10, 
                    text_similarity_threshold=0.7, min_total_weight=2.0):
        """Search using graph-based approach with a theoretical node."""
        if self.G is None:
            print("Error: Graph not available")
            return []
            
        print("Performing graph search...")
        
        # Normalize input articles
        normalized_articles = [self.normalize_article(art) for art in articles]
        
        # Create a theoretical node
        theo_node = {
            'description': description,
            'articles': normalized_articles,
            'law_area': law_area,
            'court': court
        }
        
        # Create embeddings for the query if needed for text similarity
        query_embedding = self.model.encode(description)
        
        # Calculate similarity scores with all rulings
        edge_weights = {}
        
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Calculating edge weights"):
            decision_id = row['decision_id']
            weight = 0
            edge_data = {}
            
            # 1. Common articles
            doc_articles = [self.normalize_article(art) for art in row['articles_mentioned']]
            common_articles = set(normalized_articles) & set(doc_articles)
            
            if len(common_articles) >= 2:  # Minimum 2 common articles
                article_weight = len(common_articles) * 1  # Weight of 1 per article
                weight += article_weight
                edge_data['common_articles'] = list(common_articles)
            
            # 2. Same court AND law area
            if row['court'] == court or row['law_area_en'] == law_area:
                court_law_weight = 2
                weight += court_law_weight
                edge_data['same_court_and_law_area'] = True
            
            # 3. Text similarity using embeddings
            doc_embedding_idx = i
            text_sim = cosine_similarity([query_embedding], [self.embeddings[doc_embedding_idx]])[0][0]
            
            if text_sim > text_similarity_threshold:
                text_weight = text_sim * 10  # Weight multiplier for text similarity
                weight += text_weight
                edge_data['text_similarity'] = float(text_sim)
            
            # Only consider edges above the minimum weight threshold
            if weight >= min_total_weight:
                edge_weights[decision_id] = (weight, edge_data)
        
        # First level: Get top-k rulings by direct weight
        first_level = sorted(edge_weights.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        results = [(decision_id, weight) for decision_id, (weight, _) in first_level]
        
        # If we have less than top_k results, perform BFS to find more
        if len(results) < top_k:
            # Get the IDs from the first level
            first_level_ids = set(decision_id for decision_id, _ in results)
            remaining_k = top_k - len(results)
            
            # Find connections from first level results
            second_level_candidates = []
            for first_id, _ in results:
                # Get neighbors of this node in the actual graph
                if first_id in self.G:
                    for neighbor in self.G.neighbors(first_id):
                        if neighbor not in first_level_ids:  # Avoid duplicates
                            # Use edge weight as score
                            weight = self.G[first_id][neighbor]['weight']
                            second_level_candidates.append((neighbor, weight))
            
            # Sort and get top remaining_k from second level
            second_level = sorted(second_level_candidates, key=lambda x: x[1], reverse=True)[:remaining_k]
            results.extend(second_level)
        
        return results
    
    def combined_search(self, description, articles, keywords, law_area, court, top_k=10):
        """Combine results from all three search methods."""
        print("\nRunning combined search with all three methods...")
        
        # Run all three search methods
        vector_results = self.vector_search(description, top_k)
        keyword_results = self.keyword_search(keywords, articles, top_k)
        graph_results = self.graph_search(description, articles, law_area, court, top_k)
        
        print("\nSearch results:")
        
        # Print vector search results
        print("\nVector Search Results (Top 10):")
        print("-" * 50)
        for i, (decision_id, score) in enumerate(vector_results):
            # Get additional metadata
            row = self.df[self.df['decision_id'] == decision_id].iloc[0]
            print(f"{i+1}. Decision ID: {decision_id}")
            print(f"   File Number: {row['file_number']}")
            print(f"   HTML URL: {row['html_url']}")
            print(f"   PDF URL: {row['pdf_url']}")
            print(f"   Score: {score:.4f}")
            print(f"   Court: {row['court']}")
            print(f"   Law Area: {row['law_area']}")
            if not pd.isna(row['year']):
                print(f"   Year: {int(row['year'])}")
            print()
        
        # Print keyword search results
        print("\nKeyword Search Results (Top 10):")
        print("-" * 50)
        for i, (decision_id, score) in enumerate(keyword_results):
            # Get additional metadata
            row = self.df[self.df['decision_id'] == decision_id].iloc[0]
            print(f"{i+1}. Decision ID: {decision_id}")
            print(f"   File Number: {row['file_number']}")
            print(f"   HTML URL: {row['html_url']}")
            print(f"   PDF URL: {row['pdf_url']}")
            print(f"   Score: {score:.4f}")
            print(f"   Court: {row['court']}")
            print(f"   Law Area: {row['law_area']}")
            if not pd.isna(row['year']):
                print(f"   Year: {int(row['year'])}")
            print()
        
        # Print graph search results
        print("\nGraph Search Results (Top 10):")
        print("-" * 50)
        for i, (decision_id, score) in enumerate(graph_results):
            # Get additional metadata
            try:
                row = self.df[self.df['decision_id'] == decision_id].iloc[0]
                print(f"{i+1}. Decision ID: {decision_id}")
                print(f"   File Number: {row['file_number']}")
                print(f"   HTML URL: {row['html_url']}")
                print(f"   PDF URL: {row['pdf_url']}")
                print(f"   Weight: {score:.4f}")
                print(f"   Court: {row['court']}")
                print(f"   Law Area: {row['law_area']}")
                if not pd.isna(row['year']):
                    print(f"   Year: {int(row['year'])}")
            except IndexError:
                print(f"{i+1}. Decision ID: {decision_id} (No metadata available)")
                print(f"   Weight: {score:.4f}")
            print()
        
        return {
            "vector_results": vector_results,
            "keyword_results": keyword_results,
            "graph_results": graph_results
        }

def main():
    # Initialize the retriever
    retriever = RulingsRetriever(data_path='../data/swiss_rulings_with_article_references.csv')
    
    # Example usage
    description = "A 28-year-old individual is stopped during a routine police control in Zurich. During a search, law enforcement finds 1.5 grams of" \
                  "cocaine in his possession. The individual claims it was for personal use and not intended for resale. He has no prior drug convictions but is known to occasionally frequent nightlife venues." \
                  "The individual is arrested and charged with drug possession. The court must decide whether to impose a fine or a prison sentence. The defendant is a Swiss citizen."
    articles_input = "Art. 19 BtMG,Art. 20 BtMG,Art. 15 StGB,Art. 106 BGG"
    keywords_input = "narcotics,cocaine,kg,kilos,drugs,recreational,grams,cocaingenic"
    law_area = "Criminal"
    court = "00ec26f7-275f-4011-9f37-9b2332dfa675"
    
    # Process inputs
    articles = [art.strip() for art in articles_input.split(",")]
    keywords = [kw.strip() for kw in keywords_input.split(",")]
    
    # Run the combined search
    retriever.combined_search(description, articles, keywords, law_area, court)

if __name__ == "__main__":
    main()


"""
A 28-year-old individual is stopped during a routine police control in Zurich. During a search, law enforcement finds 1.5 grams of cocaine in his possession. The individual claims it was for personal use and not intended for resale. He has no prior drug convictions but is known to occasionally frequent nightlife venues. The individual is arrested and charged with drug possession. The court must decide whether to impose a fine or a prison sentence. The defendant is a Swiss citizen.







"""