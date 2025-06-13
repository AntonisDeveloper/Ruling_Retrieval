import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
from tqdm import tqdm
import json
import pickle
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import ast

class RulingsPreprocessor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Convert string representation of lists to actual lists
        self.df['articles_mentioned'] = self.df['articles_mentioned'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )
        self.df['full_article_references'] = self.df['full_article_references'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )
        self.keyword_index = defaultdict(list)
        self.graph = nx.Graph()
        self.embeddings = None
        self.article_pattern = re.compile(r'art\.?\s*\d+', re.IGNORECASE)
        self.feature_names = None  # Store feature names for analysis
        
    def extract_articles(self, text):
        """Extract article numbers from text."""
        if pd.isna(text):
            return []
        articles = self.article_pattern.findall(text)
        # Clean up article numbers (remove 'art.' and spaces)
        return [re.sub(r'art\.?\s*', '', art).strip() for art in articles]
    
    def analyze_indexed_terms(self, num_examples=10):
        """Analyze and print examples of indexed terms."""
        if self.feature_names is None or len(self.feature_names) == 0:
            print("No terms indexed yet. Run build_keyword_index first.")
            return
            
        # Get examples of single words and bigrams
        single_words = [term for term in self.feature_names if ' ' not in term]
        bigrams = [term for term in self.feature_names if ' ' in term]
        
        print("\nIndexed Terms Analysis:")
        print(f"Total number of terms: {len(self.feature_names)} (max_features limit: 10000)")
        print(f"Number of single words (unigrams): {len(single_words)}")
        print(f"Number of word pairs (bigrams): {len(bigrams)}")
        print(f"Percentage of unigrams: {(len(single_words)/len(self.feature_names))*100:.1f}%")
        print(f"Percentage of bigrams: {(len(bigrams)/len(self.feature_names))*100:.1f}%")
        
        print("\nExample Single Words:")
        for word in single_words[:num_examples]:
            print(f"- {word}")
            
        print("\nExample Bigrams:")
        for bigram in bigrams[:num_examples]:
            print(f"- {bigram}")
            
        # Show some statistics about term frequencies
        print("\nTerm Frequency Statistics:")
        term_freqs = {term: len(docs) for term, docs in self.keyword_index.items()}
        sorted_terms = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMost Common Single Words:")
        for term, freq in sorted_terms[:num_examples]:
            if ' ' not in term:
                print(f"- {term}: appears in {freq} documents")
                
        print("\nMost Common Bigrams:")
        for term, freq in sorted_terms[:num_examples]:
            if ' ' in term:
                print(f"- {term}: appears in {freq} documents")
                
        # Show distribution of term frequencies
        print("\nTerm Frequency Distribution:")
        freq_ranges = [(1, 5), (6, 20), (21, 50), (51, 100), (101, float('inf'))]
        for start, end in freq_ranges:
            unigrams_in_range = sum(1 for term, freq in term_freqs.items() 
                                  if ' ' not in term and start <= freq <= end)
            bigrams_in_range = sum(1 for term, freq in term_freqs.items() 
                                 if ' ' in term and start <= freq <= end)
            print(f"Terms appearing {start}-{end if end != float('inf') else '∞'} times:")
            print(f"  - Unigrams: {unigrams_in_range}")
            print(f"  - Bigrams: {bigrams_in_range}")
    
    def build_keyword_index(self):
        """Build inverted index for keyword search."""
        print("Building keyword index...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # This means both single words and pairs of words
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9]*\b'  # Only words starting with letters
        )
        
        # Fit and transform the text
        tfidf_matrix = vectorizer.fit_transform(self.df['full_text_en'].fillna(''))
        
        # Get feature names (words/phrases)
        self.feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF scores for each term
        avg_scores = {}
        for i, text in enumerate(tqdm(self.df['full_text_en'])):
            if pd.isna(text):
                continue
                
            # Get non-zero elements for this document
            doc_vector = tfidf_matrix[i]
            nonzero_indices = doc_vector.nonzero()[1]
            
            # Get the decision_id for this document
            decision_id = self.df.iloc[i]['decision_id']
            
            # Add to inverted index and track scores
            for idx in nonzero_indices:
                word = self.feature_names[idx]
                score = float(doc_vector[0, idx])
                
                # Track average scores
                if word not in avg_scores:
                    avg_scores[word] = {'sum': 0, 'count': 0}
                avg_scores[word]['sum'] += score
                avg_scores[word]['count'] += 1
                
                self.keyword_index[word].append({
                    'decision_id': decision_id,  # Using decision_id instead of file_number
                    'score': score
                })
        
        # Calculate and print statistics about term importance
        print("\nTerm Importance Analysis:")
        unigram_scores = [(term, stats['sum']/stats['count']) 
                         for term, stats in avg_scores.items() 
                         if ' ' not in term]
        bigram_scores = [(term, stats['sum']/stats['count']) 
                        for term, stats in avg_scores.items() 
                        if ' ' in term]
        
        # Sort by average TF-IDF score
        unigram_scores.sort(key=lambda x: x[1], reverse=True)
        bigram_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Most Important Unigrams (by average TF-IDF score):")
        for term, score in unigram_scores[:10]:
            print(f"- {term}: {score:.4f}")
            
        print("\nTop 10 Most Important Bigrams (by average TF-IDF score):")
        for term, score in bigram_scores[:10]:
            print(f"- {term}: {score:.4f}")
        
        # Sort by score for each word in the index
        for word in self.keyword_index:
            self.keyword_index[word].sort(key=lambda x: x['score'], reverse=True)
            
        # Analyze the indexed terms
        self.analyze_indexed_terms()
    
    def build_graph(self, min_common_articles=2, text_similarity_threshold=0.8, min_total_weight=2.0):
        """Build graph where nodes are rulings and edges represent legal relationships."""
        print("Building graph...")
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Add nodes with metadata
        for i, row in self.df.iterrows():
            self.graph.add_node(row['decision_id'], 
                              title=row['decision_id'],
                              court=row['court'],
                              year=row['year'],
                              law_area=row['law_area'])
        
        # Load pre-computed embeddings
        print("Loading pre-computed embeddings...")
        try:
            self.embeddings = np.load('preprocessed_data/embeddings.npy')
            print("Successfully loaded embeddings")
        except FileNotFoundError:
            print("Warning: Embeddings file not found. Building embeddings first...")
            self.build_embeddings()
        
        # Compute cosine similarity matrix from embeddings
        print("Computing text similarity from embeddings...")
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # Initialize weight counters for each criterion
        total_weight = 0
        criterion_weights = {
            'common_articles': 0,
            'same_court_and_law_area': 0,
            'text_similarity': 0
        }
        
        # Create connections based on multiple criteria
        for i in tqdm(range(len(self.df))):
            row_i = self.df.iloc[i]
            
            for j in range(i + 1, len(self.df)):
                row_j = self.df.iloc[j]
                
                # Initialize edge weight and metadata
                weight = 0
                edge_metadata = {}
                
                # 1. Common articles (legal references)
                articles_i = set(row_i['articles_mentioned'])
                articles_j = set(row_j['articles_mentioned'])
                common_articles = articles_i & articles_j
                if len(common_articles) >= min_common_articles:  # Only if minimum number of common articles
                    article_weight = len(common_articles) * 2
                    weight += article_weight
                    criterion_weights['common_articles'] += article_weight
                    edge_metadata['common_articles'] = list(common_articles)
                
                # 2. Same court AND same law area
                if row_i['court'] == row_j['court'] and row_i['law_area'] == row_j['law_area']:
                    year_diff = abs(row_i['year'] - row_j['year'])
                    if year_diff <= 2:  # Still keep the temporal proximity
                        court_law_weight = 2
                        weight += court_law_weight
                        criterion_weights['same_court_and_law_area'] += court_law_weight
                        edge_metadata['same_court_and_law_area'] = True
                        edge_metadata['year_diff'] = year_diff
                
                # 3. Text similarity using pre-computed embeddings
                text_sim = similarity_matrix[i, j]
                if text_sim > text_similarity_threshold:
                    text_weight = text_sim * 5
                    weight += text_weight
                    criterion_weights['text_similarity'] += text_weight
                    edge_metadata['text_similarity'] = float(text_sim)
                
                # Add edge only if total weight exceeds minimum threshold
                if weight >= min_total_weight:
                    self.graph.add_edge(row_i['decision_id'], 
                                      row_j['decision_id'],
                                      weight=weight,
                                      **edge_metadata)
                    total_weight += weight
        
        # Print graph statistics
        print("\nGraph Statistics:")
        print(f"Number of vertices (nodes): {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        
        # Calculate and print average degree
        avg_degree = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        print(f"Average edges per vertex: {avg_degree:.2f}")
        
        # Analyze connected components
        components = list(nx.connected_components(self.graph))
        print(f"\nNumber of connected components: {len(components)}")
        
        # Analyze component sizes
        component_sizes = [len(comp) for comp in components]
        print("\nComponent Size Distribution:")
        print(f"Largest component: {max(component_sizes)} vertices")
        print(f"Smallest component: {min(component_sizes)} vertices")
        print(f"Average component size: {sum(component_sizes)/len(component_sizes):.2f} vertices")
        
        # Count isolated vertices (components of size 1)
        isolated_vertices = sum(1 for comp in components if len(comp) == 1)
        print(f"Number of isolated vertices: {isolated_vertices}")
        
        # Print weight distribution by criterion
        print("\nWeight Distribution by Criterion:")
        for criterion, weight in criterion_weights.items():
            percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
            print(f"{criterion}: {weight:.2f} ({percentage:.1f}%)")
        
        # Print connection type distribution
        print("\nConnection Type Distribution (Number of Edges):")
        connection_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            if 'common_articles' in data:
                connection_types['legal_references'] += 1
            if data.get('same_court_and_law_area', False):
                connection_types['same_court_and_law_area'] += 1
            if 'text_similarity' in data:
                connection_types['text_similarity'] += 1
        
        total_edges = self.graph.number_of_edges()
        for conn_type, count in connection_types.items():
            percentage = (count / total_edges) * 100
            print(f"{conn_type}: {count} edges ({percentage:.1f}% of total edges)")

                # Save graph
        print("Saving graph...")
        pickle.dump(self.graph, open("graph_file.gpickle", "wb"))

    def visualize_graph(self, output_file='rulings_graph.png', max_nodes=1000):
        """Visualize the graph with a subset of nodes for clarity."""
        print("Visualizing graph...")
        
        # Get the largest connected component
        largest_cc = max(nx.connected_components(self.graph), key=len)
        
        # If the component is too large, take a random sample
        if len(largest_cc) > max_nodes:
            nodes_to_plot = set(np.random.choice(list(largest_cc), max_nodes, replace=False))
        else:
            nodes_to_plot = largest_cc
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_plot)
        
        # Set up the plot
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color='lightblue',
                             node_size=100,
                             alpha=0.6)
        
        # Draw edges with varying widths based on weight
        edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
        nx.draw_networkx_edges(subgraph, pos,
                             width=[w/2 for w in edge_weights],
                             alpha=0.4)
        
        # Add labels
        nx.draw_networkx_labels(subgraph, pos,
                              font_size=8,
                              font_family='sans-serif')
        
        plt.title("Court Rulings Graph\n(Showing largest connected component)")
        plt.axis('off')
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
    
    def build_embeddings(self):
        """Create embeddings using Sentence-BERT with chunking for long texts."""
        print("Creating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = model.tokenizer
        max_tokens = 512  # Maximum tokens per chunk
        
        # Prepare texts for embedding
        texts = self.df['full_text_en'].fillna('').tolist()
        final_embeddings = []
        
        print("Processing documents and creating chunked embeddings...")
        for text in tqdm(texts):
            if not text:
                # If text is empty, use zero vector of correct dimension
                final_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
                continue
                
            # Tokenize the text
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= max_tokens:
                # If text is short enough, embed directly
                embedding = model.encode(text, convert_to_numpy=True)
                final_embeddings.append(embedding)
            else:
                # Split into chunks and process each chunk
                chunks = []
                chunk_weights = []
                
                # Create chunks of max_tokens length
                for i in range(0, len(tokens), max_tokens):
                    chunk_tokens = tokens[i:i + max_tokens]
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    chunks.append(chunk_text)
                    chunk_weights.append(len(chunk_tokens))
                
                # Get embeddings for each chunk
                chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
                
                # Calculate weighted average
                weights = np.array(chunk_weights) / sum(chunk_weights)
                weighted_embedding = np.average(chunk_embeddings, weights=weights, axis=0)
                final_embeddings.append(weighted_embedding)
        
        # Convert to numpy array
        self.embeddings = np.array(final_embeddings)
        
        # Print statistics about the embeddings
        print("\nEmbedding Statistics:")
        print(f"Total documents processed: {len(texts)}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print(f"Number of documents requiring chunking: {sum(1 for text in texts if text and len(tokenizer.encode(text)) > max_tokens)}")
        
        # Save embeddings
        print("\nSaving embeddings...")
        os.makedirs('preprocessed_data', exist_ok=True)
        np.save('preprocessed_data/embeddings.npy', self.embeddings)
        
        # Save metadata about chunking
        metadata = {
            'embedding_dim': self.embeddings.shape[1],
            'max_tokens_per_chunk': max_tokens,
            'total_documents': len(texts),
            'chunked_documents': sum(1 for text in texts if text and len(tokenizer.encode(text)) > max_tokens)
        }
        with open('preprocessed_data/embedding_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_all(self, output_dir='preprocessed_data'):
        """Save all preprocessed data."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save keyword index
        print("Saving keyword index...")
        with open(f'{output_dir}/keyword_index.json', 'w') as f:
            json.dump(self.keyword_index, f)
        
        # Save graph
        print("Saving graph...")
        nx.write_gpickle(self.graph, f'{output_dir}/rulings_graph.gpickle')
        
        # Save embeddings
        print("Saving embeddings...")
        np.save(f'{output_dir}/embeddings.npy', self.embeddings)
        
        # Save metadata
        print("Saving metadata...")
        metadata = {
            'num_documents': len(self.df),
            'num_keywords': len(self.keyword_index),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else None
        }
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f)

def main():
    # Initialize preprocessor with the new CSV file
    preprocessor = RulingsPreprocessor('data/swiss_rulings_with_article_references.csv')
    
    # # Build keyword index
    # print("\n=== Building Keyword Index ===")
    # preprocessor.build_keyword_index()
    
    # # Build embeddings
    # print("\n=== Building Embeddings ===")
    # preprocessor.build_embeddings()
    
    # Build graph
    print("\n=== Building Graph ===")
    preprocessor.build_graph()
    
    # Visualize graph
    print("\n=== Visualizing Graph ===")
    preprocessor.visualize_graph()
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 