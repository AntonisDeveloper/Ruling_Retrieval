import os
import sys
import json
import pandas as pd
import pickle
from typing import Dict, List, Union, Optional
from retrieve_court_rulings import RulingsRetriever
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'swiss_rulings_with_article_references.csv')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class CourtRulingsAPI:
    def __init__(self):
        """Initialize the API with the RulingsRetriever."""
        self.retriever = RulingsRetriever(DATA_PATH)
        
    def search(self, 
              query: str,
              articles: Optional[List[str]] = None,
              law_area: str = '',
              court: str = '',
              num_results: int = 10,
              use_vector: bool = True,
              use_keyword: bool = True,
              use_graph: bool = True) -> Dict:
        """
        Perform a search across all methods and return combined results.
        
        Args:
            query: Search query text
            articles: List of article references to search for
            law_area: Law area to search for
            court: Court to search for
            num_results: Number of results to return per method
            use_vector: Whether to use vector search
            use_keyword: Whether to use keyword search
            use_graph: Whether to use graph search
            
        Returns:
            Dict containing:
            - results: List of rulings with metadata
            - stats: Statistics about the search
            - sources: Which search methods found each result
        """
        all_results = []
        sources = {}
        stats = {
            "vector_results": 0,
            "keyword_results": 0,
            "graph_results": 0,
            "total_unique": 0
        }
        df = self.retriever.df

        # Helper to get full ruling info as dict
        def get_ruling_dict(decision_id):
            try:
                row = df[df['decision_id'] == decision_id].iloc[0]
                # Convert NaN values to None for JSON serialization
                ruling_dict = {
                    "decision_id": str(row['decision_id']),
                    "title": str(row.get('title', '')) if pd.notna(row.get('title')) else '',
                    "court": str(row.get('court', '')) if pd.notna(row.get('court')) else '',
                    "year": None if pd.isna(row.get('year')) else int(row.get('year')),
                    "law_area": str(row.get('law_area_en', '')) if pd.notna(row.get('law_area_en')) else '',
                    "html_url": str(row.get('html_url', '')) if pd.notna(row.get('html_url')) else None,
                    "articles_mentioned": list(row.get('articles_mentioned', [])) if isinstance(row.get('articles_mentioned'), (list, np.ndarray)) else [],
                }
                # Print debug info
                print(f"Found ruling {decision_id} with URL:")
                print(f"HTML URL: {ruling_dict['html_url']}")
                return ruling_dict
            except Exception as e:
                print(f"Error getting ruling dict for {decision_id}: {str(e)}")
                return None

        try:
            # Vector search
            if use_vector:
                vector_results = self.retriever.vector_search(query, num_results)
                stats["vector_results"] = len(vector_results)
                for decision_id, _ in vector_results:
                    if decision_id not in sources:
                        sources[decision_id] = []
                    sources[decision_id].append("vector")
                    all_results.append(decision_id)

            # Keyword search
            if use_keyword:
                keyword_results = self.retriever.keyword_search([query], articles or [], num_results)
                stats["keyword_results"] = len(keyword_results)
                for decision_id, _ in keyword_results:
                    if decision_id not in sources:
                        sources[decision_id] = []
                    sources[decision_id].append("keyword")
                    all_results.append(decision_id)

            # Graph search
            if use_graph:
                graph_results = self.retriever.graph_search(query, articles or [], law_area, court, num_results)
                stats["graph_results"] = len(graph_results)
                for decision_id, _ in graph_results:
                    if decision_id not in sources:
                        sources[decision_id] = []
                    sources[decision_id].append("graph")
                    all_results.append(decision_id)

            # Remove duplicates, preserve order
            seen = set()
            unique_results = []
            for decision_id in all_results:
                if decision_id not in seen:
                    seen.add(decision_id)
                    ruling = get_ruling_dict(decision_id)
                    if ruling is not None:  # Only add if we successfully got the ruling
                        ruling["sources"] = sources[decision_id]
                        unique_results.append(ruling)
            stats["total_unique"] = len(unique_results)

            return {
                "results": unique_results,
                "stats": stats,
                "sources": sources
            }
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {
                "results": [],
                "stats": stats,
                "sources": {},
                "error": str(e)
            }

# Initialize the API
api = CourtRulingsAPI()

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query', '')
        articles = data.get('articles', [])
        law_area = data.get('law_area', '')
        court = data.get('court', '')
        num_results = int(data.get('num_results', 10))
        use_vector = data.get('use_vector', True)
        use_keyword = data.get('use_keyword', True)
        use_graph = data.get('use_graph', True)
        
        results = api.search(
            query=query,
            articles=articles,
            law_area=law_area,
            court=court,
            num_results=num_results,
            use_vector=use_vector,
            use_keyword=use_keyword,
            use_graph=use_graph
        )
        return jsonify(results)
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "results": [],
            "stats": {
                "vector_results": 0,
                "keyword_results": 0,
                "graph_results": 0,
                "total_unique": 0
            },
            "sources": {}
        }), 500

if __name__ == "__main__":
    print(f"Starting server with data from: {DATA_PATH}")
    app.run(host='0.0.0.0', port=5000) 