import React, { useState } from 'react';
import './App.css';

interface SearchResult {
  decision_id: string;
  title: string;
  court: string;
  year: number;
  law_area: string;
  html_url: string;
  pdf_url: string;
  articles_mentioned: string[];
  sources: string[];
}

interface SearchResponse {
  results: SearchResult[];
  stats: {
    vector_results: number;
    keyword_results: number;
    graph_results: number;
    total_unique: number;
  };
  sources: Record<string, string[]>;
}

function App() {
  const [query, setQuery] = useState('');
  const [articles, setArticles] = useState('');
  const [lawArea, setLawArea] = useState('');
  const [court, setCourt] = useState('');
  const [numResults, setNumResults] = useState(10);
  const [useVector, setUseVector] = useState(true);
  const [useKeyword, setUseKeyword] = useState(true);
  const [useGraph, setUseGraph] = useState(true);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      // Parse articles string into array
      const articlesArray = articles
        .split(',')
        .map(article => article.trim())
        .filter(article => article.length > 0);

      // Call the Python backend
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          articles: articlesArray,
          law_area: lawArea,
          court: court,
          num_results: numResults,
          use_vector: useVector,
          use_keyword: useKeyword,
          use_graph: useGraph,
        }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Swiss Court Rulings Search</h1>
      </header>

      <main className="App-main">
        <div className="search-form">
          <div className="form-group">
            <label htmlFor="query">Search Query:</label>
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your search query"
            />
          </div>

          <div className="form-group">
            <label htmlFor="articles">Article References (comma-separated):</label>
            <input
              id="articles"
              type="text"
              value={articles}
              onChange={(e) => setArticles(e.target.value)}
              placeholder="e.g., Art. 95 BGG, Art. 96 ZGB"
            />
          </div>

          <div className="form-group">
            <label htmlFor="lawArea">Law Area:</label>
            <input
              id="lawArea"
              type="text"
              value={lawArea}
              onChange={(e) => setLawArea(e.target.value)}
              placeholder="e.g., Criminal, Civil"
            />
          </div>

          <div className="form-group">
            <label htmlFor="court">Court:</label>
            <input
              id="court"
              type="text"
              value={court}
              onChange={(e) => setCourt(e.target.value)}
              placeholder="e.g., Federal Supreme Court"
            />
          </div>

          <div className="form-group">
            <label htmlFor="numResults">Number of Results:</label>
            <input
              id="numResults"
              type="number"
              value={numResults}
              onChange={(e) => setNumResults(Number(e.target.value))}
              min="1"
              max="100"
            />
          </div>

          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={useVector}
                onChange={(e) => setUseVector(e.target.checked)}
              />
              Use Vector Search
            </label>
            <label>
              <input
                type="checkbox"
                checked={useKeyword}
                onChange={(e) => setUseKeyword(e.target.checked)}
              />
              Use Keyword Search
            </label>
            <label>
              <input
                type="checkbox"
                checked={useGraph}
                onChange={(e) => setUseGraph(e.target.checked)}
              />
              Use Graph Search
            </label>
          </div>

          <button 
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="search-button"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {results && (
          <div className="results-container">
            <div className="stats">
              <h2>Search Statistics</h2>
              <p>Vector Results: {results.stats.vector_results}</p>
              <p>Keyword Results: {results.stats.keyword_results}</p>
              <p>Graph Results: {results.stats.graph_results}</p>
              <p>Total Unique Results: {results.stats.total_unique}</p>
            </div>

            <div className="results-list">
              <h2>Search Results</h2>
              {results.results.map((result) => (
                <div key={result.decision_id} className="result-card">
                  <h3>{result.title}</h3>
                  <div className="result-meta">
                    <span>Court: {result.court}</span>
                    <span>Year: {result.year}</span>
                    <span>Law Area: {result.law_area}</span>
                  </div>
                  <div className="result-sources">
                    Found by: {result.sources.join(', ')}
                  </div>
                  <div className="result-articles">
                    Articles: {result.articles_mentioned.join(', ')}
                  </div>
                  <div className="result-urls">
                    <div>HTML URL: {result.html_url}</div>
                  </div>
                  <div className="result-links">
                    <a href={result.html_url} target="_blank" rel="noopener noreferrer" className="link-button">View HTML</a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 