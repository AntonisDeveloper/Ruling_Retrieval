const express = require('express');
const path = require('path');
const cors = require('cors');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Serve static files from the React app
app.use(express.static(path.join(__dirname, 'build')));

app.post('/api/search', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5000/api/search', req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error calling Python backend:', error);
    res.status(500).json({ error: error.message || 'Failed to get response from search service' });
  }
});

// The "catchall" handler: for any request that doesn't
// match one above, send back React's index.html file.
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(port, () => {
  console.log(`Frontend server is running on port ${port}`);
}); 