# Court Rulings Search Application

A full-stack application for searching and analyzing Swiss court rulings, built with React and Flask.

## Project Structure

```
.
├── frontend/           # React frontend
├── src/               # Python backend
├── data/              # Data files
└── requirements.txt   # Python dependencies
```

## Local Development

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend server:
   ```bash
   python src/search_rulings_api.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm start
   ```

## Deployment

### Backend (Heroku)
1. Create a new Heroku app
2. Add the Heroku remote:
   ```bash
   heroku git:remote -a your-app-name
   ```
3. Deploy:
   ```bash
   git push heroku main
   ```

### Frontend (Vercel)
1. Connect your GitHub repository to Vercel
2. Configure the build settings:
   - Build Command: `cd frontend && npm install && npm run build`
   - Output Directory: `frontend/build`
3. Add environment variables:
   - `REACT_APP_API_URL`: Your Heroku backend URL

## Environment Variables

### Backend
- `PORT`: Port number (default: 5000)

### Frontend
- `REACT_APP_API_URL`: Backend API URL

## API Endpoints

- `POST /api/search`: Search for court rulings
  - Parameters:
    - `query`: Search text
    - `articles`: List of article references
    - `law_area`: Law area filter
    - `court`: Court filter
    - `num_results`: Number of results to return 