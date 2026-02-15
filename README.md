# TerrainTiles
Demos & utils for terrain tiles manipulation

A simple Flask application for terrain tiles processing and manipulation.

## Directory Structure

```
TerrainTiles/
├── lib/           # Library code for terrain tiles manipulation
├── Temp/          # Temporary directory for operation in/out dumping
├── demos/         # Sample input files for testing
├── app.py         # Main Flask application
└── requirements.txt
```

## Setup Instructions

### 1. Create and Activate Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode

Start the Flask application:

```bash
python app.py
```

The application will start on `http://localhost:5000`

**Note:** The application runs in debug mode by default, which is suitable for development but should **NOT** be used in production. Debug mode can expose sensitive information and security vulnerabilities.

### Production Deployment

For production environments:
1. Set `debug=False` in `app.py`
2. Use a production WSGI server like gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Making Requests

### Health Check

Check if the API is running:

```bash
curl http://localhost:5000/health
```

### Get API Information

```bash
curl http://localhost:5000/
```

### Process a Terrain Tile

Using curl with inline JSON:

```bash
curl -X POST http://localhost:5000/tile \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sample_tile",
    "coordinates": {
      "lat": 40.7128,
      "lon": -74.0060
    },
    "zoom_level": 12
  }'
```

Using the sample demo file:

```bash
curl -X POST http://localhost:5000/tile \
  -H "Content-Type: application/json" \
  -d @demos/sample_request.json
```

## API Endpoints

- `GET /` - Returns API information and available endpoints
- `GET /health` - Health check endpoint
- `POST /tile` - Process a terrain tile request

### Tile Processing Request Format

```json
{
  "name": "tile_name",
  "coordinates": {
    "lat": 40.7128,
    "lon": -74.0060
  },
  "zoom_level": 12
}
```

## Development

The `/Temp` directory is used for storing temporary processing files and will contain request dumps during tile processing.
