"""
TerrainTiles Flask Application

A simple Flask API for terrain tiles manipulation.
"""

from flask import Flask, request, jsonify
import os
import json
import re

app = Flask(__name__)

# Configuration
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'Temp')
DEMOS_DIR = os.path.join(os.path.dirname(__file__), 'demos')

# Ensure Temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


def sanitize_filename(filename):
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A safe filename with only alphanumeric characters, underscores, and hyphens
    """
    # Remove any path separators and keep only safe characters
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    # Ensure the filename is not empty after sanitization
    if not safe_name:
        safe_name = 'unnamed'
    return safe_name


@app.route('/')
def index():
    """Root endpoint."""
    return jsonify({
        'message': 'Welcome to TerrainTiles API',
        'endpoints': {
            '/': 'This help message',
            '/health': 'Health check endpoint',
            '/tile': 'Process a terrain tile (POST)'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


@app.route('/tile', methods=['POST'])
def process_tile():
    """
    Process a terrain tile request.
    
    Expects JSON with:
    - name: tile name
    - coordinates: {lat, lon}
    - zoom_level: integer
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['name', 'coordinates', 'zoom_level']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Sanitize the filename to prevent path traversal
        safe_name = sanitize_filename(data['name'])
        
        # Save request to Temp directory for processing
        temp_file = os.path.join(TEMP_DIR, f"{safe_name}_request.json")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Process the tile (placeholder logic)
        result = {
            'status': 'processed',
            'tile': data['name'],
            'coordinates': data['coordinates'],
            'zoom_level': data['zoom_level'],
            'temp_file': temp_file
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
