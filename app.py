#!/usr/bin/env python3
"""
Virtual Try-On Backend API
Flask application for CP-VTON model integration
"""

import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import config
from pathlib import Path

from src.utils.logger import Logger
from src.controllers.tryon_controller import tryon_bp

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configure app
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = str(config.path_config.UPLOADS)

# Enable CORS
CORS(app)

# Initialize logger
logger = Logger(__name__)

# Register blueprints
app.register_blueprint(tryon_bp)

@app.route('/')
def index():
    """Serve the main virtual try-on UI."""
    return render_template('index.html')

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve generated try-on results."""
    try:
        return send_from_directory(str(config.path_config.RESULTS), filename)
    except FileNotFoundError:
        return jsonify({'error': 'Result file not found'}), 404

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 400

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large', 'message': 'File size exceeds 10MB limit'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

if __name__ == '__main__':
    logger.info("Starting Virtual Try-On Flask Application...")
    logger.info(f"Upload folder: {config.path_config.UPLOADS}")
    logger.info(f"Results folder: {config.path_config.RESULTS}")
    
    # Ensure directories exist
    config.path_config.UPLOADS.mkdir(parents=True, exist_ok=True)
    config.path_config.RESULTS.mkdir(parents=True, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
