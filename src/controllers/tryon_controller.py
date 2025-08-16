import os
import uuid
from pathlib import Path
from typing import Dict, Any
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_cors import CORS

from ..services.tryon_service import TryOnService
from ..utils.logger import Logger
from ..utils.validators import validate_image_file
import config

# Create Blueprint for try-on routes
tryon_bp = Blueprint('tryon', __name__, url_prefix='/api/v1')
CORS(tryon_bp)

class TryOnController:
    """
    Controller for handling virtual try-on API endpoints.
    Manages file uploads, validation, and response formatting.
    """
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.service = TryOnService()
        self.logger.info("TryOnController initialized successfully")
    
    def _handle_tryon_request(self) -> Dict[str, Any]:
        """Handle the main try-on POST request."""
        try:
            self.logger.info("Received try-on request")
            
            # Check if files are present
            if 'person_image' not in request.files:
                return self._error_response("Missing person_image file", 400)
            
            if 'clothing_image' not in request.files:
                return self._error_response("Missing clothing_image file", 400)
            
            person_file = request.files['person_image']
            clothing_file = request.files['clothing_image']
            
            # Check if files were selected
            if person_file.filename == '':
                return self._error_response("No person image selected", 400)
            
            if clothing_file.filename == '':
                return self._error_response("No clothing image selected", 400)
            
            # Validate file types
            if not validate_image_file(person_file):
                return self._error_response("Invalid person image format. Supported: JPG, PNG, BMP", 400)
            
            if not validate_image_file(clothing_file):
                return self._error_response("Invalid clothing image format. Supported: JPG, PNG, BMP", 400)
            
            # Generate unique filenames
            person_filename = self._generate_unique_filename(person_file.filename)
            clothing_filename = self._generate_unique_filename(clothing_file.filename)
            
            # Save uploaded files
            person_path = self._save_uploaded_file(person_file, person_filename)
            clothing_path = self._save_uploaded_file(clothing_file, clothing_filename)
            
            try:
                # Process try-on
                output_path, output_url = self.service.process_tryon(
                    person_path, 
                    clothing_path,
                    request.form.get('output_filename')
                )
                
                # Clean up uploaded files
                self.service.cleanup_temp_files(person_path, clothing_path)
                
                # Return success response
                return self._success_response({
                    "message": "Try-on completed successfully",
                    "result_url": output_url,
                    "result_path": output_path,
                    "request_id": str(uuid.uuid4())
                })
                
            except Exception as e:
                # Clean up uploaded files on error
                self.service.cleanup_temp_files(person_path, clothing_path)
                raise e
            
        except Exception as e:
            self.logger.error(f"Error processing try-on request: {str(e)}")
            return self._error_response(f"Try-on processing failed: {str(e)}", 500)
    
    def _handle_status_request(self) -> Dict[str, Any]:
        """Handle status request."""
        try:
            status = self.service.get_processing_status()
            return self._success_response(status)
        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            return self._error_response("Failed to get status", 500)
    
    def _serve_result_file(self, filename: str):
        """Serve result image files."""
        try:
            # Validate filename for security
            if not filename or '..' in filename or '/' in filename:
                return self._error_response("Invalid filename", 400)
            
            result_dir = config.path_config.RESULTS
            file_path = result_dir / filename
            
            if not file_path.exists():
                return self._error_response("File not found", 404)
            
            return send_from_directory(str(result_dir), filename)
            
        except Exception as e:
            self.logger.error(f"Error serving result file: {str(e)}")
            return self._error_response("Error serving file", 500)
    
    def _save_uploaded_file(self, file, filename: str) -> str:
        """Save uploaded file to uploads directory."""
        try:
            # Ensure uploads directory exists
            config.path_config.UPLOADS.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Uploads directory: {config.path_config.UPLOADS} (exists: {config.path_config.UPLOADS.exists()})")
            
            # Save file
            file_path = config.path_config.UPLOADS / filename
            self.logger.info(f"Attempting to save uploaded file to: {file_path}")
            
            file.save(str(file_path))
            
            # Verify file was created
            if file_path.exists():
                self.logger.info(f"Uploaded file successfully saved: {file_path} (size: {file_path.stat().st_size} bytes)")
            else:
                raise RuntimeError(f"Uploaded file was not created at {file_path}")
            
            self.logger.info(f"Saved uploaded file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving uploaded file: {str(e)}")
            raise
    
    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename for uploaded file."""
        # Get file extension
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = '.jpg'  # Default extension
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        return f"upload_{unique_id}{ext}"
    
    def _success_response(self, data: Dict[str, Any], status_code: int = 200) -> tuple:
        """Format successful response."""
        response = {
            "success": True,
            "data": data,
            "timestamp": str(uuid.uuid4())  # Simple timestamp placeholder
        }
        return jsonify(response), status_code
    
    def _error_response(self, message: str, status_code: int = 400) -> tuple:
        """Format error response."""
        response = {
            "success": False,
            "error": {
                "message": message,
                "code": status_code
            },
            "timestamp": str(uuid.uuid4())  # Simple timestamp placeholder
        }
        return jsonify(response), status_code

# Create controller instance for route handlers
tryon_controller = TryOnController()

@tryon_bp.route('/tryon', methods=['POST'])
def tryon_endpoint():
    """Main try-on endpoint for processing person and clothing images."""
    return tryon_controller._handle_tryon_request()

@tryon_bp.route('/status', methods=['GET'])
def status_endpoint():
    """Get service status and model information."""
    return tryon_controller._handle_status_request()

@tryon_bp.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    """Serve result images."""
    return tryon_controller._serve_result_file(filename)
