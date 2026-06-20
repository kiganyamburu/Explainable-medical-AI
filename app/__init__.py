import os
from flask import Flask, jsonify
from flask_cors import CORS

# Imports config and SQLAlchemy instance
from config import active_config
from app.database.models import db
from app.routes.api import api_bp
from app.routes.web import web_bp

def create_app(config_class=active_config) -> Flask:
    """Flask App Factory to configure database, register blueprints, and build required directories."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable Cross-Origin Resource Sharing (CORS)
    CORS(app)
    
    # Initialize SQLAlchemy database mapping
    db.init_app(app)
    
    # Create necessary asset upload and output directories on startup
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
        os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
        
        # Verify and build SQLite tables if they do not exist
        try:
            db.create_all()
            print("SQL Database tables verified and created successfully.")
        except Exception as e:
            print(f"Error creating database tables on startup: {e}")
            
    # Register blueprints
    # Serving REST endpoints and web SPA directly at root level
    app.register_blueprint(api_bp, url_prefix='/')
    app.register_blueprint(web_bp, url_prefix='/')
    
    # Global error handlers
    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({
            'error': 'File is too large. Maximum allowed size is 16 Megabytes.'
        }), 413
        
    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify({
            'error': 'The requested URL or resource was not found on this server.'
        }), 404
        
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({
            'error': 'An internal server error occurred. Please contact the administrator.'
        }), 500
        
    return app
