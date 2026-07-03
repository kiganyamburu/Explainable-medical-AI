import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base Configuration class."""
    SECRET_KEY = os.getenv("SECRET_KEY", "prod-medical-ai-secret-key-12345")
    
    # Database settings - Fallback to local sqlite
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{os.path.join(BASE_DIR, 'app', 'database', 'pneumonia.db')}"
    )
    # Fix for PostgreSQL DATABASE_URL format (Render/Railway use postgres:// instead of postgresql://)
    if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
        
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload and static assets settings
    # We place files under app/static so Flask can easily serve them via HTTP
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    HEATMAP_FOLDER = os.path.join(BASE_DIR, "app", "static", "heatmaps")
    REPORT_FOLDER = os.path.join(BASE_DIR, "app", "static", "reports")
    
    # Security limits
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    
    # Machine Learning Settings
    MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pth")
    DATASET_DIR = os.getenv(
        "DATASET_DIR", 
        r"c:\Users\User\Desktop\Projects\AI-based-pneumonia-detection-from-chest-X-ray-images\chest_xray"
    )

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
    # Connection pooling configuration for production PostgreSQL databases
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 1800,
        "pool_pre_ping": True
    }

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'app', 'database', 'test_pneumonia.db')}"

# Export active configuration based on environment variable
ENV = os.getenv("FLASK_ENV", "development").lower()
if ENV == "production":
    active_config = ProductionConfig
elif ENV == "testing":
    active_config = TestingConfig
else:
    active_config = DevelopmentConfig
