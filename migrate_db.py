import os
import sys

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from app.database.models import db, PredictionRecord

def migrate():
    # 1. Open SQLite database connection to fetch old records
    sqlite_db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app', 'database', 'pneumonia.db')
    
    if not os.path.exists(sqlite_db_path):
        print(f"SQLite database file not found at: {sqlite_db_path}. Nothing to migrate.")
        return
        
    sqlite_uri = f"sqlite:///{sqlite_db_path}"
    
    from config import active_config
    class SQLiteMigrationConfig(active_config):
        SQLALCHEMY_DATABASE_URI = sqlite_uri
        
    app_sqlite = create_app(SQLiteMigrationConfig)
    
    records_to_migrate = []
    with app_sqlite.app_context():
        print(f"Reading historical records from SQLite database...")
        sqlite_records = db.session.scalars(db.select(PredictionRecord)).all()
        for r in sqlite_records:
            records_to_migrate.append({
                'patient_name': r.patient_name,
                'filename': r.filename,
                'prediction': r.prediction,
                'confidence': r.confidence / 100.0 if r.confidence > 1.0 else r.confidence, # Normalize percentage representation
                'confidence_normal': r.confidence_normal / 100.0 if r.confidence_normal > 1.0 else r.confidence_normal,
                'confidence_pneumonia': r.confidence_pneumonia / 100.0 if r.confidence_pneumonia > 1.0 else r.confidence_pneumonia,
                'processing_time': r.processing_time,
                'heatmap_path': r.heatmap_path,
                'shap_path': r.shap_path,
                'lime_path': r.lime_path,
                'report_path': r.report_path,
                'date': r.date
            })
            
    if not records_to_migrate:
        print("No diagnostic records found in SQLite database to migrate.")
        return
        
    print(f"Found {len(records_to_migrate)} records in SQLite.")
    
    # 2. Open active PostgreSQL database connection (loaded via .env)
    app_pg = create_app()
    with app_pg.app_context():
        db.create_all() # Ensure PostgreSQL tables exist
        print(f"Connecting to target database: {app_pg.config['SQLALCHEMY_DATABASE_URI']}...")
        
        migrated_count = 0
        for data in records_to_migrate:
            # Prevent duplicate migrations
            exists = db.session.scalar(
                db.select(PredictionRecord).where(
                    PredictionRecord.filename == data['filename'],
                    PredictionRecord.date == data['date']
                )
            )
            if not exists:
                record = PredictionRecord(**data)
                db.session.add(record)
                migrated_count += 1
                
        db.session.commit()
        print(f"Migration successful! Copied {migrated_count} diagnostic records to PostgreSQL.")

if __name__ == '__main__':
    migrate()
