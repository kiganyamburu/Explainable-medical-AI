import sys
import os

# Add current directory to path to ensure app imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from app.database.models import db, PredictionRecord

def check_database():
    app = create_app()
    with app.app_context():
        # Query all prediction records sorted by date descending
        records = db.session.scalars(db.select(PredictionRecord).order_by(PredictionRecord.date.desc())).all()
        
        if not records:
            print("\n================================================================================")
            print("                 EXPLAINABLE MEDICAL AI - DATABASE STATUS                       ")
            print("================================================================================")
            print(" No diagnostic records found in the database.")
            print(" Go to http://localhost:5000 to upload and run predictions first.")
            print("================================================================================\n")
            return
            
        print("\n================================================================================")
        print(f"                 EXPLAINABLE MEDICAL AI - DATABASE STATUS ({len(records)} records)")
        print("================================================================================")
        print(f"{'ID':<5} | {'Patient Name':<28} | {'Prediction':<12} | {'Confidence':<10} | {'Date (UTC)':<19}")
        print("-" * 80)
        
        for r in records:
            # Format and truncate patient name if too long for tabular view
            name = r.patient_name
            if len(name) > 28:
                name = name[:25] + "..."
                
            conf = f"{r.confidence * 100:.1f}%" if r.confidence <= 1.0 else f"{r.confidence:.1f}%"
            date_str = r.date.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{r.id:<5} | {name:<28} | {r.prediction:<12} | {conf:<10} | {date_str:<19}")
            
        print("================================================================================\n")

if __name__ == '__main__':
    check_database()
