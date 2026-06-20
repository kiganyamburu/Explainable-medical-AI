from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Initialize db instance (bound to app in factory)
db = SQLAlchemy()

class PredictionRecord(db.Model):
    """Database model to store chest X-ray predictions, confidence scores, processing times, and explainability assets."""
    __tablename__ = 'prediction_records'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_name = db.Column(db.String(100), nullable=False, default="Anonymous Patient")
    filename = db.Column(db.String(255), nullable=False)  # Original file name uploaded
    prediction = db.Column(db.String(20), nullable=False)  # Normal vs Pneumonia
    confidence = db.Column(db.Float, nullable=False)       # Winning class confidence (e.g. 0.874)
    confidence_normal = db.Column(db.Float, nullable=False)
    confidence_pneumonia = db.Column(db.Float, nullable=False)
    processing_time = db.Column(db.Float, nullable=False)  # Prediction runtime in seconds
    
    # Paths to generated visual explanation files
    heatmap_path = db.Column(db.String(255), nullable=True)  # Grad-CAM overlay path
    shap_path = db.Column(db.String(255), nullable=True)     # SHAP overlay path
    lime_path = db.Column(db.String(255), nullable=True)     # LIME overlay path
    report_path = db.Column(db.String(255), nullable=True)   # PDF diagnostic report path
    
    date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> dict:
        """Serializes the SQLAlchemy record object to a dictionary for API delivery."""
        return {
            'id': self.id,
            'patient_name': self.patient_name,
            'filename': self.filename,
            'prediction': self.prediction,
            'confidence': round(self.confidence * 100, 2),  # Represent as percentage: 87.40
            'confidence_normal': round(self.confidence_normal * 100, 2),
            'confidence_pneumonia': round(self.confidence_pneumonia * 100, 2),
            'processing_time': round(self.processing_time, 2),
            'heatmap_path': self.heatmap_path,
            'shap_path': self.shap_path,
            'lime_path': self.lime_path,
            'report_path': self.report_path,
            'date': self.date.strftime('%Y-%m-%d %H:%M:%S')
        }
