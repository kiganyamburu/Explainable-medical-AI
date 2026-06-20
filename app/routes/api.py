import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename

# Database and services imports
from app.database import db, PredictionRecord
from app.services.ml_service import MLService
from app.services.pdf_service import PDFService

api_bp = Blueprint('api', __name__)

def allowed_file(filename: str) -> bool:
    """Helper to check if uploaded file has a valid medical image extension."""
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Securely uploads a chest X-ray file and returns its filename and server path."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Sanitize name to prevent path traversal
        orig_name = secure_filename(file.filename)
        _, ext = os.path.splitext(orig_name)
        if not ext:
            ext = '.jpg'
            
        # Generate unique UUID to prevent file naming collisions
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_orig{ext}"
        
        # Save file to uploads folder
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, filename)
        
        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({'error': f"Failed to save upload image: {e}"}), 500
            
        relative_url = f"/static/uploads/{filename}"
        return jsonify({
            'success': True,
            'filename': filename,
            'url': relative_url,
            'unique_id': unique_id
        }), 201
        
    return jsonify({'error': 'File type not allowed. Please upload PNG, JPG, or JPEG.'}), 400

@api_bp.route('/predict', methods=['POST'])
def run_prediction():
    """Runs prediction pipelines, generates visual explanations, and registers a record in DB."""
    # Instantiates ml and pdf services dynamically from app config
    ml_service = MLService(current_app.config['MODEL_PATH'])
    pdf_service = PDFService(
        static_dir=os.path.join(current_app.root_path, 'static'),
        reports_dir=current_app.config['REPORT_FOLDER']
    )
    
    patient_name = request.form.get('patient_name', '').strip()
    if not patient_name:
        patient_name = "Anonymous Patient"
        
    # Check if a file was uploaded directly in this request
    if 'image' in request.files:
        # Combined Upload & Predict flow
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file upload'}), 400
            
        # Perform upload
        orig_name = secure_filename(file.filename)
        _, ext = os.path.splitext(orig_name)
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_orig{ext}"
        
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, filename)
        file.save(image_path)
    else:
        # Reference-based Prediction flow (expects pre-uploaded filename)
        filename = request.form.get('filename') or request.json.get('filename') if request.is_json else None
        if not filename:
            return jsonify({'error': 'No image provided for prediction. Upload an image first.'}), 400
            
        # Verify uploaded file exists on disk
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        image_path = os.path.join(uploads_dir, filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Uploaded image file not found on server.'}), 404
            
        # Extract unique ID from filename
        unique_id = filename.split('_orig')[0]
        
    # Trigger inference, Grad-CAM, LIME, and SHAP
    try:
        heatmaps_dir = current_app.config['HEATMAP_FOLDER']
        os.makedirs(heatmaps_dir, exist_ok=True)
        
        # ML + XAI orchestrator
        results = ml_service.predict_and_explain(
            image_path=image_path,
            unique_id=unique_id,
            uploads_dir=uploads_dir,
            heatmaps_dir=heatmaps_dir
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"AI Inference engine failure: {str(e)}"}), 500
        
    # Save record to database
    record = PredictionRecord(
        patient_name=patient_name,
        filename=filename,
        prediction=results['prediction'],
        confidence=results['confidence'],
        confidence_normal=results['confidence_normal'],
        confidence_pneumonia=results['confidence_pneumonia'],
        processing_time=results['processing_time'],
        heatmap_path=results['heatmap_path'],
        shap_path=results['shap_path'],
        lime_path=results['lime_path']
    )
    
    try:
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Database save failure: {e}"}), 500
        
    # Generate the PDF Diagnostic Report
    try:
        relative_orig_url = f"/static/uploads/{filename}"
        report_url = pdf_service.generate_report(
            record_dict=record.to_dict(),
            original_image_url=relative_orig_url,
            gradcam_image_url=results['heatmap_path']
        )
        
        # Update database with PDF report path
        record.report_path = report_url
        db.session.commit()
    except Exception as e:
        # Log error, but proceed (non-critical failure for prediction)
        print(f"Non-critical Error generating PDF report: {e}")
        
    return jsonify(record.to_dict()), 201

@api_bp.route('/history', methods=['GET'])
def get_history():
    """Queries all past prediction records from DB, sorted by descending date."""
    try:
        records = PredictionRecord.query.order_by(PredictionRecord.date.desc()).all()
        return jsonify([r.to_dict() for r in records]), 200
    except Exception as e:
        return jsonify({'error': f"Database query failure: {e}"}), 500

@api_bp.route('/report/<int:record_id>', methods=['GET'])
def serve_report(record_id: int):
    """Serves the generated clinical PDF report, recreating it on-the-fly if missing."""
    record = PredictionRecord.query.get(record_id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404
        
    pdf_filename = f"report_{record.id}.pdf"
    pdf_path = os.path.join(current_app.config['REPORT_FOLDER'], pdf_filename)
    
    # Auto-regenerate PDF report if deleted from disk but database record remains
    if not os.path.exists(pdf_path):
        try:
            pdf_service = PDFService(
                static_dir=os.path.join(current_app.root_path, 'static'),
                reports_dir=current_app.config['REPORT_FOLDER']
            )
            relative_orig_url = f"/static/uploads/{record.filename}"
            pdf_service.generate_report(
                record_dict=record.to_dict(),
                original_image_url=relative_orig_url,
                gradcam_image_url=record.heatmap_path
            )
        except Exception as e:
            return jsonify({'error': f"Failed to regenerate PDF report: {e}"}), 500
            
    return send_from_directory(current_app.config['REPORT_FOLDER'], pdf_filename, as_attachment=True)

@api_bp.route('/history/<int:record_id>', methods=['DELETE'])
def delete_record(record_id: int):
    """Deletes a prediction record and purges all of its associated image and PDF files from disk."""
    record = PredictionRecord.query.get(record_id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404
        
    # Helper to resolve database paths to physical server paths
    def resolve_physical_path(relative_url: str) -> str:
        if not relative_url:
            return ""
        clean = relative_url.lstrip('/')
        if clean.startswith('static/'):
            clean = clean.replace('static/', '', 1)
        return os.path.join(current_app.root_path, 'static', clean.replace('/', os.sep))

    # Compile files linked to this record
    file_paths = [
        resolve_physical_path(f"/static/uploads/{record.filename}"),
        resolve_physical_path(record.heatmap_path),
        resolve_physical_path(record.shap_path),
        resolve_physical_path(record.lime_path),
        resolve_physical_path(record.report_path)
    ]
    
    # Delete files from filesystem
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                # Log error and continue to avoid blocking database deletion
                print(f"Error purging file {path}: {e}")
                
    try:
        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True, 'message': f'Diagnostic record {record_id} successfully deleted.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Database deletion failure: {e}"}), 500
