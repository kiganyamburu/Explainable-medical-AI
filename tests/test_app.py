import os
import sys
import unittest
import uuid
import torch
import numpy as np
from io import BytesIO
from PIL import Image

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database.models import db, PredictionRecord
from config import TestingConfig
from ml.cnn import PneumoniaCNN
from ml.gradcam import generate_gradcam_overlay

class ExplainableMedicalAITestCase(unittest.TestCase):
    """Unit and integration test cases for the Explainable Medical AI web application."""
    
    def setUp(self) -> None:
        """Sets up the Flask test client and recreates the test database tables."""
        # Use TestingConfig
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Build tables
        db.create_all()
        
        # Path references
        self.uploads_dir = self.app.config['UPLOAD_FOLDER']
        self.heatmaps_dir = self.app.config['HEATMAP_FOLDER']
        self.reports_dir = self.app.config['REPORT_FOLDER']

    def tearDown(self) -> None:
        """Purges testing database tables and removes test contexts."""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        
        # Clean up any leftover test files in test uploads
        test_db_path = self.app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except OSError:
                pass

    def _create_mock_image(self) -> BytesIO:
        """Helper to generate a mock 224x224 RGB image in memory."""
        img_bytes = BytesIO()
        image = Image.new('RGB', (224, 224), color='white')
        image.save(img_bytes, 'JPEG')
        img_bytes.seek(0)
        return img_bytes

    def test_database_model_serialization(self):
        """Tests that the PredictionRecord SQL model serializes correctly into dictionaries."""
        record = PredictionRecord(
            patient_name="Test Patient J",
            filename="test_image.jpg",
            prediction="Normal",
            confidence=0.9523,
            confidence_normal=0.9523,
            confidence_pneumonia=0.0477,
            processing_time=1.23,
            heatmap_path="/static/heatmaps/test_heatmap.jpg",
            shap_path="/static/heatmaps/test_shap.jpg",
            lime_path="/static/heatmaps/test_lime.jpg",
            report_path="/static/reports/test_report.pdf"
        )
        db.session.add(record)
        db.session.commit()
        
        serialized = record.to_dict()
        self.assertEqual(serialized['patient_name'], "Test Patient J")
        self.assertEqual(serialized['prediction'], "Normal")
        self.assertEqual(serialized['confidence'], 95.23)  # Expressed as percentage
        self.assertEqual(serialized['processing_time'], 1.23)

    def test_upload_endpoint_validation(self):
        """Tests file upload constraints including missing data and unsupported formats."""
        # 1. Test upload with no files
        response = self.client.post('/upload', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No image file uploaded', response.data)
        
        # 2. Test upload with invalid file extension
        invalid_file = (BytesIO(b"dummy text data"), 'test.txt')
        response = self.client.post('/upload', data={'image': invalid_file})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'File type not allowed', response.data)

        # 3. Test successful upload of valid image
        mock_img = self._create_mock_image()
        response = self.client.post(
            '/upload', 
            data={'image': (mock_img, 'test_xray.jpg')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 201)
        res_data = response.get_json()
        self.assertTrue(res_data['success'])
        self.assertTrue(res_data['filename'].endswith('.jpg'))
        self.assertIn('/static/uploads/', res_data['url'])
        
        # Verify physical file was written
        uploaded_file = os.path.join(self.uploads_dir, res_data['filename'])
        self.assertTrue(os.path.exists(uploaded_file))
        
        # Cleanup
        if os.path.exists(uploaded_file):
            os.remove(uploaded_file)

    def test_predict_endpoint_direct_upload(self):
        """Tests running /predict via direct file multipart upload."""
        mock_img = self._create_mock_image()
        response = self.client.post(
            '/predict',
            data={
                'image': (mock_img, 'patient_scan.png'),
                'patient_name': 'Test Jane Doe'
            },
            content_type='multipart/form-data'
        )
        # Should complete and return prediction details
        self.assertEqual(response.status_code, 201)
        res_data = response.get_json()
        self.assertEqual(res_data['patient_name'], 'Test Jane Doe')
        self.assertIn(res_data['prediction'], ['Normal', 'Pneumonia'])
        self.assertGreater(res_data['confidence'], 0.0)
        
        # Cleanup physical file dependencies generated
        orig_path = os.path.join(self.uploads_dir, res_data['filename'])
        hm_path = os.path.join(self.app.root_path, 'static', res_data['heatmap_path'].lstrip('/static/').replace('/', os.sep))
        sp_path = os.path.join(self.app.root_path, 'static', res_data['shap_path'].lstrip('/static/').replace('/', os.sep))
        lm_path = os.path.join(self.app.root_path, 'static', res_data['lime_path'].lstrip('/static/').replace('/', os.sep))
        rp_path = os.path.join(self.reports_dir, f"report_{res_data['id']}.pdf")
        
        for path in (orig_path, hm_path, sp_path, lm_path, rp_path):
            if os.path.exists(path):
                os.remove(path)

    def test_history_and_delete_endpoints(self):
        """Tests retrieving history logs and deleting records with files purge."""
        # Pre-seed record in DB
        record = PredictionRecord(
            patient_name="Delete Target Patient",
            filename="del_test.jpg",
            prediction="Pneumonia",
            confidence=0.88,
            confidence_normal=0.12,
            confidence_pneumonia=0.88,
            processing_time=1.5
        )
        db.session.add(record)
        db.session.commit()
        record_id = record.id
        
        # Create dummy physical files to test file system cleanup
        dummy_files = [
            os.path.join(self.uploads_dir, "del_test.jpg"),
            os.path.join(self.heatmaps_dir, "del_test_gradcam.jpg"),
            os.path.join(self.reports_dir, f"report_{record_id}.pdf")
        ]
        for path in dummy_files:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write("dummy data")
                
        # Link mock paths to record
        record.heatmap_path = f"/static/heatmaps/del_test_gradcam.jpg"
        record.report_path = f"/static/reports/report_{record_id}.pdf"
        db.session.commit()
        
        # 1. Test GET /history
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        history_list = response.get_json()
        self.assertEqual(len(history_list), 1)
        self.assertEqual(history_list[0]['patient_name'], "Delete Target Patient")
        
        # 2. Test DELETE /history/<id>
        del_response = self.client.delete(f'/history/{record_id}')
        self.assertEqual(del_response.status_code, 200)
        
        # Assert database entry was removed
        self.assertIsNone(PredictionRecord.query.get(record_id))
        
        # Assert linked files were deleted from disk
        for path in dummy_files:
            self.assertFalse(os.path.exists(path))

    def test_gradcam_overlay_execution(self):
        """Tests that Grad-CAM triggers and calculates heatmaps on model representations successfully."""
        model = PneumoniaCNN()
        img_tensor = torch.randn(1, 3, 224, 224)
        mock_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        out_path = os.path.join(self.heatmaps_dir, "test_gradcam_exec.jpg")
        
        # Grad-CAM function call assertion
        path, val = generate_gradcam_overlay(model, img_tensor, mock_img, out_path, target_class_idx=1)
        
        self.assertEqual(path, out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreaterEqual(val, 0.0)
        
        # Cleanup
        if os.path.exists(out_path):
            os.remove(out_path)

if __name__ == '__main__':
    unittest.main()
