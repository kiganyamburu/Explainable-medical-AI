# Explainable Medical AI for Pneumonia Detection

An advanced, production-grade clinical decision-support web application that classifies chest X-ray scans as **Normal** or **Pneumonia** and uses three Explainable AI (XAI) techniques (**Grad-CAM**, **LIME**, and **SHAP**) to explain *why* the neural network reached its diagnostic conclusion.

This project is built using a clean, service-oriented architecture separating deep learning workflows, backend REST APIs, SQL database storage, and a responsive single-page web dashboard.

---

## Key Features

- **Pneumonia Classification**: High-accuracy binary classifier trained on chest radiographs.
- **Explainable AI (XAI) Suite**:
  - **Grad-CAM**: Visualizes activation density maps in the final convolutional layers.
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Identifies contiguous superpixel regions supporting the prediction.
  - **SHAP (SHapley Additive exPlanations)**: Evaluates Shapley coalitions across superpixels to gauge pixel influence.
- **Side-by-Side Comparison Tool**: Allows clinicians to upload and review two radiographs simultaneously with visual overlays.
- **Clinical PDF Reports**: Automatically generates downloadable PDF diagnostic reports with patient metadata, original scans, Grad-CAM maps, and AI checklists.
- **Diagnostic History & Analytics**: Log database storing predictions, processing latencies, and statistics distributions.
- **Healthcare Theme & Dark Mode**: Responsive design with light and dark mode toggles.

---

## Technology Stack

- **Deep Learning**: PyTorch (`torch`, `torchvision`)
- **Backend API**: Python 3.12+, Flask, Flask Blueprints, Flask CORS
- **Database**: SQLAlchemy ORM (SQLite for development, PostgreSQL-ready)
- **Explainability Libraries**: `shap`, `lime`, `scikit-image`
- **Asset Processing**: OpenCV (`opencv-python`), Pillow, NumPy, Pandas
- **Visualization & PDF**: Matplotlib, ReportLab
- **Frontend SPA**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Production Server**: Gunicorn, Docker

---

## Directory Structure

```
Explainable-medical-AI/
├── app/
│   ├── database/
│   │   ├── __init__.py
│   │   └── models.py          # SQLAlchemy prediction logging models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py             # REST API blueprint (predict, upload, history, PDF)
│   │   └── web.py             # Router to serve the SPA
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_service.py      # Coordinates PyTorch inference and XAI overlays
│   │   └── pdf_service.py     # Generates clinical report documents via ReportLab
│   ├── templates/
│   │   └── index.html         # Responsive dashboard SPA
│   └── static/
│       ├── css/
│       │   └── styles.css     # UI colors, gauge styling, and dark mode rules
│       ├── js/
│       │   └── app.js         # REST interactions, Chart.js stats, and image toggling
│       ├── uploads/           # Original uploads folder (gitignored)
│       ├── heatmaps/          # Generated XAI overlay images
│       └── reports/           # Generated PDF report assets
├── ml/
│   ├── cnn.py                 # Custom 4-block PyTorch CNN model
│   ├── gradcam.py             # Hook-based Grad-CAM heatmap generator
│   ├── lime_explainer.py      # LIME superpixel segmenter and explainer
│   ├── shap_explainer.py      # SHAP KernelExplainer on coalitions
│   ├── train.py               # Dataset trainer, scheduler, and evaluator
│   └── predict.py             # CLI prediction helper script
├── tests/
│   └── test_app.py            # Complete Flask, DB, and XAI unit tests
├── config.py                  # Global application configurations
├── run.py                     # Entry point for development execution
├── Dockerfile                 # Docker container assembly instructions
└── requirements.txt           # Python library requirements
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+ (Python 3.14 compatible)
- Git

### 1. Clone & Initialize Environment
```bash
# Clone this repository
git clone https://github.com/kiganyamburu/Explainable-medical-AI.git
cd Explainable-medical-AI

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Dataset Setup
We utilize the **Kaggle Chest X-ray Pneumonia Dataset**. Organize the folder structure as follows:
```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Configure your dataset path in `config.py` under `DATASET_DIR`, or export it as an environment variable:
```bash
# Windows
set DATASET_DIR=C:\path\to\your\chest_xray
# Linux/macOS
export DATASET_DIR=/path/to/your/chest_xray
```

### 3. Model Training
To train the CNN model from scratch and evaluate test statistics, run the training pipeline:
```bash
python ml/train.py --train-dir C:\path\to\chest_xray\train --val-dir C:\path\to\chest_xray\val --test-dir C:\path\to\chest_xray\test --epochs 10 --batch-size 32
```
This script runs training, performs validation checkpoints (saving the best weights to `ml/model.pth`), outputs classification scores (F1, Precision, ROC AUC), and plots performance metrics in `app/static/images/`.

---

## Running the Web Application

### 1. Start the Flask Server
```bash
python run.py
```
This initializes the database (`app/database/pneumonia.db`), registers API blueprints, creates target directories, and starts the server.

Open your browser and navigate to:
**`http://localhost:5000`**

### 2. Run CLI Predictions
You can perform quick classifications from your terminal:
```bash
python ml/predict.py --image path/to/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg --model ml/model.pth
```

---

## API Documentation

### 1. Secure Image Upload
- **Endpoint**: `POST /upload`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `image`: File (PNG, JPEG, JPG)
- **Response (201 Created)**:
```json
{
  "success": true,
  "filename": "79b38965-f481-42db-a7b3-8b776fd845bc_orig.jpg",
  "url": "/static/uploads/79b38965-f481-42db-a7b3-8b776fd845bc_orig.jpg",
  "unique_id": "79b38965-f481-42db-a7b3-8b776fd845bc"
}
```

### 2. Diagnostic Prediction & Explainability (XAI)
- **Endpoint**: `POST /predict`
- **Content-Type**: `application/json` or `multipart/form-data`
- **Request Parameters (Form/JSON)**:
  - `filename`: String (returned by `/upload`) OR direct file upload as `image`
  - `patient_name`: String (Optional)
- **Response (201 Created)**:
```json
{
  "id": 1,
  "patient_name": "Jane Doe",
  "filename": "79b38965-f481-42db-a7b3-8b776fd845bc_orig.jpg",
  "prediction": "Pneumonia",
  "confidence": 87.42,
  "confidence_normal": 12.58,
  "confidence_pneumonia": 87.42,
  "processing_time": 1.42,
  "heatmap_path": "/static/heatmaps/79b38965-f481-42db-a7b3-8b776fd845bc_gradcam.jpg",
  "shap_path": "/static/heatmaps/79b38965-f481-42db-a7b3-8b776fd845bc_shap.jpg",
  "lime_path": "/static/heatmaps/79b38965-f481-42db-a7b3-8b776fd845bc_lime.jpg",
  "report_path": "/static/reports/report_1.pdf",
  "date": "2026-07-01 03:40:00"
}
```

### 3. Retrieve Historical Logs
- **Endpoint**: `GET /history`
- **Response (200 OK)**:
Returns a JSON array of all past diagnostic entries stored in the database.

### 4. Fetch/Download PDF Diagnostic Report
- **Endpoint**: `GET /report/<id>`
- **Response (200 OK)**:
Serves the generated PDF report as a direct download. Automatically regenerates the document if deleted from disk.

### 5. Delete Diagnostic Entry
- **Endpoint**: `DELETE /history/<id>`
- **Response (200 OK)**:
```json
{
  "success": true,
  "message": "Diagnostic record 1 successfully deleted."
}
```
Deletes database records and purges all original files, overlays, and PDF reports from storage.

---

## Production Deployment

### 1. Docker Deployment
We use a multi-stage Docker build optimized for minimal footprint:
```bash
# Build the Docker image
docker build -t pneumonia-xai .

# Run the container
docker run -p 5000:5000 -e SECRET_KEY="my-key" pneumonia-xai
```

### 2. Render / Railway Deployment
Deploying to cloud platforms:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --workers=1 --threads=2 --bind=0.0.0.0:$PORT run:app`
- Set `FLASK_ENV` as `production`. Set `DATABASE_URL` to connect to PostgreSQL (the app handles connection strings automatically).

---

## Verification & Testing
Run python unit tests to verify API endpoints, database operations, and explainability overlays:
```bash
python -m unittest discover tests
```

---

## Future Scope
- **Multi-Class Extensions**: Diagnose COVID-19, pleural effusion, and tuberculosis.
- **Transformer Architectures**: Introduce Vision Transformers (ViT) and compute Attention Rollout visualizations.
- **DICOM Integration**: Directly support parsing PACS system `.dcm` files.

---

## License
Distributed under the MIT License. See `LICENSE` for more details.
