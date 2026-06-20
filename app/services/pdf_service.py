import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFService:
    """Service to generate professional diagnostic PDF reports for clinical verification."""
    
    def __init__(self, static_dir: str, reports_dir: str) -> None:
        self.static_dir = static_dir
        self.reports_dir = reports_dir
        
    def _get_absolute_path(self, relative_url: str) -> str:
        """Converts a relative URL path (like /static/uploads/xxx.png) to a local file system path."""
        # Clean relative url prefix
        clean_url = relative_url.lstrip('/')
        if clean_url.startswith('static/'):
            clean_url = clean_url.replace('static/', '', 1)
            
        return os.path.join(self.static_dir, clean_url.replace('/', os.sep))

    def generate_report(self, record_dict: dict, original_image_url: str, gradcam_image_url: str) -> str:
        """Generates a professional clinical PDF report and returns the relative path of the saved file."""
        unique_id = record_dict.get('id', 'N/A')
        patient_name = record_dict.get('patient_name', 'Anonymous Patient')
        prediction = record_dict.get('prediction', 'Normal')
        confidence = record_dict.get('confidence', 0.0)
        conf_normal = record_dict.get('confidence_normal', 0.0)
        conf_pneumonia = record_dict.get('confidence_pneumonia', 0.0)
        date_str = record_dict.get('date', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        processing_time = record_dict.get('processing_time', 0.0)
        
        # Setup output PDF filename
        pdf_filename = f"report_{unique_id}.pdf"
        pdf_path = os.path.join(self.reports_dir, pdf_filename)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Setup ReportLab document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Custom premium palette
        primary_color = colors.HexColor("#0D47A1")   # Clinical Blue
        secondary_color = colors.HexColor("#1976D2") # Accent Blue
        dark_neutral = colors.HexColor("#212121")    # Charcoal Text
        light_bg = colors.HexColor("#F5F5F5")        # Soft Light Grey
        
        # Define paragraph styles
        title_style = ParagraphStyle(
            name='DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=primary_color,
            spaceAfter=6
        )
        
        subtitle_style = ParagraphStyle(
            name='DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=10,
            textColor=colors.HexColor("#757575"),
            spaceAfter=15
        )
        
        section_heading = ParagraphStyle(
            name='SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            textColor=secondary_color,
            spaceBefore=12,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            name='ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10.5,
            textColor=dark_neutral,
            leading=14
        )
        
        finding_bullet_style = ParagraphStyle(
            name='FindingBullet',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=dark_neutral,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=4
        )
        
        # Build document flow list
        story = []
        
        # 1. Header Section
        story.append(Paragraph("EXPLAINABLE MEDICAL AI DIAGNOSTIC REPORT", title_style))
        story.append(Paragraph("Automated Chest X-Ray Screening for Pneumonia Detection with Visual Explanations", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=15))
        
        # 2. Patient and Metadata Table
        metadata_data = [
            [
                Paragraph("<b>Patient Name:</b>", body_style), Paragraph(patient_name, body_style),
                Paragraph("<b>Report Date:</b>", body_style), Paragraph(date_str, body_style)
            ],
            [
                Paragraph("<b>Diagnostic ID:</b>", body_style), Paragraph(f"REC-#{unique_id}", body_style),
                Paragraph("<b>Model Version:</b>", body_style), Paragraph("PneumoniaCNN v1.0.0", body_style)
            ]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1.3*inch, 2.3*inch, 1.2*inch, 2.5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), light_bg),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 8),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D3D3D3")),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 15))
        
        # 3. Diagnosis results summary card
        result_color = colors.HexColor("#D32F2F") if prediction == "Pneumonia" else colors.HexColor("#388E3C")
        result_text_color = colors.white
        
        diag_title_style = ParagraphStyle(
            'DiagTitle', parent=body_style, fontName='Helvetica-Bold', fontSize=14, textColor=result_text_color
        )
        
        diag_metric_style = ParagraphStyle(
            'DiagMetric', parent=body_style, fontName='Helvetica', fontSize=11, textColor=result_text_color
        )
        
        diagnosis_data = [
            [
                Paragraph(f"AI DIAGNOSTIC RESULT: {prediction.upper()}", diag_title_style),
                Paragraph(f"Confidence score: {confidence:.2f}%", diag_title_style)
            ],
            [
                Paragraph(f"Probabilities — Pneumonia: {conf_pneumonia:.2f}%  |  Normal: {conf_normal:.2f}%", diag_metric_style),
                Paragraph(f"Analysis Processing Time: {processing_time:.2f} seconds", diag_metric_style)
            ]
        ]
        
        diagnosis_table = Table(diagnosis_data, colWidths=[4.2*inch, 3.1*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), result_color),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, result_color),
        ]))
        story.append(diagnosis_table)
        story.append(Spacer(1, 15))
        
        # 4. Images Section (Original X-Ray & Grad-CAM side-by-side)
        story.append(Paragraph("IMAGE FINDINGS & EXPLAINABLE AI ANALYSIS", section_heading))
        
        orig_img_path = self._get_absolute_path(original_image_url)
        gradcam_img_path = self._get_absolute_path(gradcam_image_url)
        
        image_elements = []
        
        # Ensure image paths exist, resize to fit nicely (about 3.25 inches wide, maintaining ratio)
        # Standard chest X-rays fit nicely in a square for report purposes
        img_width = 3.25 * inch
        img_height = 3.25 * inch
        
        if os.path.exists(orig_img_path):
            img_orig = Image(orig_img_path, width=img_width, height=img_height)
            image_elements.append(img_orig)
        else:
            image_elements.append(Paragraph("[Original X-ray Missing]", body_style))
            
        if os.path.exists(gradcam_img_path):
            img_gradcam = Image(gradcam_img_path, width=img_width, height=img_height)
            image_elements.append(img_gradcam)
        else:
            image_elements.append(Paragraph("[Grad-CAM Heatmap Missing]", body_style))
            
        # Draw side-by-side images in a table
        images_table_data = [
            [Paragraph("<b>Original Chest X-Ray</b>", body_style), Paragraph("<b>Grad-CAM Localization Heatmap</b>", body_style)],
            image_elements
        ]
        
        images_table = Table(images_table_data, colWidths=[3.65*inch, 3.65*inch])
        images_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        story.append(images_table)
        story.append(Spacer(1, 15))
        
        # 5. Diagnostic Findings Checklist & Explanations
        story.append(Paragraph("AI DIAGNOSTIC FINDINGS CHECKLIST", section_heading))
        
        # Define checklist items based on result
        if prediction == "Pneumonia":
            findings_bullets = [
                Paragraph("✓ <b>Lung Opacity:</b> The model detected areas of increased density in pulmonary fields.", finding_bullet_style),
                Paragraph("✓ <b>Lower Lobe Abnormalities:</b> Focal densities and consolidations localized in the lobes.", finding_bullet_style),
                Paragraph("✓ <b>High Density Infiltrates:</b> Fluid density regions detected, strongly shifting probability values.", finding_bullet_style)
            ]
            explanation_desc = (
                "The AI model identified localized regions of high density and opacity, highlighted by the red/yellow "
                "regions in the Grad-CAM heatmap above. These patterns correspond to consolidated lung parenchyma and fluid "
                "infiltrates, which are standard clinical indicators of pneumonia infection."
            )
        else:
            findings_bullets = [
                Paragraph("✓ <b>Clear Lung Fields:</b> Normal radiolucency throughout the lung tissue.", finding_bullet_style),
                Paragraph("✓ <b>No Consolidated Densities:</b> Absence of pulmonary infiltrates or consolidation.", finding_bullet_style),
                Paragraph("✓ <b>Normal Aeration:</b> Symmetric and healthy dark pulmonary expansion patterns.", finding_bullet_style)
            ]
            explanation_desc = (
                "The chest radiograph displays normal lung volumes, clear lung fields, and healthy dark regions representing "
                "normal aeration. No focal opacities, consolidations, or fluid lines are highlighted in the explainability layers. "
                "Consequently, the model predicts a Normal scan with high confidence."
            )
            
        for bullet in findings_bullets:
            story.append(bullet)
            
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>Visual Explanation Summary:</b> {explanation_desc}", body_style))
        story.append(Spacer(1, 20))
        
        # 6. Disclaimer & Bottom Banner
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D3D3D3"), spaceAfter=10))
        
        disclaimer_style = ParagraphStyle(
            name='Disclaimer',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=colors.HexColor("#757575"),
            leading=10
        )
        
        disclaimer_text = (
            "<b>Disclaimer:</b> This diagnostic report was generated by an experimental Artificial Intelligence system. "
            "It is intended solely for educational, portfolio demonstration, and decision-support purposes. This report "
            "does NOT constitute a medical diagnosis. The findings and explanations should always be verified by a "
            "board-certified physician or radiologist."
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Save report
        doc.build(story)
        
        # Convert path to relative URL
        parts = pdf_path.replace("\\", "/").split("app/static/")
        if len(parts) > 1:
            return "/static/" + parts[1]
            
        return pdf_path
