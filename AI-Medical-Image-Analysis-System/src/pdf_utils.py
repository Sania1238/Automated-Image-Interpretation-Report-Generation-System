from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.units import inch
import io
import streamlit as st
from datetime import datetime

def create_pdf_report(image, prediction, confidence, report_text, patient_info, analysis_time):
    """
    Create a comprehensive PDF report
    
    Args:
        image: PIL Image object
        prediction: Predicted condition
        confidence: Confidence score
        report_text: Generated report text
        patient_info: Patient information dict
        analysis_time: Time of analysis
    
    Returns:
        BytesIO: PDF buffer
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
        
        # Build story (content) for PDF
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=20,
            alignment=1  # Center alignment
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=10
        )
        
        # Title
        story.append(Paragraph("MEDICAL IMAGE ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Analysis summary table
        summary_data = [
            ['Analysis Date:', analysis_time],
            ['Predicted Condition:', prediction],
            ['AI Confidence:', f"{confidence:.1%}"],
            ['System:', 'AI Medical Imaging Analysis v1.0']
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Patient information (if provided)
        if patient_info and any(patient_info.values()):
            story.append(Paragraph("PATIENT INFORMATION", header_style))
            
            patient_data = []
            for key, value in patient_info.items():
                if value:
                    patient_data.append([f"{key}:", str(value)])
            
            if patient_data:
                patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
                patient_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                story.append(patient_table)
                story.append(Spacer(1, 20))
        
        # X-ray image
        if image:
            story.append(Paragraph("CHEST X-RAY IMAGE", header_style))
            
            # Convert PIL image to ReportLab image
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Scale image to fit page
            img_width, img_height = image.size
            max_width, max_height = 4*inch, 4*inch
            scale = min(max_width/img_width, max_height/img_height)
            scaled_width = img_width * scale
            scaled_height = img_height * scale
            
            rl_image = RLImage(img_buffer, width=scaled_width, height=scaled_height)
            story.append(rl_image)
            story.append(Spacer(1, 20))
        
        # Medical report
        story.append(Paragraph("DETAILED MEDICAL REPORT", header_style))
        
        # Process text to replace Markdown-style bold (**text**) with HTML-style bold (<b>text</b>)
        def process_text(text):
            import re
            return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

        # Update report paragraphs to use processed text
        report_paragraphs = [process_text(para) for para in report_text.split('\n\n')]

        for para in report_paragraphs:
            if para.strip():
                # Check if it's a section header (all caps)
                if para.strip().isupper() and len(para.strip()) < 50:
                    story.append(Paragraph(para.strip(), header_style))
                else:
                    story.append(Paragraph(para.strip(), styles['Normal']))
                story.append(Spacer(1, 10))
        
        # Disclaimer
        story.append(Spacer(1, 30))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.red,
            alignment=1
        )
        
        disclaimer_text = """
        MEDICAL DISCLAIMER: This report is generated by an AI system for educational and research purposes only. 
        This analysis should not be used as the sole basis for medical diagnosis or treatment decisions. 
        Always consult with qualified healthcare professionals for proper medical evaluation and care.
        """
        
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")

def create_simple_pdf_report(image, prediction, confidence, report_text, patient_info, analysis_time):
    """
    Create a simpler PDF report using canvas (fallback method)
    
    Args:
        image: PIL Image object
        prediction: Predicted condition  
        confidence: Confidence score
        report_text: Generated report text
        patient_info: Patient information dict
        analysis_time: Time of analysis
    
    Returns:
        BytesIO: PDF buffer
    """
    try:
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Header
        p.setFont("Helvetica-Bold", 18)
        p.drawCentredString(width/2, height - 50, "MEDICAL IMAGE ANALYSIS REPORT")
        
        # Analysis info
        y_position = height - 100
        p.setFont("Helvetica", 12)
        p.drawString(50, y_position, f"Analysis Date: {analysis_time}")
        y_position -= 20
        p.drawString(50, y_position, f"Predicted Condition: {prediction}")
        y_position -= 20
        p.drawString(50, y_position, f"AI Confidence: {confidence:.1%}")
        y_position -= 40
        
        # Patient info
        if patient_info and any(patient_info.values()):
            p.setFont("Helvetica-Bold", 14)
            p.drawString(50, y_position, "PATIENT INFORMATION:")
            y_position -= 20
            
            p.setFont("Helvetica", 10)
            for key, value in patient_info.items():
                if value:
                    p.drawString(70, y_position, f"{key}: {value}")
                    y_position -= 15
            y_position -= 20
        
        # Image
        if image:
            p.setFont("Helvetica-Bold", 14)
            p.drawString(50, y_position, "CHEST X-RAY IMAGE:")
            y_position -= 30
            
            # Add image
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)
            
            # Scale image
            img_width, img_height = image.size
            max_width, max_height = 200, 200
            scale = min(max_width/img_width, max_height/img_height)
            scaled_width = img_width * scale
            scaled_height = img_height * scale
            
            p.drawImage(img_reader, 50, y_position - scaled_height, 
                       width=scaled_width, height=scaled_height)
            y_position -= (scaled_height + 30)
        
        # Report text
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_position, "MEDICAL REPORT:")
        y_position -= 20
        
        # Add report text with word wrapping
        p.setFont("Helvetica", 9)
        lines = report_text.split('\n')
        for line in lines:
            if y_position < 50:  # Start new page
                p.showPage()
                y_position = height - 50
            
            # Handle long lines
            if len(line) > 80:
                words = line.split(' ')
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        if current_line:
                            p.drawString(50, y_position, current_line.strip())
                            y_position -= 12
                        current_line = word + " "
                if current_line:
                    p.drawString(50, y_position, current_line.strip())
                    y_position -= 12
            else:
                p.drawString(50, y_position, line)
                y_position -= 12
        
        # Disclaimer
        if y_position < 100:
            p.showPage()
            y_position = height - 50
        
        y_position -= 30
        p.setFont("Helvetica-Bold", 10)
        p.setFillColor(colors.red)
        p.drawString(50, y_position, "MEDICAL DISCLAIMER:")
        y_position -= 15
        
        p.setFont("Helvetica", 8)
        disclaimer_text = [
            "This report is generated by an AI system for educational purposes only.",
            "Do not use as sole basis for medical diagnosis or treatment decisions.",
            "Always consult qualified healthcare professionals for proper medical care."
        ]
        
        for line in disclaimer_text:
            p.drawString(50, y_position, line)
            y_position -= 12
        
        p.save()
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        raise Exception(f"Simple PDF generation failed: {str(e)}")

def validate_pdf_generation():
    """Test if PDF generation libraries are working"""
    try:
        # Try to create a simple test PDF
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(50, 750, "Test PDF Generation")
        p.save()
        buffer.seek(0)
        return True, "PDF generation libraries working correctly"
    except Exception as e:
        return False, f"PDF generation error: {str(e)}"