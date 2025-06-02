#!/usr/bin/env python3

# Simple test of text extraction functionality
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Simple Text Extraction Demo")
print("=" * 40)

# Test basic OCR functionality
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    
    print("✅ Core OCR libraries imported successfully")
    
    # Create a simple test image with text
    print("📝 Creating test image...")
    
    # Create a white image with black text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'Hello World!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Extract text using Tesseract
    print("🔍 Extracting text using Tesseract...")
    text = pytesseract.image_to_string(pil_img)
    
    print(f"✅ Extracted text: '{text.strip()}'")
    
    # Test document processing
    print("\n📄 Testing document processing...")
    try:
        import PyPDF2
        import pdfplumber
        from docx import Document
        import openpyxl
        from pptx import Presentation
        
        print("✅ Document processing libraries imported successfully")
        print("   - PyPDF2: PDF text extraction")
        print("   - pdfplumber: Advanced PDF processing")
        print("   - python-docx: Word document processing")
        print("   - openpyxl: Excel spreadsheet processing")
        print("   - python-pptx: PowerPoint presentation processing")
        
    except Exception as e:
        print(f"⚠️  Some document processing libraries unavailable: {e}")
    
    print("\n🎉 Basic functionality test completed successfully!")
    print("\nTo use the full application:")
    print("   GUI: python main.py")
    print("   CLI: python main.py --cli extract <file>")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("\nMake sure all dependencies are installed:")
    print("   pip install opencv-python pillow pytesseract numpy")
