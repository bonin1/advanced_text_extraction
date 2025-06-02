#!/usr/bin/env python3

"""
Advanced Text Extraction Software - Final Demo
==============================================

This script demonstrates the complete functionality of our 
advanced text extraction software including:

1. Text extraction from images using multiple OCR engines
2. Document processing (PDF, Word, Excel, PowerPoint)
3. Advanced image preprocessing
4. Confidence scoring and quality assessment
5. Export capabilities

The software now includes:
- Multi-engine OCR (Tesseract + EasyOCR)
- Modern GUI with drag-and-drop
- Powerful CLI with batch processing
- Advanced image preprocessing
- Document format support
- Export options (TXT, JSON, CSV)
- Extraction history and caching
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_image():
    """Create a test image with text for demonstration"""
    print("📝 Creating test image with text...")
    
    # Create a white background
    img = Image.new('RGB', (600, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add text content
    text_lines = [
        "Advanced Text Extraction Demo",
        "",
        "This is a sample text document",
        "created for testing OCR capabilities.",
        "",
        "Features tested:",
        "• Multi-engine OCR processing",
        "• Image preprocessing",
        "• Confidence scoring",
        "• Language detection"
    ]
    
    # Try to use a better font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 28)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        title_font = font
    
    y = 30
    for i, line in enumerate(text_lines):
        if i == 0:  # Title
            draw.text((50, y), line, fill='black', font=title_font)
            y += 40
        else:
            draw.text((50, y), line, fill='black', font=font)
            y += 25
    
    # Save the test image
    test_image_path = "test_document.png"
    img.save(test_image_path)
    print(f"✅ Test image saved as: {test_image_path}")
    return test_image_path

def demo_text_extraction():
    """Demonstrate text extraction functionality"""
    print("\n🔍 Advanced Text Extraction Demo")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    
    try:        # Import our text extractor
        from text_extractor import AdvancedTextExtractor
        
        print("\n🚀 Initializing Advanced Text Extractor...")
        extractor = AdvancedTextExtractor()
        
        print(f"\n📄 Extracting text from: {test_image}")
        result = extractor.extract_from_file(test_image)
        
        print("\n📝 Extraction Results:")
        print("-" * 30)
        print(f"Extracted Text:\n{result.text}")
        print(f"\nConfidence Score: {result.confidence:.2f}")
        print(f"Detected Language: {result.language}")
        print(f"Processing Engine: {result.metadata.get('engine', 'Unknown')}")
        print(f"Processing Time: {result.metadata.get('processing_time', 'Unknown')}")
          # Test different formats
        print("\n💾 Testing Export Formats...")
        
        # Export as JSON
        extractor.export_results([result], "extraction_result.json", format="json")
        print("✅ JSON export: extraction_result.json")
        
        # Export as CSV
        extractor.export_results([result], "extraction_result.csv", format="csv")
        print("✅ CSV export: extraction_result.csv")
          # Export as TXT
        extractor.export_results([result], "extraction_result.txt", format="txt")
        print("✅ TXT export: extraction_result.txt")
        
        print(f"\n📊 OCR functionality demonstrated successfully!")
        print(f"Available features: Text extraction, Export formats, Database caching")
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo use the full application:")
        print("  GUI Mode:  python main.py")
        print("  CLI Mode:  python main.py --cli extract <file>")
        print("  Help:      python main.py --help")
        
        # Cleanup
        if os.path.exists(test_image):
            os.remove(test_image)
            print(f"\n🧹 Cleaned up test file: {test_image}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def show_features():
    """Display the complete feature list"""
    print("\n🌟 Advanced Text Extraction Software Features")
    print("=" * 55)
    
    features = {
        "🔍 OCR Engines": [
            "• Tesseract OCR (Google's OCR engine)",
            "• EasyOCR (Deep learning-based OCR)",
            "• Automatic engine selection based on image type",
            "• Confidence scoring for quality assessment"
        ],
        "📄 Document Support": [
            "• PDF files (text extraction + OCR fallback)",
            "• Microsoft Word (.docx)",
            "• Microsoft Excel (.xlsx, .xls)",
            "• Microsoft PowerPoint (.pptx)",
            "• Plain text files (.txt)",
            "• Images (PNG, JPG, JPEG, TIFF, BMP)"
        ],
        "🖼️ Image Processing": [
            "• Automatic image preprocessing",
            "• Noise reduction and denoising",
            "• Skew correction and deskewing",
            "• Contrast enhancement",
            "• Binarization for better OCR",
            "• Shadow removal"
        ],
        "🎯 AI Features": [
            "• Language detection",
            "• Quality assessment and scoring",
            "• Intelligent text post-processing",
            "• Metadata extraction",
            "• Batch processing with progress tracking"
        ],
        "💾 Data Management": [
            "• SQLite database for extraction history",
            "• File hashing for duplicate detection",
            "• Caching for improved performance",
            "• Multiple export formats (TXT, JSON, CSV)"
        ],
        "🖥️ User Interfaces": [
            "• Modern dark-themed GUI with CustomTkinter",
            "• Drag-and-drop file support",
            "• Rich CLI with progress bars and tables",
            "• Batch processing capabilities",
            "• Real-time preview and results"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}")
        for item in items:
            print(f"  {item}")
    
    print(f"\n📈 Performance & Scalability:")
    print(f"  • Multi-threaded processing for large batches")
    print(f"  • Memory-efficient handling of large documents")
    print(f"  • Progress tracking and cancellation support")
    print(f"  • Error handling and recovery")
    
    print(f"\n🔧 Technical Specifications:")
    print(f"  • Python 3.8+ compatibility")
    print(f"  • Cross-platform support (Windows, macOS, Linux)")
    print(f"  • Modular architecture for easy extension")
    print(f"  • Enterprise-ready with comprehensive logging")

if __name__ == "__main__":
    print("🚀 Advanced Text Extraction Software")
    print("🔥 Production-Ready Enterprise Solution")
    print("=" * 55)
    
    # Show features
    show_features()
    
    # Run demo
    demo_text_extraction()
