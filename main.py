"""
Advanced Text Extraction Software - Main Launcher
===============================================

This is the main entry point for the Advanced Text Extraction Software.
It provides options to launch either the GUI or CLI interface and handles
dependency checking and setup.
"""

import sys
import argparse
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    required_deps = [
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"), 
        ("pytesseract", "pytesseract"),
        ("numpy", "numpy"),
        ("PyPDF2", "PyPDF2"),
        ("pdfplumber", "pdfplumber"),
        ("python-docx", "docx"),
        ("openpyxl", "openpyxl"),
        ("python-pptx", "pptx")
    ]
    
    missing_deps = []
    
    for pkg_name, import_name in required_deps:
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(pkg_name)
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\\n💡 Install missing dependencies with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True


def check_optional_dependencies():
    """Check optional dependencies and warn if missing"""
    optional_deps = {
        "easyocr": "EasyOCR engine for better accuracy",
        "paddleocr": "PaddleOCR engine for Asian languages", 
        "customtkinter": "Modern GUI interface",
        "tkinterdnd2": "Drag and drop support in GUI",
        "rich": "Enhanced CLI interface",
        "selenium": "Advanced web scraping",
        "langdetect": "Language detection",
        "googletrans": "Translation support",
        "loguru": "Enhanced logging"
    }
    
    missing_optional = []
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append((dep, description))
    
    if missing_optional:
        print("⚠️  Optional dependencies not installed (some features may be limited):")
        for dep, desc in missing_optional:
            print(f"   - {dep}: {desc}")
        print("\\n💡 Install optional dependencies with:")
        deps_list = [dep for dep, _ in missing_optional]
        print(f"   pip install {' '.join(deps_list)}")
        print()


def check_tesseract():
    """Check if Tesseract OCR is installed and accessible"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is available")
        return True
    except Exception as e:
        print("⚠️  Tesseract OCR not found or not properly configured")
        print("   Tesseract is required for OCR functionality.")
        print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"   Error: {e}")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    
    # Install from requirements.txt if available
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("❌ requirements.txt not found")
        return False


def launch_gui():
    """Launch the GUI interface"""
    try:
        from gui import main as gui_main
        print("🚀 Launching GUI interface...")
        gui_main()
    except ImportError as e:
        print(f"❌ Cannot launch GUI: {e}")
        print("   Make sure all GUI dependencies are installed:")
        print("   pip install customtkinter tkinterdnd2")
        sys.exit(1)
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        sys.exit(1)


def launch_cli(args):
    """Launch the CLI interface"""
    try:
        from cli import main as cli_main
        
        # Replace sys.argv with CLI arguments
        original_argv = sys.argv
        sys.argv = ['cli.py'] + args
        
        try:
            cli_main()
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"❌ Cannot launch CLI: {e}")
        print("   Make sure all CLI dependencies are installed:")
        print("   pip install rich click")
        sys.exit(1)
    except Exception as e:
        print(f"❌ CLI launch failed: {e}")
        sys.exit(1)


def show_system_info():
    """Show system information and dependency status"""
    print("🔍 System Information")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Executable: {sys.executable}")
    print()
    
    print("🔧 Dependency Status")
    print("=" * 50)
    
    # Check required dependencies
    required_ok = check_dependencies()
    tesseract_ok = check_tesseract()
    
    print()
    check_optional_dependencies()
    
    if required_ok and tesseract_ok:
        print("✅ All required dependencies are available!")
    else:
        print("❌ Some required dependencies are missing.")
        
        if not required_ok:
            print("   Run 'python main.py --install' to install missing packages.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Text Extraction Software",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Launch GUI interface
  python main.py --cli extract file.pdf   # CLI extraction
  python main.py --info             # Show system information
  python main.py --install          # Install dependencies
        """
    )
    
    parser.add_argument(
        '--cli',
        nargs=argparse.REMAINDER,
        help='Launch CLI with specified arguments'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI interface (default)'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and dependency status'
    )
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install required dependencies'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check dependencies without launching'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("🚀 Advanced Text Extraction Software")
    print("=" * 50)
    
    # Check Python version first
    if not check_python_version():
        sys.exit(1)
    
    # Handle special commands
    if args.info:
        show_system_info()
        return
    
    if args.install:
        success = install_dependencies()
        if success:
            print("\\n✅ Installation completed! You can now run the application.")
        else:
            print("\\n❌ Installation failed. Please install dependencies manually.")
        return
    
    if args.check:
        required_ok = check_dependencies()
        tesseract_ok = check_tesseract()
        check_optional_dependencies()
        
        if required_ok and tesseract_ok:
            print("✅ All systems ready!")
            sys.exit(0)
        else:
            print("❌ Dependency issues found.")
            sys.exit(1)
    
    # Check dependencies before launching
    if not check_dependencies():
        print("\\n💡 Try running: python main.py --install")
        sys.exit(1)
    
    check_tesseract()  # Warn but don't exit
    check_optional_dependencies()  # Inform about optional features
    
    # Launch appropriate interface
    if args.cli is not None:
        launch_cli(args.cli)
    else:
        # Default to GUI
        launch_gui()


if __name__ == "__main__":
    main()
