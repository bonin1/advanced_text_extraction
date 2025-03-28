import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_spacy():
    """Setup spaCy and download required models."""
    try:
        logger.info("Checking for spaCy...")
        import spacy
        logger.info(f"spaCy version {spacy.__version__} is installed")
        
        # Try to load the model to see if it's installed
        try:
            model_name = "en_core_web_sm"
            spacy.load(model_name)
            logger.info(f"Model '{model_name}' is already installed")
        except OSError:
            # Model is not installed, so download it
            logger.info(f"Downloading model '{model_name}'...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            logger.info(f"Successfully downloaded model '{model_name}'")
            
            # Verify installation
            try:
                spacy.load(model_name)
                logger.info(f"Model '{model_name}' has been successfully installed and loaded")
            except Exception as e:
                logger.error(f"Failed to load model after installation: {e}")
                return False
                
        return True
    except ImportError:
        logger.error("spaCy is not installed. Installing spaCy...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            logger.info("Successfully installed spaCy")
            
            # Now try to download the model
            import spacy
            logger.info(f"Downloading model 'en_core_web_sm'...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("Successfully downloaded model 'en_core_web_sm'")
            return True
        except Exception as e:
            logger.error(f"Failed to install spaCy or download the model: {e}")
            return False

if __name__ == "__main__":
    print("Setting up NLP components for Advanced Data Extractor...")
    if setup_spacy():
        print("NLP setup completed successfully!")
    else:
        print("NLP setup encountered errors. Some features may not work properly.")
        print("You can try manually running: pip install spacy && python -m spacy download en_core_web_sm")
