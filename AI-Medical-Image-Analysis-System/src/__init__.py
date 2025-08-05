# Medical Image Analysis System
# Version: 1.0.0
# Author: Saniya
# Description: AI-powered medical image analysis with automated report generation

__version__ = "1.0.0"
__author__ = "[Your Name]"
__description__ = "AI Medical Image Analysis System"

# Import main modules for easier access
from . import model_utils
from . import report_generator
from . import image_processor
from . import pdf_utils
from . import ui_components

__all__ = [
    'model_utils',
    'report_generator', 
    'image_processor',
    'pdf_utils',
    'ui_components'
]