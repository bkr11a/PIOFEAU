import os
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PIOFE-Unrolling root path is: {PROJ_ROOT}")

# NOTE THESE MIGHT HAVE TO CHANGE!
DATA_DIR = os.path.join(PROJ_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')

MODELS_DIR = os.path.join(PROJ_ROOT, 'models')

REPORTS_DIR = os.path.join(PROJ_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
except ModuleNotFoundError:
    pass