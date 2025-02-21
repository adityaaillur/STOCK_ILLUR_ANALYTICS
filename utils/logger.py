import logging
from config import Config

def setup_logger():
    logging.basicConfig(
        filename=Config.LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("ultimate_app")


# Usage: 
# from utils.logger import setup_logger
# logger = setup_logger()
