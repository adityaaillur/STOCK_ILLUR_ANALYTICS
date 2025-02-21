import os

class Config:
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
    REPORT_PATH = os.path.join(DATA_DIR, "reports")
    LOG_FILE = os.path.join(BASE_DIR, "app.log")
