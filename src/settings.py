import os
import requests
import base64
import datetime
import json
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv
from cloud_config import CloudConfig

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / '.env')

class Config:
    # API Credentials
    API_USER = os.getenv("API_USERNAME")
    API_PASS = os.getenv("API_PASSWORD")
    
    # Download Configuration - Use cloud temp directory
    DOWNLOAD_DIR = CloudConfig.TMP_DIR / "downloads"
    CHUNK_SIZE = 65536  
    MAX_WORKERS = 3
    
    @classmethod
    def get_auth_headers(cls):
        """Generates Basic Auth headers from credentials"""
        auth_string = f"{cls.API_USER}:{cls.API_PASS}"
        encoded_auth = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        return {
            'Authorization': f'Basic {encoded_auth}',
            'Content-Type': 'application/json'
        }
    
    @classmethod
    def setup_dirs(cls):
        """Ensure required directories exist"""
        cls.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DOWNLOAD_DIR

# Initialize directories on import
Config.setup_dirs()