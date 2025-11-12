import os
import boto3
from pathlib import Path

class CloudConfig:
    S3_BUCKET = os.getenv('S3_BUCKET', 'automated-customer-reports')
    TMP_DIR = Path('/tmp')
    
    @classmethod
    def get_s3_client(cls):
        return boto3.client('s3')