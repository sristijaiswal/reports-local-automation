import boto3
import os
from logger import setup_logger

logger = setup_logger()

def save_file_to_s3(file_path, bucket, key):
    """Save local file to S3"""
    try:
        s3 = boto3.client('s3')
        s3.upload_file(file_path, bucket, key)
        logger.info(f"Successfully saved file to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Failed to save file to S3: {e}")
        return False

def save_reports_to_s3(customer_name, Results_Final, Results_CC, report_name):
    """Save both Final and Customer Copy reports to S3 with week-based structure"""
    try:
        s3_bucket = "automated-customer-reports"
        
        # Get the week folder name from Results_Final path
        week_folder = os.path.basename(os.path.dirname(Results_Final))
        
        # Save Final version
        final_excel_path_final = os.path.join(Results_Final, report_name)
        if os.path.exists(final_excel_path_final):
            s3_key_final = f"reports/{week_folder}/Final/{customer_name}_{report_name}"
            save_file_to_s3(final_excel_path_final, s3_bucket, s3_key_final)
            logger.info(f"Saved Final version: {report_name}")
        
        # Save Customer Copy version
        final_excel_path_cc = os.path.join(Results_CC, report_name.replace('.xlsx', '_CC.xlsx'))
        if os.path.exists(final_excel_path_cc):
            s3_key_cc = f"reports/{week_folder}/Customer Copy/{customer_name}_{report_name.replace('.xlsx', '_CC.xlsx')}"
            save_file_to_s3(final_excel_path_cc, s3_bucket, s3_key_cc)
            logger.info(f"Saved Customer Copy version: {report_name.replace('.xlsx', '_CC.xlsx')}")
            
        logger.info(f"All reports saved to S3: s3://{s3_bucket}/reports/{week_folder}/")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save reports to S3: {e}")
        return False