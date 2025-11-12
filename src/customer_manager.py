
import boto3
from logger import setup_logger

logger = setup_logger()

def get_customers_from_parameter_store():
    """Get customer list from AWS Parameter Store"""
    try:
        ssm = boto3.client('ssm')
        response = ssm.get_parameter(Name='/automated-reports/customers')
        customers_string = response['Parameter']['Value']
        customers = [customer.strip() for customer in customers_string.split(',')]
        logger.info(f"Loaded {len(customers)} customers from Parameter Store")
        return customers
    except Exception as e:
        logger.error(f"Failed to get customers from Parameter Store: {e}")
        return ["Oxford_City_Sightseeing"]  # Fallback