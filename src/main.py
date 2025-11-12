# main.py
from logger import setup_logger
from customer_manager import get_customers_from_parameter_store
from customer_processor import process_customer

logger = setup_logger()

def main():
    
    customers = get_customers_from_parameter_store()
    
    if not customers:
        logger.error("No customers found to process")
        return
    
    logger.info(f"Processing {len(customers)} customers sequentially")
    
    for customer in customers:
        try:
            process_customer(customer)
        except Exception as e:
            logger.error(f"Failed to process {customer}: {e}")
            continue  
    
    logger.info("All customers processing completed")

if __name__ == "__main__":
    main()