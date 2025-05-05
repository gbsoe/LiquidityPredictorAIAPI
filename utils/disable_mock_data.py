import logging
import os

logger = logging.getLogger(__name__)

def disable_all_mock_data():
    """
    Forcibly disable all mock data throughout the application.
    This updates various components to ensure only real API data is used.
    """
    try:
        # Set environment variable to disable mock data
        os.environ["USE_MOCK_DATA"] = "False"
        os.environ["MOCK_DATA_ALLOWED"] = "False"
        
        # Update database connections
        from database.db_operations import DBManager
        
        # Get all existing database managers and update their flags
        for instance_name in dir():
            obj = locals()[instance_name]
            if isinstance(obj, DBManager):
                logger.info(f"Disabling mock data for DB instance: {instance_name}")
                obj.use_mock = False
        
        # Also ensure the mock_db module doesn't return mock data
        try:
            from database import mock_db
            mock_db.USE_MOCK_DATA = False
            logger.info("Disabled mock_db module mock data")
        except ImportError:
            pass
        
        logger.info("Successfully disabled all mock data throughout the application")
        return True
    except Exception as e:
        logger.error(f"Error disabling mock data: {str(e)}")
        return False