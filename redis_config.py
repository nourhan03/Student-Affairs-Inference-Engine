from redis import Redis
import logging

logger = logging.getLogger(__name__)

def create_redis_client(max_retries=3):
    for attempt in range(max_retries):
        try:
            client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
            client.ping()  
            return client
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt+1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("All Redis connection attempts failed")
                raise
    
redis_client = create_redis_client() 