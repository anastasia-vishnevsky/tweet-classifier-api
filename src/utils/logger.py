import logging
import os

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logger = logging.getLogger("tweet_classifier")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.INFO)

# Create formatter and add it to handler
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)