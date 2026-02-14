import os
from dotenv import load_dotenv

# load variables from .env into the environment
load_dotenv()  

# get environment variables
host_dss_url = os.getenv("HOST_DSS_URL")
