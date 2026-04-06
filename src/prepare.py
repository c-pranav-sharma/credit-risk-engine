import pandas as pd
import hashlib
import os
from dotenv import load_dotenv

# Load the salt from the .env file
load_dotenv()
SALT = os.getenv('PII_SALT', 'fallback_salt_if_missing').encode()

def mask_id(customer_id):
    """Securely hashes ID to maintain GDPR/CCPA compliance."""
    return hashlib.sha256(str(customer_id).encode() + SALT).hexdigest()

def prepare_data(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Apply SHA-256 masking to the primary key
    if 'SK_ID_CURR' in df.columns:
        print("Masking PII (SK_ID_CURR)...")
        df['SK_ID_CURR'] = df['SK_ID_CURR'].apply(mask_id)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Hardened interim data saved to {output_path}")

if __name__ == '__main__':
    # Define paths relative to the project root
    RAW_DATA_PATH = 'data/raw/application_train.csv'
    INTERIM_DATA_PATH = 'data/interim/application_masked.csv'
    
    prepare_data(RAW_DATA_PATH, INTERIM_DATA_PATH)