import pandas as pd
from typing import Optional
from src.utils.logger import get_logger
from src.utils.db_client import get_db
from src.processing.features import engineer_features

logger = get_logger(__name__)

def load_market_data(days_back: int = 30) -> Optional[pd.DataFrame]:
    """
    Extracts raw data from MongoDB and converts it to a pandas DataFrame.
    
    Args:
        days_back (int): Number of days of historical data to retrieve. Default is 30.
        
    Returns:
        Optional[pd.DataFrame]: Processed dataframe, or None if extraction fails.
    """
    logger.info("Starting extraction of market data for the last %s days.", days_back)
    
    try:
        db = get_db()
        collection = db["raw-market-data"]
        
        # Note: In production, add a filter on timestamp using $gte
        cursor = collection.find({}) 
        
        data = list(cursor)
        
        if not data:
            logger.warning("No data found in 'raw_market_data' collection.")
            return None
            
        df = pd.json_normalize(data)
        
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        # Standardize column names (snake_case)
        df.columns = df.columns.str.replace(r'[\.\s]+', '_', regex=True).str.lower()

        # Type casting for string columns
        string_cols = ['ticker', 'company_name', 'source_type', 'metadata_status', 'metadata_message']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Type casting for numeric columns
        numeric_cols = [col for col in df.columns if 'price' in col or 'high' in col or 'low' in col or 'open' in col or 'close' in col]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Type casting for volume
        if 'metrics_volume' in df.columns:
            df['metrics_volume'] = pd.to_numeric(df['metrics_volume'], errors='coerce').fillna(0).astype(int)

        # Type casting for timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info("DataFrame successfully formatted: %s rows and %s columns.", df.shape[0], df.shape[1])
        return df
        
    except Exception as e:
        logger.error("Failed to load data from MongoDB: %s", e)
        return None

if __name__ == "__main__":
    from src.models.pipeline import model_engine
    
    df_raw = load_market_data()
    if df_raw is not None:
        df_engineered = engineer_features(df_raw)
        
        # Training test
        logger.info("Testing Pipeline Training...")
        model_engine.train(df_engineered)
        
        # Inference test
        logger.info("Testing Pipeline Inference...")
        df_final, _ = model_engine.predict(df_engineered)
        print(df_final.head())
        