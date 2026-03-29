import pandas as pd
from typing import Optional
from src.utils.logger import get_logger
from src.utils.db_client import get_db

logger = get_logger(__name__)

def load_market_data(days_back: int = 30) -> Optional[pd.DataFrame]:
    """
    Extrait les données brutes de MongoDB et les convertit en DataFrame.
    """
    logger.info("Début de l'extraction des données de marché sur %s jours.", days_back)
    
    try:
        db = get_db()
        collection = db["raw-market-data"]
        
        # En production, on ajouterait un filtre sur la date ici avec $gte
        # cursor = collection.find({"timestamp": {"$gte": ...}})
        cursor = collection.find({}) 
        
        data = list(cursor)
        
        if not data:
            logger.warning("Aucune donnée trouvée dans la collection 'raw_market_data'.")
            return None
            
        df = pd.json_normalize(data)
        
        # 1. Nettoyage de l'ID MongoDB
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        # 2. Standardisation des noms de colonnes (Snake Case)
        # Remplace les espaces et les points par des underscores, et met tout en minuscules
        df.columns = df.columns.str.replace(r'[\.\s]+', '_', regex=True).str.lower()

        # 3. Forçage des types (Type Casting)
        # Colonnes textuelles
        string_cols = ['ticker', 'company_name', 'source_type', 'metadata_status', 'metadata_message']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Colonnes numériques (Prix)
        numeric_cols = [col for col in df.columns if 'price' in col or 'high' in col or 'low' in col or 'open' in col or 'close' in col]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Transforme les erreurs en NaN

        # Volume (Entier)
        if 'metrics_volume' in df.columns:
            df['metrics_volume'] = pd.to_numeric(df['metrics_volume'], errors='coerce').fillna(0).astype(int)

        # Timestamp (Format Date)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info("DataFrame formaté avec succès : %s lignes et %s colonnes.", df.shape[0], df.shape[1])
        return df
        
    except Exception as e:
        logger.error("Échec du chargement des données depuis MongoDB : %s", e)
        return None

if __name__ == "__main__":
    df = load_market_data()
    if df is not None:
        print(df.head())
        df.info()
        # print(df.describe())
        