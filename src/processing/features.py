import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates normalized financial indicators for the ML model.

    Args:
        df (pd.DataFrame): Raw market data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame passed to Feature Engineering.")
        return df

    try:
        # Volatility: (High - Low) / Open
        df['feature_volatility'] = (df['metrics_dayhigh'] - df['metrics_daylow']) / df['metrics_open']

        # Momentum: (Current - Open) / Open
        df['feature_momentum'] = (df['metrics_current_price'] - df['metrics_open']) / df['metrics_open']

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['feature_volatility', 'feature_momentum'], inplace=True)

        logger.info("Feature Engineering complete.")
        return df

    except KeyError as e:
        logger.error("Missing column for Feature Engineering: %s", e)
        return df