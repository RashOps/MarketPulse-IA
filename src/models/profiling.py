import pandas as pd
from typing import Dict
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)

def generate_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes cluster distributions on original features to derive business intelligence.
    
    Args:
        df (pd.DataFrame): Dataframe containing clustering results and original features.
        
    Returns:
        pd.DataFrame: Formatted dataframe with cluster profiles (medians and asset counts).
    """
    if df is None or 'cluster' not in df.columns:
        logger.error("Invalid DataFrame or missing clustering column.")
        return pd.DataFrame()

    # Features interpreted by business users
    business_cols = ['feature_volatility', 'feature_momentum', 'metrics_volume']
    available_cols = [col for col in business_cols if col in df.columns]

    if not available_cols:
        logger.warning("No raw business columns found for profiling.")
        return pd.DataFrame()

    logger.info("Aggregating statistics per cluster...")

    # Calculate real centroids (Medians are robust to outliers)
    profiles = df.groupby('cluster')[available_cols].median().reset_index()

    # Calculate Market Share (Asset count per cluster)
    cluster_sizes = df.groupby('cluster').size().reset_index(name='asset_count')
    
    profiles = pd.merge(profiles, cluster_sizes, on='cluster')

    # Formatting for business readability
    if 'feature_volatility' in profiles.columns:
        profiles['feature_volatility_pct'] = (profiles['feature_volatility'] * 100).round(2).astype(str) + '%'
    if 'feature_momentum' in profiles.columns:
        profiles['feature_momentum_pct'] = (profiles['feature_momentum'] * 100).round(2).astype(str) + '%'
    if 'metrics_volume' in profiles.columns:
        profiles['metrics_volume_formatted'] = profiles['metrics_volume'].apply(lambda x: f"{int(x):,}")

    logger.info("Profiling completed for %s clusters.", len(profiles))
    return profiles

def assign_business_labels(profiles: pd.DataFrame) -> Dict[int, str]:
    """
    Assigns descriptive business labels to clusters based on their statistical profiles.
    
    Args:
        profiles (pd.DataFrame): Dataframe containing cluster-wise statistics.
        
    Returns:
        Dict[int, str]: Mapping of cluster IDs to business segment names.
    """
    labels = {}
    for _, row in profiles.iterrows():
        cluster_id = int(row['cluster'])
        
        # Heuristic-based labeling logic
        volatility = float(row.get('feature_volatility', 0))
        momentum = float(row.get('feature_momentum', 0))
        
        if volatility > 0.05: # > 5% daily volatility
            labels[cluster_id] = "High Volatility (Speculative)"
        elif volatility < 0.01: # < 1% daily volatility
            labels[cluster_id] = "Safe Haven (Low Risk)"
        elif momentum > 0.02:
            labels[cluster_id] = "Bullish Momentum"
        else:
            labels[cluster_id] = "Standard Market Assets"
            
    return labels

if __name__ == "__main__":
    from src.processing.cleaner import load_market_data
    from src.processing.features import engineer_features
    from src.models.pipeline import model_engine

    # End-to-End Test for Profiling
    logger.info("Running standalone profiling test...")
    df_raw = load_market_data()
    if df_raw is not None:
        df_engineered = engineer_features(df_raw)
        
        # Use unified pipeline (Inference Mode for test)
        try:
            df_final, _ = model_engine.predict(df_engineered)
            
            cluster_profiles = generate_cluster_profiles(df_final)
            print("\n--- STRATEGIC CLUSTER REPORT ---")
            print(cluster_profiles.to_string(index=False))
            
            business_labels = assign_business_labels(cluster_profiles)
            print("\n--- SUGGESTED BUSINESS LABELS ---")
            for k, v in business_labels.items():
                print(f"Cluster {k} : {v}")
        except Exception as e:
            logger.error("Profiling test failed: %s. Did you train the model first?", e)
