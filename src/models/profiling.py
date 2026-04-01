import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la distribution des clusters sur les variables d'origine
    pour en déduire une interprétation métier (Business Intelligence).
    """
    if df is None or 'cluster' not in df.columns:
        logger.error("DataFrame invalide ou clustering manquant.")
        return pd.DataFrame()

    # 1. Sélection des features brutes (interprétables par un humain)
    business_cols = ['feature_volatility', 'feature_momentum', 'metrics_volume']
    available_cols = [col for col in business_cols if col in df.columns]

    if not available_cols:
        logger.warning("Aucune colonne métier brute trouvée pour le profiling.")
        return pd.DataFrame()

    logger.info("Agrégation des statistiques par cluster...")

    # 2. Calcul des centroïdes réels (Moyenne ou Médiane)
    # L'utilisation de la médiane permet d'être robuste face aux anomalies dans un cluster
    profiles = df.groupby('cluster')[available_cols].median().reset_index()

    # 3. Calcul du "Poids" de chaque cluster (Market Share)
    cluster_sizes = df.groupby('cluster').size().reset_index(name='asset_count')
    
    # Fusion des informations
    profiles = pd.merge(profiles, cluster_sizes, on='cluster')

    # 4. Formattage pour la lisibilité métier
    if 'feature_volatility' in profiles.columns:
        profiles['feature_volatility'] = (profiles['feature_volatility'] * 100).round(2).astype(str) + '%'
    if 'feature_momentum' in profiles.columns:
        profiles['feature_momentum'] = (profiles['feature_momentum'] * 100).round(2).astype(str) + '%'
    if 'metrics_volume' in profiles.columns:
        profiles['metrics_volume'] = profiles['metrics_volume'].apply(lambda x: f"{int(x):,}")

    logger.info("Profiling terminé avec succès pour %s clusters.", len(profiles))
    return profiles

def assign_business_labels(profiles: pd.DataFrame) -> dict:
    """
    (Optionnel) Logique conditionnelle pour nommer automatiquement les clusters
    selon leurs statistiques. À adapter selon tes observations.
    """
    labels = {}
    for index, row in profiles.iterrows():
        cluster_id = row['cluster']
        # Exemple de logique métier basique :
        volatility = float(row['feature_volatility'].replace('%', ''))
        
        if volatility > 5.0:
            labels[cluster_id] = "Haute Volatilité (Risque Élevé)"
        elif volatility < 1.0:
            labels[cluster_id] = "Valeurs Refuges (Stables)"
        else:
            labels[cluster_id] = "Actifs Standard"
            
    return labels

# ==========================================
# BLOC DE TEST
# ==========================================
if __name__ == "__main__":
    from src.processing.cleaner import load_market_data
    from src.processing.features import engineer_features, scale_features
    from src.models.pca_model import apply_pca
    from src.models.clustering import apply_clustering

    # Exécution du pipeline complet
    df_raw = load_market_data()
    df_engineered = engineer_features(df_raw)
    df_scaled = scale_features(df_engineered, is_training=True)
    df_pca = apply_pca(df_scaled, n_components=2, is_training=True)
    df_final = apply_clustering(df_pca, max_k=5, is_training=True)

    # Lancement du Profiling
    cluster_profiles = generate_cluster_profiles(df_final)
    
    print("\n--- RAPPORT STRATÉGIQUE DES CLUSTERS ---")
    print(cluster_profiles.to_string(index=False))
    
    business_labels = assign_business_labels(cluster_profiles)
    print("\n--- LABELS SUGGÉRÉS ---")
    for k, v in business_labels.items():
        print(f"Cluster {k} : {v}")