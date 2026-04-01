import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Définition des chemins pour les artifacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
KMEANS_PATH = ARTIFACTS_DIR / "kmeans_model.pkl"

def find_optimal_k(X: pd.DataFrame, max_k: int = 8) -> KMeans:
    """
    Recherche le modèle K-Means optimal en maximisant le Silhouette Score.
    """
    best_k = 2
    best_score = -1.0
    best_model = None

    logger.info("Début du tuning des hyperparamètres pour k in [2, %s]...", max_k)

    for k in range(2, max_k + 1):
        # init='k-means++' garantit une meilleure convergence initiale
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        # Calcul de la métrique d'évaluation
        score = silhouette_score(X, labels)
        logger.debug("Test k=%s | Silhouette Score = %.4f", k, score)

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    logger.info("Optimal k trouvé : %s (Silhouette Score: %.4f)", best_k, best_score)
    return best_model

def apply_clustering(df: pd.DataFrame, max_k: int = 8, is_training: bool = True) -> pd.DataFrame:
    """
    Applique l'algorithme K-Means sur les composantes principales (PCA).
    Mode Training: Trouve le meilleur k, entraîne et sauvegarde.
    Mode Inférence: Charge le modèle et assigne les nouveaux clusters.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vide passé au Clustering.")
        return df

    # On isole exclusivement les colonnes générées par la PCA
    pca_columns = [col for col in df.columns if col.startswith('PCA_')]
    if not pca_columns:
        logger.error("Aucune composante PCA trouvée. Lancez apply_pca() d'abord.")
        return df

    # Sécurité et alignement des index
    df = df.dropna(subset=pca_columns).copy()
    X_pca = df[pca_columns]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_training:
        logger.info("Mode Entraînement : Recherche du K-Means optimal...")
        # 1. Hyperparameter Tuning
        best_kmeans = find_optimal_k(X_pca, max_k=max_k)
        
        # 2. Assignation des clusters
        clusters = best_kmeans.labels_
        
        # 3. Sauvegarde (Model Persistence)
        joblib.dump(best_kmeans, KMEANS_PATH)
        logger.info("Modèle K-Means sauvegardé dans %s", KMEANS_PATH)
        
    else:
        logger.info("Mode Inférence : Chargement du modèle K-Means existant...")
        if not KMEANS_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable à {KMEANS_PATH}.")
            
        kmeans_model = joblib.load(KMEANS_PATH)
        # Prediction sans ré-entraînement
        clusters = kmeans_model.predict(X_pca)

    # 4. Injection du résultat métier dans le DataFrame
    df['cluster'] = clusters
    
    logger.info("Clustering terminé. Les actifs ont été segmentés.")
    return df

# ==========================================
# BLOC DE TEST / EXÉCUTION
# ==========================================
if __name__ == "__main__":
    from src.processing.cleaner import load_market_data
    from src.processing.features import engineer_features, scale_features
    from src.models.pca_model import apply_pca
    
    # Exécution de la pipeline complète
    df_raw = load_market_data()
    if df_raw is not None:
        df_engineered = engineer_features(df_raw)
        df_scaled = scale_features(df_engineered, is_training=True)
        df_pca = apply_pca(df_scaled, n_components=2, is_training=True)
        
        # Phase 4 : Le Clustering
        df_final = apply_clustering(df_pca, max_k=6, is_training=True)
        
        print("\n--- Aperçu de la Segmentation de Marché ---")
        cols_to_show = ['ticker', 'PCA_1', 'PCA_2', 'cluster']
        print(df_final[cols_to_show].head(10))
        
        print("\n--- Distribution des Clusters ---")
        print(df_final['cluster'].value_counts())