import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Définition des chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PCA_PATH = ARTIFACTS_DIR / "pca_model.pkl"

def apply_pca(df: pd.DataFrame, n_components: int = 2, is_training: bool = True) -> pd.DataFrame:
    """
    Applique la PCA sur les colonnes standardisées.
    Mode Training: Entraîne et sauvegarde le modèle.
    Mode Inférence: Charge le modèle et transforme les nouvelles données.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vide passé à la PCA.")
        return df

    # 1. Isolation de la Data Standardisée
    scaled_columns = [col for col in df.columns if col.endswith('_scaled')]
    if not scaled_columns:
        logger.error("Aucune colonne standardisée trouvée pour la PCA.")
        return df

    # 2. Alignement strict : on supprime les NaNs sur le DF principal pour garder la trace des Tickers
    df = df.dropna(subset=scaled_columns).copy()
    X_scaled = df[scaled_columns]

    # Création du dossier artifacts s'il manque
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_training:
        logger.info("Mode Entraînement : Fit et sauvegarde du modèle PCA (%s composantes)...", n_components)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)
        
        # Sauvegarde
        joblib.dump(pca, PCA_PATH)
        logger.info("Modèle PCA sauvegardé dans %s", PCA_PATH)
        
    else:
        logger.info("Mode Inférence : Chargement du modèle PCA existant...")
        if not PCA_PATH.exists():
            raise FileNotFoundError(f"Modèle PCA introuvable à {PCA_PATH}. Lancez l'entraînement d'abord.")
            
        pca = joblib.load(PCA_PATH)
        pca_result = pca.transform(X_scaled)

    # 3. Réinjection des Composantes Principales dans le DataFrame
    # On crée de nouvelles colonnes "PCA_1", "PCA_2", etc.
    for i in range(n_components):
        df[f'PCA_{i+1}'] = pca_result[:, i]

    logger.info("Transformation PCA réussie. Nouvelles colonnes ajoutées.")
    return df

# ==========================================
# BLOC DE TEST / EXÉCUTION
# ==========================================
if __name__ == "__main__":
    from src.processing.cleaner import load_market_data
    from src.processing.features import engineer_features, scale_features
    
    # Pipeline d'exécution
    df_raw = load_market_data()
    if df_raw is not None:
        df_engineered = engineer_features(df_raw)
        df_scaled = scale_features(df_engineered, is_training=True)
        
        # Application de la PCA
        df_final = apply_pca(df_scaled, n_components=2, is_training=True)
        
        print("\n--- Aperçu des données post-PCA ---")
        cols_to_show = ['ticker'] + [col for col in df_final.columns if col.startswith('PCA_')]
        print(df_final[cols_to_show].head())