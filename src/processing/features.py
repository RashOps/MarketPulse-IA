import os
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Génère des indicateurs financiers normalisés pour le modèle ML."""
    if df is None or df.empty:
        logger.warning("DataFrame vide passé au Feature Engineering.")
        return df

    try:
        # 1. Volatilité (High - Low) / Open
        df['feature_volatility'] = (df['metrics_dayhigh'] - df['metrics_daylow']) / df['metrics_open']
        
        # 2. Momentum (Current - Open) / Open
        df['feature_momentum'] = (df['metrics_current_price'] - df['metrics_open']) / df['metrics_open']
        
        # Remplacement des infinis par des NaN (cas d'une ouverture à 0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Suppression des lignes où le calcul a échoué
        df.dropna(subset=['feature_volatility', 'feature_momentum'], inplace=True)

        logger.info("Feature Engineering terminé. Nouvelles variables calculées.")
        return df

    except KeyError as e:
        logger.error("Colonne manquante pour le Feature Engineering : %s", e)
        return df

# --------------------------------------------------------------------------------------------
# Définition du chemin absolu vers le dossier des artefacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCALER_PATH = ARTIFACTS_DIR / "standard_scaler.pkl"

def scale_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Applique le StandardScaler. 
    Sauvegarde le scaler en mode training, le charge en mode inférence.
    """
    if df is None or df.empty:
        return df

    features_to_scale = ['feature_volatility', 'feature_momentum', 'metrics_volume']
    
    # Sécurité : vérifier que les colonnes existent
    if not all(col in df.columns for col in features_to_scale):
        logger.error("Colonnes manquantes pour la standardisation.")
        return df

    # Création du dossier artifacts s'il n'existe pas
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if is_training:
        logger.info("Mode Entraînement : Fit et sauvegarde du Scaler...")
        scaler = StandardScaler()
        # FIT + TRANSFORM
        scaled_data = scaler.fit_transform(df[features_to_scale])
        
        # Sauvegarde de l'objet mathématique
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Scaler sauvegardé avec succès dans %s", SCALER_PATH)
        
    else:
        logger.info("Mode Inférence : Chargement du Scaler existant...")
        if not SCALER_PATH.exists():
            logger.critical("Aucun Scaler trouvé ! Lancez le mode training d'abord.")
            raise FileNotFoundError(f"Scaler introuvable à {SCALER_PATH}")
            
        # Chargement de l'objet
        scaler = joblib.load(SCALER_PATH)
        # TRANSFORM ONLY (On ne recalcule pas la moyenne/variance)
        scaled_data = scaler.transform(df[features_to_scale])

    # Injection des données standardisées dans le DataFrame
    for i, col in enumerate(features_to_scale):
        df[f"{col}_scaled"] = scaled_data[:, i]

    return df