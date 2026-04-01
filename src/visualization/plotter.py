import os
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def plot_market_clusters(df: pd.DataFrame, save_html: bool = True) -> None:
    """
    Génère un Scatter Plot interactif utilisant les labels métier (Profiling).
    """
    if df is None or df.empty:
        return

    # On vérifie que la colonne issue du profiling est bien là
    if 'business_segment' not in df.columns:
        logger.error("La colonne 'business_segment' est manquante. Exécutez le profiling d'abord.")
        return

    logger.info("Génération du Scatter Plot interactif orienté Business...")

    fig = px.scatter(
        df,
        x='PCA_1',
        y='PCA_2',
        color='business_segment', # <-- Utilisation des vrais noms générés par ton algo
        hover_name='ticker',
        hover_data={
            'PCA_1': False, 
            'PCA_2': False,
            'cluster': False,
            'business_segment': False,
            'feature_volatility': ':.2f', 
            'feature_momentum': ':.2f',
            'metrics_volume': True
        },
        title="MarketPulse AI : Cartographie Stratégique du Marché",
        labels={
            "PCA_1": "Axe de Variance Dominante",
            "PCA_2": "Axe de Variance Secondaire",
            "business_segment": "Classification Algorithmique"
        },
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold 
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        title_font=dict(size=20, family="Courier New, monospace"),
        legend_title_text='Profils de Marché',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.show()

    if save_html:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        html_path = OUTPUT_DIR / "strategic_clusters.html"
        fig.write_html(str(html_path))
        logger.info("Graphique sauvegardé dans %s", html_path)


# ==========================================
# BLOC D'EXÉCUTION COMPLET (End-to-End)
# ==========================================
if __name__ == "__main__":
    from src.processing.cleaner import load_market_data
    from src.processing.features import engineer_features, scale_features
    from src.models.pca_model import apply_pca
    from src.models.clustering import apply_clustering
    
    # Importation de ton module de profiling
    from src.models.profiling import generate_cluster_profiles, assign_business_labels
    
    # 1. Pipeline ML
    df_raw = load_market_data()
    df_engineered = engineer_features(df_raw)
    df_scaled = scale_features(df_engineered, is_training=True)
    df_pca = apply_pca(df_scaled, n_components=2, is_training=True)
    df_final = apply_clustering(df_pca, max_k=5, is_training=True)
    
    if df_final is not None:
        # 2. Application du Profiling Business
        cluster_profiles = generate_cluster_profiles(df_final)
        business_labels = assign_business_labels(cluster_profiles)
        
        # MAPPING : On traduit les IDs mathématiques (0, 1, 2) en texte ("Haute Volatilité", etc.)
        # Si un ID n'a pas de label, on garde "Segment Inconnu" par sécurité
        df_final['business_segment'] = df_final['cluster'].map(business_labels).fillna("Segment Non Défini")
        
        # 3. Visualisation avec la nouvelle colonne
        plot_market_clusters(df_final)