import pandas as pd
import plotly.express as px
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)

def plot_market_clusters(df: pd.DataFrame, save_html: bool = True) -> None:
    """
    Generates an interactive scatter plot using business segment labels.

    Args:
        df (pd.DataFrame): Dataframe containing clustering and profiling results.
        save_html (bool): If True, saves the plot as an HTML file.
    """
    if df is None or df.empty:
        return

    if 'business_segment' not in df.columns:
        logger.error("Column 'business_segment' is missing. Run profiling first.")
        return

    logger.info("Generating interactive business-oriented Scatter Plot...")

    fig = px.scatter(
        df,
        x='PCA_1',
        y='PCA_2',
        color='business_segment',
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
        title="MarketPulse AI: Strategic Market Mapping",
        labels={
            "PCA_1": "Principal Variance Axis",
            "PCA_2": "Secondary Variance Axis",
            "business_segment": "Algorithmic Classification"
        },
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold 
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        title_font=dict(size=20, family="Courier New, monospace"),
        legend_title_text='Market Profiles',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    if save_html:
        output_dir = settings.PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "strategic_clusters.html"
        fig.write_html(str(html_path))
        logger.info("Plot saved successfully to %s", html_path)

if __name__ == "__main__":
    from src.processing.cleaner import load_market_data
    from src.processing.features import engineer_features
    from src.models.pipeline import model_engine
    from src.models.profiling import generate_cluster_profiles, assign_business_labels

    # End-to-End Test for Visualization
    df_raw = load_market_data()
    if df_raw is not None:
        df_engineered = engineer_features(df_raw)

        # Use unified pipeline (Inference Mode)
        try:
            df_final, _ = model_engine.predict(df_engineered)

            cluster_profiles = generate_cluster_profiles(df_final)
            business_labels = assign_business_labels(cluster_profiles)

            df_final['business_segment'] = df_final['cluster'].map(business_labels).fillna("Unknown Segment")

            plot_market_clusters(df_final, save_html=True)
        except Exception as e:
            logger.error("Visualization test failed: %s. Model might not be trained.", e)