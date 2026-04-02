import joblib
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import settings
from src.utils.logger import get_logger
from src.utils.db_client import get_db
from src.utils.exceptions import ModelInferenceError

logger = get_logger(__name__)

class MarketPulseModel:
    """
    Unified ML Pipeline for MarketPulse AI.
    Encapsulates Scaling, PCA, and Clustering into a single Scikit-Learn Pipeline.
    """

    def __init__(self, n_components: int = 2, max_clusters: int = 8):
        """
        Initializes the model with PCA and Clustering parameters.
        
        Args:
            n_components (int): Number of principal components for PCA.
            max_clusters (int): Maximum clusters for K-Means tuning.
        """
        self.n_components = n_components
        self.max_clusters = max_clusters
        self.pipeline: Optional[Pipeline] = None
        self.version: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _persist_metrics(self, k: int, silhouette: float, explained_variance: List[float]) -> None:
        """
        Persists training metrics into MongoDB for monitoring.
        """
        try:
            db = get_db()
            metrics_payload = {
                "version": self.version,
                "timestamp": datetime.now(timezone.utc),
                "params": {
                    "n_components": self.n_components,
                    "max_clusters": self.max_clusters
                },
                "results": {
                    "optimal_k": k,
                    "silhouette_score": silhouette,
                    "pca_explained_variance": explained_variance,
                    "total_variance_explained": sum(explained_variance)
                }
            }
            db["model-metrics"].insert_one(metrics_payload)
            logger.info("Training metrics persisted to MongoDB.")
        except Exception as e:
            logger.warning("Failed to persist model metrics: %s", e)

    def _find_optimal_k(self, X: pd.DataFrame) -> Tuple[int, float]:
        """
        Finds the optimal number of clusters and returns k and its silhouette score.
        """
        best_k = 2
        best_score = -1.0
        
        for k in range(2, self.max_clusters + 1):
            preprocessor = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.n_components))
            ])
            X_preprocessed = preprocessor.fit_transform(X)
            
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_preprocessed)
            score = silhouette_score(X_preprocessed, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                
        return best_k, best_score

    def train(self, df: pd.DataFrame) -> str:
        """
        Fits the entire pipeline and persists metrics.
        """
        features = ['feature_volatility', 'feature_momentum', 'metrics_volume']
        X = df[features]

        # 1. Tuning
        optimal_k, best_silhouette = self._find_optimal_k(X)

        # 2. Fit Final Pipeline
        pca_step = PCA(n_components=self.n_components)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca_step),
            ('kmeans', KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10))
        ])
        
        logger.info("Fitting unified pipeline with k=%s...", optimal_k)
        self.pipeline.fit(X)
        
        # 3. Extract and Persist Metrics
        explained_variance = pca_step.explained_variance_ratio_.tolist()
        self._persist_metrics(optimal_k, best_silhouette, explained_variance)
        
        # 4. Persistence
        model_name = f"marketpulse_model_{self.version}.pkl"
        model_path = settings.artifacts_dir / model_name
        latest_path = settings.artifacts_dir / "marketpulse_model_latest.pkl"
        
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.pipeline, latest_path)
        
        return str(model_path)

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs inference using the latest model.
        """
        latest_path = settings.artifacts_dir / "marketpulse_model_latest.pkl"
        
        if not latest_path.exists():
            raise ModelInferenceError("Latest model artifact not found.")
            
        self.pipeline = joblib.load(latest_path)
        features = ['feature_volatility', 'feature_momentum', 'metrics_volume']
        X = df[features]
        
        preprocessor = Pipeline(self.pipeline.steps[:-1])
        pca_results = preprocessor.transform(X)
        clusters = self.pipeline.predict(X)
        
        df_results = df.copy()
        for i in range(self.n_components):
            df_results[f'PCA_{i+1}'] = pca_results[:, i]
        df_results['cluster'] = clusters
        
        return df_results, pd.DataFrame(pca_results, columns=[f'PCA_{i+1}' for i in range(self.n_components)])

model_engine = MarketPulseModel(
    n_components=settings.pca_components,
    max_clusters=settings.max_clusters
)
