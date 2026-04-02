# MarketPulse AI 📈

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-05998b.svg)](https://fastapi.tiangolo.com/)
[![Package Manager](https://img.shields.io/badge/uv-managed-6366f1.svg)](https://github.com/astral-sh/uv)
[![Database](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)](https://www.mongodb.com/)

**Real-time Market Segmentation Engine using Unsupervised Learning.**

MarketPulse AI is a high-performance market segmentation engine built on unsupervised learning. It automates market data collection (API & Web Scraping), performs advanced feature engineering, and applies a unified **Standardization → PCA → K-Means** pipeline to detect emerging market regimes and strategic asset clusters.

The system is designed for professional environments, featuring strict typing, automated model versioning, and real-time performance monitoring.

---

### 1. Key Features

- **Automated Ingestion**: Dynamic ticker discovery (S&P 500) and multi-source financial news collection.
- **Unified ML Pipeline**: Encapsulated Scikit-Learn pipeline ensuring strict parity between training and inference.
- **Dynamic Hyper-parameter Tuning**: Automatic selection of the optimal number of clusters ($k$) based on Silhouette scores.
- **Business Intelligence**: Automated cluster profiling and strategic labeling for human-readable insights.
- **Monitoring & Observability**: Real-time tracking of ML metrics (explained variance, silhouette) persisted in MongoDB.

---

### 2. Technical Stack

- **Runtime**: [Bun](https://bun.sh/) (Tooling) & [Python 3.12+](https://www.python.org/)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (Fast, reliable dependency management)
- **Backend Framework**: FastAPI (Pydantic v2)
- **ML / Data Science**: 
  - `scikit-learn` (StandardScaler, PCA, KMeans)
  - `pandas`, `numpy` (Advanced data manipulation)
- **Infrastructure**:
  - **Database**: MongoDB Atlas (Storage of raw data, news, and ML metrics)
  - **Configuration**: `pydantic-settings` (Environment-based centralized config)
- **Scraping**: `BeautifulSoup4`, `httpx`, `yfinance`

---

### 3. Architecture & Project Structure

The codebase follows a modular architecture aligned with Senior Data Science standards.

```bash
MarketPulse_AI/
├── artifacts/          # Versioned ML models (.pkl)
├── logs/               # Application & Audit logs
├── src/
│   ├── api/            # FastAPI layer (Routes, Pydantic schemas)
│   ├── config.py       # Centralized settings & environment management
│   ├── ingestion/      # Data acquisition (Yahoo Finance, RSS Scrapers)
│   ├── models/         # ML Logic (Unified Pipeline, Business Profiling)
│   ├── processing/     # Data cleaning & Feature engineering
│   └── utils/          # Core utilities (Logger, DB Client, Custom Exceptions)
├── pyproject.toml      # uv-managed dependencies
└── README.md           # This documentation
```

#### Unified Pipeline Detail
Unlike traditional prototypes, MarketPulse AI uses a single `sklearn.pipeline.Pipeline` object. This prevents **Data Leakage** by ensuring that the `StandardScaler` and `PCA` parameters used during training are exactly the same during real-time inference.

---

### 4. Installation & Setup

#### 4.1 Prerequisites
- Python 3.12 or higher.
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- A running MongoDB instance (Local or Atlas).

#### 4.2 Clone & Install
```bash
git clone <repository_url>
cd MarketPulse_AI
uv sync
```

#### 4.3 Environment Configuration
Create a `.env` file in the root directory:
```env
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
DB_NAME="marketpulse"
CORS_ORIGINS='["http://localhost:3000"]'
```

#### 4.4 Run the API
```bash
uv run uvicorn src.api.main:app --reload
```
API Documentation available at: `http://127.0.0.1:8000/docs`

---

### 5. API Endpoints (Highlights)

- **`GET /market-segments`**: Retrieves real-time asset clustering with business labels and PCA projections.
- **`POST /trigger-update`**: Forces a background retraining of the ML pipeline with new data.
- **`GET /monitoring/latest-metrics`**: Returns the health status of the latest model (Silhouette score, PCA variance).
- **`GET /market-news`**: Fetches the latest financial news feed (Yahoo Finance / Investing.com).

---

### 6. Roadmap

- [x] **v0.1**: Initial prototype and API.
- [x] **v0.2**: Unified Pipeline refactoring and Model Versioning.
- [x] **v0.3**: ML Metrics persistence and Custom Error Handling.
- [ ] **v0.4**: Integration with a Next.js / TailwindCSS Dashboard.
- [ ] **v0.5**: Advanced Anomaly Detection using Isolation Forests.

---

### 7. Development Standards

- **Typing**: Strict type hints enforced throughout the project.
- **Documentation**: All docstrings follow the **Google-style** in English.
- **Logging**: Rotating file logs for production auditability.
- **ROI Driven**: Every technical decision is linked to data reliability and business scalability.
