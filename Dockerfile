# --- STAGE 1: Build & Dependency Resolution ---
    FROM python:3.12-slim AS builder

    # Install uv for fast dependency resolution
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    
    WORKDIR /app
    
    # Enable bytecode compilation for faster startups
    ENV UV_COMPILE_BYTECODE=1
    
    # Copy ONLY project definition files first (Leveraging Docker Layer Caching)
    COPY pyproject.toml uv.lock ./
    
    # Install dependencies into a virtual environment (Strict lockfile compliance)
    RUN uv sync --frozen --no-install-project --no-dev --no-cache
    
    # --- STAGE 2: Final Runtime Image ---
    FROM python:3.12-slim
    
    # Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="/app" \
        PATH="/app/.venv/bin:$PATH"
    
    WORKDIR /app
    
    # Install curl strictly for the Docker healthcheck, then clean apt cache to save space
    RUN apt-get update && apt-get install -y --no-install-recommends curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Create a non-privileged user/group
    RUN groupadd -r appgroup && useradd -r -g appgroup appuser
    
    # Copy virtual environment and source code
    COPY --from=builder /app/.venv /app/.venv
    COPY src/ /app/src/
    
    # Create specific directories and strictly assign ownership to the appuser
    RUN mkdir -p /app/artifacts /app/logs && \
        chown -R appuser:appgroup /app
    
    # Drop root privileges
    USER appuser
    
    EXPOSE 8000
    
    CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]