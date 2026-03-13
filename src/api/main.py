from fastapi import FastAPI
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(title="MarketPulse AI API")

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to MarketPulse AI API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
