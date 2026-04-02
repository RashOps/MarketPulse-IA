from fastapi import HTTPException, status

class MarketPulseError(Exception):
    """Base exception for MarketPulse AI application."""
    pass

class MarketDataNotFoundError(MarketPulseError):
    """Raised when no market data is found in the database."""
    pass

class ModelInferenceError(MarketPulseError):
    """Raised when model inference fails due to missing or corrupted artifacts."""
    pass

class IngestionError(MarketPulseError):
    """Raised during data collection from APIs or Scrapers."""
    pass

def handle_marketpulse_exception(exc: Exception) -> HTTPException:
    """
    Maps internal MarketPulse exceptions to FastAPI HTTPExceptions.
    """
    if isinstance(exc, MarketDataNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No market data available to process."
        )
    if isinstance(exc, ModelInferenceError):
        return HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail="ML Model not ready. Please trigger an update first."
        )
    
    # Generic Internal Server Error
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"An unexpected internal error occurred: {str(exc)}"
    )
