from datetime import datetime, timezone

from dateutil import parser

from src.utils.logger import get_logger

logger = get_logger(__name__)


def standardize_date(date_str: str) -> datetime | None:
    """
    Parse une date textuelle arbitraire vers un objet datetime UTC natif.
    Retourne None si la date est invalide ou absente.
    """
    if not date_str:
        return None
        
    try:
        # 1. Parsing intelligent de la chaîne de caractères
        parsed_date = parser.parse(date_str)
        
        # 2. Sécurisation : on s'assure que la date possède une timezone
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            
        # 3. Conversion absolue en UTC pour un stockage unifié dans MongoDB
        utc_date = parsed_date.astimezone(timezone.utc)
        return utc_date
        
    except Exception as e:
        logger.warning("Échec de la normalisation temporelle pour la chaîne: '%s' - %s", date_str, e)
        return None