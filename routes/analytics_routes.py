"""System analytics endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from routes.schemas import AnalyticsResponse
from src.analytics.metrics import get_system_analytics
from src.database.base import get_db

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("", response_model=AnalyticsResponse)
def analytics(db: Session = Depends(get_db)):
    """Returns knowledge-base-wide usage statistics: document counts, chunk
    counts, category distribution, and most-queried documents."""
    return AnalyticsResponse(**get_system_analytics(db))
