from pydantic import BaseModel
from typing import List

class RankingRequest(BaseModel):
    """Request model for candidate ranking endpoint."""
    role_query: str
    k: int = 30

class RankingResponse(BaseModel):
    """Response model for candidate ranking endpoint."""
    ranked_candidates: List[str] 