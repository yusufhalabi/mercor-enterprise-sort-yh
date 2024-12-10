from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import pandas as pd
import json
import logging
from app.core.ranking_engine import FastRankingEngine

router = APIRouter()
ranking_engine = FastRankingEngine()
logger = logging.getLogger(__name__)

@router.post("/rank_candidates")
async def rank_candidates(
    file: UploadFile = File(...),
    role_query: str = Form(...),
    k: int = Form(default=5)
):
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        logger.info(f"Loaded CSV with {len(df)} rows")
        
        # Print first row for debugging
        logger.info(f"First row sample: {df.iloc[0].to_dict()}")
        
        # Prepare candidates
        candidates = []
        for _, row in df.iterrows():
            try:
                candidates.append({
                    'UserID': str(row['UserID']),
                    'ParsedResume': row['ParsedResume'],
                    'ParsedTranscript': row.get('ParsedTranscript', '')
                })
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
        
        logger.info(f"Processed {len(candidates)} candidates")
        logger.info(f"First candidate sample: {candidates[0] if candidates else 'No candidates'}")
        
        # Perform ranking
        rankings = await ranking_engine.rank_candidates(
            candidates=candidates,
            role_query=role_query,
            k=k
        )
        
        logger.info(f"Got rankings: {rankings}")
        
        # Return just the IDs
        result = [id for id, _ in rankings][:k]
        logger.info(f"Returning result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))