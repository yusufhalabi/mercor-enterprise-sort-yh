import asyncio
import pandas as pd
import json
from app.core.ranking_engine import FastRankingEngine

async def test_ranking():
    # Load test data from the provided CSV
    df = pd.read_csv('processed_trial_data_anonymized.csv')
    
    # Prepare candidates
    candidates = []
    for _, row in df.iterrows():
        parsed_resume = json.loads(row['ParsedResume'])
        candidates.append({
            'resume_id': row['UserID'],
            'resume_text': json.dumps(parsed_resume, indent=2),
            'interview_text': row.get('ParsedTranscript', '')
        })
    
    # Initialize ranker
    ranker = FastRankingEngine()
    
    # Perform ranking
    rankings = await ranker.rank_candidates(
        candidates=candidates,
        role_query="software engineer",
        k=5
    )
    
    # Print results
    print("\nRankings:")
    for rank, (candidate_id, score) in enumerate(rankings[:5], 1):
        print(f"{rank}. {candidate_id} (score: {score})")

if __name__ == "__main__":
    asyncio.run(test_ranking()) 