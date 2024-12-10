import asyncio
import time
import pandas as pd
import json
from app.core.ranking_engine import FastRankingEngine

def parse_resume(resume_str: str) -> dict:
    """Parse resume string to dict"""
    try:
        # Direct JSON parsing
        return json.loads(resume_str)
    except json.JSONDecodeError:
        try:
            # Try with single quote replacement
            cleaned = resume_str.replace("'", '"')
            return json.loads(cleaned)
        except:
            return None

async def test_parallel_ranking():
    # Load test data
    print("Loading test data...")
    df = pd.read_csv('processed_trial_data_anonymized.csv')
    
    # Prepare candidates
    candidates = []
    success_count = 0
    failed_count = 0
    
    for _, row in df.iterrows():
        parsed = parse_resume(row['ParsedResume'])
        if not parsed or 'data' not in parsed:
            failed_count += 1
            continue
            
        try:
            candidate = {
                'resume_id': row['UserID'],
                'resume_text': json.dumps(parsed['data']),
                'interview_text': row.get('ParsedTranscript', '')
            }
            candidates.append(candidate)
            success_count += 1
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Successfully parsed {success_count} resumes")
    print(f"Failed to parse {failed_count} resumes")
    
    if not candidates:
        print("No candidates loaded. Exiting.")
        return
    
    # Test with different batch sizes
    batch_sizes = [2, 3, 4]
    k = 30
    
    print("\nTesting parallel ranking with different batch sizes:")
    print("-" * 80)
    print(f"{'Batch Size':^12} | {'Time (s)':^10} | {'Comparisons':^12} | {'Avg Time/Comp':^15}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        try:
            # Initialize ranker
            ranker = FastRankingEngine(batch_size=batch_size)
            
            # Use first 20 candidates for testing
            test_candidates = candidates[:20]
            print(f"\nRanking {len(test_candidates)} candidates with batch size {batch_size}")
            
            # Time the ranking
            start_time = time.time()
            rankings = await ranker.rank_candidates(
                candidates=test_candidates,
                role_query="software engineer",
                k=min(k, len(test_candidates))
            )
            elapsed_time = time.time() - start_time
            
            # Calculate stats
            num_comparisons = len(ranker.comparison_cache)
            avg_time = elapsed_time / max(1, num_comparisons)
            
            print(f"{batch_size:^12d} | {elapsed_time:^10.2f} | {num_comparisons:^12d} | {avg_time:^15.4f}")
            
            # Show top 5 rankings
            print(f"\nTop 5 candidates (batch_size={batch_size}):")
            for rank, (candidate_id, score) in enumerate(rankings[:5], 1):
                print(f"{rank}. {candidate_id[:8]}... (score: {score:.3f})")
                
        except Exception as e:
            print(f"Error with batch size {batch_size}: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(test_parallel_ranking())