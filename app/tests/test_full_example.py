import asyncio
import pandas as pd
from app.core.ranking_engine import FastRankingEngine

async def main():
    try:
        # Load the dataset
        df = pd.read_csv('processed_trial_data_anonymized.csv')
        
        # Convert the data into the expected format
        candidates = []
        for _, row in df.iterrows():
            candidates.append({
                'resume_id': row['UserID'],
                'resume_text': row['ParsedResume'],
                'interview_text': ''
            })

        # Initialize ranking engine
        ranking_engine = FastRankingEngine()
        
        # Example role query - modify this based on what you're looking for
        role_query = """
        We are looking for a skilled software engineer with:
        - Strong programming skills in languages like Python, Java, or JavaScript
        - Experience with web development and modern frameworks
        - Good understanding of data structures and algorithms
        - Experience with cloud platforms (AWS/GCP/Azure)
        - Strong problem-solving abilities
        """

        # Rank the candidates
        ranked_candidates = await ranking_engine.rank_candidates(candidates, role_query, k=30)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(ranked_candidates, columns=['UserID', 'Score'])
        
        # Save to CSV
        results_df.to_csv('top_30_candidates.csv', index=False)
        print(f"Successfully ranked candidates and saved top 30 to 'top_30_candidates.csv'")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
