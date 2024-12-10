# test_basic.py

import json
import asyncio
from app.core.llm import create_comparison_prompt, perform_llm_comparison

async def test_candidate_comparison():
    # Sample candidate data from CSV
    candidate_a = {
        "UserID": "27e911b3-8640-4ebb-b41b-fd23f7ea3ac4",
        "ParsedResume": "{'data': {'skills': ['Python', 'Git/Github'], 'education': [{'degree': 'Bachelor of Technology', 'major': 'Mechanical Engineering'}]}}",
        "ParsedTranscript": "Candidate A interview transcript..."
    }

    candidate_b = {
        "UserID": "7611250b-fc58-4f0d-a5d2-57cf5358059a",
        "ParsedResume": "{'data': {'skills': ['C', 'C++', 'Linux Kernel'], 'education': [{'degree': 'BE', 'major': 'Electronics and Communication'}]}}",
        "ParsedTranscript": "Candidate B interview transcript..."
    }

    # Define the role query
    role_query = "Software Engineer"

    # Create the prompt using the function from llm.py
    prompt = create_comparison_prompt(role_query, candidate_a, candidate_b)

    # Get the LLM's decision
    result = await perform_llm_comparison(prompt)
    
    print(f"Candidate A is better: {result}")

if __name__ == "__main__":
    asyncio.run(test_candidate_comparison())