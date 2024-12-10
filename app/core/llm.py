import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Create a thread pool for parallel API calls
thread_pool = ThreadPoolExecutor(max_workers=10)

def prepare_candidate_text(resume_text: str, interview_text: str) -> str:
    """Prepare candidate text for comparison."""
    if not interview_text:
        interview_text = "No interview transcript available."
    
    # Clean and format the text
    resume_text = resume_text.strip()
    interview_text = interview_text.strip()
    
    return f"{resume_text}\n\n{interview_text}"

def create_comparison_prompt(role_query: str, candidate_a: dict, candidate_b: dict) -> str:
    """Create prompt for LLM comparison."""
    try:
        # Parse the JSON strings into dictionaries
        a_resume = json.loads(candidate_a['ParsedResume'])
        b_resume = json.loads(candidate_b['ParsedResume'])
        
        # Extract relevant information
        a_skills = a_resume['data']['skills']
        a_education = a_resume['data']['education']
        a_transcript = candidate_a.get('ParsedTranscript', 'No interview transcript available.')
        
        b_skills = b_resume['data']['skills']
        b_education = b_resume['data']['education']
        b_transcript = candidate_b.get('ParsedTranscript', 'No interview transcript available.')
        
        # Create formatted text for each candidate
        candidate_a_text = f"""
        Skills: {', '.join(a_skills)}
        Education: {json.dumps(a_education, indent=2)}
        Interview: {a_transcript}
        """
        
        candidate_b_text = f"""
        Skills: {', '.join(b_skills)}
        Education: {json.dumps(b_education, indent=2)}
        Interview: {b_transcript}
        """
        
        # Create the comparison prompt
        prompt = f"""You are a technical recruiter evaluating candidates for the role of {role_query}.
        Compare these two candidates and determine which is better suited for the role.
        
        Candidate A:
        {candidate_a_text}
        
        Candidate B:
        {candidate_b_text}
        
        Respond with ONLY "A" or "B" to indicate which candidate is better suited for the role."""
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error creating comparison prompt: {str(e)}")
        raise

def zero_shot_no_reasoning(role, candidateA_transcript, candidateB_transcript, candidateA_resume, candidateB_resume):
    system_prompt = f"""You are a strict JSON-only response system. You must ONLY output valid JSON.
    
    Compare these two candidates for the role of {role}. Evaluate their interviews and resumes:

    Interview of Candidate A: {candidateA_transcript}
    Interview of Candidate B: {candidateB_transcript}

    Resume of Candidate A: {candidateA_resume}
    Resume of Candidate B: {candidateB_resume}

    Respond with ONLY this JSON format, no other text:
    {{"winner": "A"}} or {{"winner": "B"}}
    """
    return system_prompt

async def perform_llm_comparison(prompt: str) -> bool:
    """Single LLM comparison, uses batch function internally"""
    results = await perform_llm_comparison_batch([prompt])
    return results[0]

async def perform_llm_comparison_batch(prompts: List[str]) -> List[bool]:
    """Perform multiple LLM comparisons in parallel"""
    async def single_comparison(prompt: str) -> bool:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: client.chat.completions.create(
                    model=settings.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Output only the JSON response."}
                    ],
                    temperature=0.1,  # Lower temperature for more consistent outputs
                    max_tokens=settings.MAX_TOKENS,
                )
            )
            
            # Add debug logging
            logger.debug(f"Raw API Response: {response.choices[0].message.content}")
            
            # Clean the response string to ensure it's valid JSON
            response_text = response.choices[0].message.content.strip()
            # Remove any potential markdown code block markers
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(response_text)
            return result["winner"] == "A"
        except Exception as e:
            logger.error(f"LLM comparison error: {str(e)}")
            raise

    # Run all comparisons in parallel
    results = await asyncio.gather(
        *(single_comparison(prompt) for prompt in prompts),
        return_exceptions=True
    )
    
    # Handle any exceptions
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Batch comparison error: {str(result)}")
            raise result
            
    return results

# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    candidate_a = {
        "ParsedResume": "{'data': {'skills': ['Python', 'Git/Github'], 'education': [{'degree': 'Bachelor of Technology', 'major': 'Mechanical Engineering'}]}}",
        "ParsedTranscript": "Candidate A interview transcript..."
    }
    candidate_b = {
        "ParsedResume": "{'data': {'skills': ['C', 'C++', 'Linux Kernel'], 'education': [{'degree': 'BE', 'major': 'Electronics and Communication'}]}}",
        "ParsedTranscript": "Candidate B interview transcript..."
    }
    
    role_query = "Software Engineer"
    prompt = create_comparison_prompt(role_query, candidate_a, candidate_b)
    
    # Run the comparison
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(perform_llm_comparison(prompt))
    print(f"Candidate A is better: {result}")