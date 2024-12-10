from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Candidate:
    """Represents a candidate with their resume and interview data."""
    resume_id: str
    parsed_resume: dict
    interview_transcript: str = ""
    current_rank: float = 0.0

    def get_resume_text(self) -> str:
        """Formats resume data into readable text."""
        resume_text = []
        
        if 'data' in self.parsed_resume:
            data = self.parsed_resume['data']
            
            # Education
            if 'education' in data:
                edu = data['education']
                edu_text = (f"Education: {edu.get('degree', '')} "
                          f"from {edu.get('school', '')} "
                          f"GPA: {edu.get('GPA', '')} "
                          f"Major: {edu.get('major', '')} "
                          f"Year: {edu.get('endYear', '')}")
                resume_text.append(edu_text)

            # Certifications
            if 'certifications' in data:
                certs = data['certifications']
                if isinstance(certs, list):
                    cert_text = "Certifications: " + ", ".join(certs)
                    resume_text.append(cert_text)

            # Awards
            if 'awards' in data:
                awards = data['awards']
                if isinstance(awards, list):
                    awards_text = "Awards: " + ", ".join(awards)
                    resume_text.append(awards_text)

        return " ".join(resume_text) 