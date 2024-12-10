import pandas as pd
import json

# Create sample data
test_data = [
    {
        'UserID': f'USER_{i}',
        'ParsedResume': json.dumps({
            'data': {
                'education': {
                    'degree': f'Degree_{i}',
                    'GPA': f'{3.0 + (i/10):.1f}',
                    'school': f'University_{i}'
                },
                'certifications': [f'Cert_{i}'],
                'awards': [f'Award_{i}']
            }
        }),
        'interview_transcript': f'Sample interview transcript {i}'
    }
    for i in range(10)  # Create 10 sample candidates
]

# Create DataFrame and save to CSV
df = pd.DataFrame(test_data)
df.to_csv('test_candidates.csv', index=False) 