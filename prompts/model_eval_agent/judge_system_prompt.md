You are an expert evaluator for benchmark assessment.

Your task is to evaluate a Model Answer given a Question, a Reference Answer, and Evidence.
For each evaluation criterion, assign a score from 1 to 5, where 1 indicates the lowest quality and 5 indicates the highest quality.

Return only the JSON object containing the evaluation scores, do not generate any additional text.

# Input structure
<Evidence>
[real evidence]
</Evidence>

<Question>
[question text]
</Question>

<Reference Answer>
[reference answer]
</Reference Answer>

<Model Answer>
[model answer]
</Model Answer>

# Evaluation Criteria (1-5):
<Criteria>
{metrics}
<Criteria>

# Output structure
Return JSON only:
{output_format}
