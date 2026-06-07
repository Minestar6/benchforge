Question:
{{ question }}

Reference Answer:
{{ reference_answer }}

Evidence / Citations:
{{ evidence }}

Model Answer:
{{ model_answer }}

Evaluation dimensions:
{% for metric in metrics %}
- {{ metric.name }}: {{ metric.description }}
{% endfor %}

Return JSON only:
{
  "scores": {
    {% for metric in metrics %}
    "{{ metric.name }}": <number from 0 to 1>{% if not loop.last %},{% endif %}
    {% endfor %}
  },
  "reason": "<brief explanation>"
}
