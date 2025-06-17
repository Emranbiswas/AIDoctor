def generate_response(model, disease):
    if disease.lower() == "normal":
        prompt = (
            "A chest X-ray analysis shows the patient is normal. Write a friendly, positive message encouraging the person. "
            "Include tips for maintaining good respiratory health."
        )
    elif disease.lower() == "covid":
        prompt = (
            "Based on recent peer-reviewed medical research, what are the WHO-recommended steps for treating and isolating "
            "a Covid-19 patient diagnosed through chest X-ray? Mention medications, home care, and tests if applicable."
        )
    elif disease.lower() == "pneumonia":
        prompt = (
            "Summarize research-based treatment and care guidelines for patients diagnosed with pneumonia from a chest X-ray. "
            "Include medications, rest, and any supportive therapy mentioned in clinical studies."
        )
    elif disease.lower() == "tuberculosis":
        prompt = (
            "Provide a research-based overview of tuberculosis treatment and therapy as diagnosed from chest X-rays. "
            "Include WHO guidelines, medications (e.g. isoniazid, rifampin), duration, and recommended machines/tests from literature."
        )
    else:
        prompt = "If a chest X-ray report is unclear, what should the patient do next based on clinical guidelines?"

    response = model.generate_content(prompt)
    return response.text.strip()
