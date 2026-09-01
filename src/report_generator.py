import ollama


def generate_report(
    prediction: str,
    confidence: float,
    model_name: str
) -> str:

    prompt = f"""
You are an AI assistant helping generate a medical imaging
research report.

The following information comes from a deep-learning model:

Model: {model_name}
Prediction: {prediction}
Confidence: {confidence:.2f}%

Generate a concise research-oriented report based ONLY on
the structured model output provided above.

Include:

1. Model
2. Prediction
3. Confidence
4. Interpretation
5. Important disclaimer

Do not claim that the prediction is a definitive diagnosis.
Do not recommend treatment.
Do not invent additional medical findings.

Clearly state that this is an AI-generated research result
and requires review by a qualified healthcare professional.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]