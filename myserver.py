from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.evaluation import load_evaluator
from langchain_groq import ChatGroq
from typing import List

app = FastAPI()

# CORS setup to allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allow cookies to be sent
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

criteria_completeness = {
    "completeness": """
    Score 1: The answer is incomplete, missing critical information.
    Score 3: The answer includes some key points but lacks significant details.
    Score 5: The answer covers most points but misses minor details.
    Score 7: The answer is complete but could include additional optional details.
    Score 10: The answer is completely thorough and includes all relevant details."""
}

criteria_clarity = {
    "clarity": """
    Score 1: The answer is unclear and difficult to understand.
    Score 3: The answer has partial clarity but is confusing in parts.
    Score 5: The answer is moderately clear but has room for improvement.
    Score 7: The answer is mostly clear and easy to follow.
    Score 10: The answer is completely clear and unambiguous."""
}

criteria_technical_accuracy = {
    "technical_accuracy": """
    Score 1: The answer contains significant technical inaccuracies.
    Score 3: The answer has minor technical inaccuracies or missing details.
    Score 5: The answer is mostly accurate but could use refinement.
    Score 7: The answer is accurate with minor errors or omissions.
    Score 10: The answer is technically perfect and aligns completely with the reference."""
}

evaluator_completeness = load_evaluator(
    "labeled_score_string",
    criteria=criteria_completeness,
    llm=ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768"),
)

evaluator_clarity = load_evaluator(
    "labeled_score_string",
    criteria=criteria_clarity,
    llm=ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768"),
)

evaluator_technical_accuracy = load_evaluator(
    "labeled_score_string",
    criteria=criteria_technical_accuracy,
    llm=ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768"),
)

class FeedbackRequest(BaseModel):
    question: str
    user_answer: str
    reference_answer: str

class FeedbackResponse(BaseModel):
    completeness: float
    clarity: float
    technical_accuracy: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Single Feedback Evaluation API..."}

@app.post("/evaluate-feedback", response_model=FeedbackResponse)
async def evaluate_feedback(request: FeedbackRequest):
    
    eval_result_completeness = evaluator_completeness.evaluate_strings(
        prediction=request.user_answer,
        reference=request.reference_answer,
        input=request.question
    )
    completeness_score = eval_result_completeness["score"]

    
    eval_result_clarity = evaluator_clarity.evaluate_strings(
        prediction=request.user_answer,
        reference=request.reference_answer,
        input=request.question
    )
    clarity_score = eval_result_clarity["score"]

    
    eval_result_accuracy = evaluator_technical_accuracy.evaluate_strings(
        prediction=request.user_answer,
        reference=request.reference_answer,
        input=request.question
    )
    accuracy_score = eval_result_accuracy["score"]

    return FeedbackResponse(
        completeness=completeness_score,
        clarity=clarity_score,
        technical_accuracy=accuracy_score
    )
