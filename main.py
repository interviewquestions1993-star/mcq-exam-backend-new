"""
FastAPI backend for Hugging Face Inference API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from hf_inference import hf_client
from firebase_client import save_ai_response
from config import API_HOST, API_PORT, HF_MODEL

# Initialize FastAPI app
app = FastAPI(
    title="Hugging Face Inference Backend",
    description="Backend for text generation using Hugging Face Inference API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class TextGenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain what a neural network is in two sentences.",
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }


class TextGenerationResponse(BaseModel):
    """Response model for text generation"""
    prompt: str
    generated_text: str
    status: str = "success"


class SummarizationRequest(BaseModel):
    """Request model for text summarization"""
    text: str
    max_length: Optional[int] = 50
    min_length: Optional[int] = 20
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hugging Face is a company that develops tools for building machine learning applications...",
                "max_length": 50,
                "min_length": 20
            }
        }


class SummarizationResponse(BaseModel):
    """Response model for text summarization"""
    original_text: str
    summary: str
    status: str = "success"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


class MCQQuestion(BaseModel):
    """Single MCQ question"""
    id: int
    question: str
    options: List[str]  # A) Option 1, B) Option 2, etc.
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    difficulty: str  # "easy", "medium", "hard"


class MCQGenerationRequest(BaseModel):
    """Request model for MCQ generation"""
    topic: str
    num_questions: Optional[int] = 5
    difficulty: Optional[str] = None  # "easy", "medium", "hard", or None for mixed
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Angular",
                "num_questions": 5,
                "difficulty": None
            }
        }


class MCQGenerationResponse(BaseModel):
    """Response model for MCQ generation"""
    topic: str
    num_questions: int
    questions: List[MCQQuestion]
    status: str = "success"


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Hugging Face Inference Backend is running"
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="Hugging Face Inference Backend is running"
    )


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest) -> TextGenerationResponse:
    """
    Generate text using Hugging Face Inference API
    
    Args:
        request: TextGenerationRequest with prompt and parameters
    
    Returns:
        TextGenerationResponse with generated text
    
    Raises:
        HTTPException: If generation fails
    """
    try:
        generated_text = hf_client.text_generation(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        save_ai_response("ai_responses", {
            "endpoint": "/generate",
            "prompt": request.prompt,
            "model": HF_MODEL,
            "request": request.dict(),
            "response": {"generated_text": generated_text},
            "status": "success"
        })
        
        return TextGenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/chat", response_model=TextGenerationResponse)
async def chat_endpoint(request: TextGenerationRequest) -> TextGenerationResponse:
    """
    Chat endpoint (alias for generate)
    
    Args:
        request: TextGenerationRequest with prompt
    
    Returns:
        TextGenerationResponse with response
    """
    return await generate_text(request)


@app.post("/api/mcq/generate", response_model=MCQGenerationResponse)
async def generate_mcq(request: MCQGenerationRequest) -> MCQGenerationResponse:
    """
    Generate MCQ questions for a given topic
    
    Args:
        request: MCQGenerationRequest with topic and number of questions
    
    Returns:
        MCQGenerationResponse with generated questions
    
    Raises:
        HTTPException: If generation fails
    """
    try:
        import json
        import re
        
        # Build the prompt for MCQ generation
        difficulty_hint = ""
        if request.difficulty:
            difficulty_hint = f"with {request.difficulty} difficulty. "
        
        prompt = f"""Generate {request.num_questions} multiple choice questions {difficulty_hint}about {request.topic}.

Each time, produce a fresh, unique set of questions. Do not repeat questions from earlier generations or reuse the same wording.

For each question, provide:
1. A clear question
2. Four options labeled A), B), C), D)
3. The correct answer (single letter: A, B, C, or D)
4. A brief explanation
5. Difficulty level (easy/medium/hard)

Format your response as a JSON array like this:
[
  {{
    "id": 1,
    "question": "What is...",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "B",
    "explanation": "Because...",
    "difficulty": "medium"
  }}
]

Generate only valid JSON, no other text."""

        # Generate using AI
        response_text = hf_client.text_generation(
            prompt=prompt,
            max_new_tokens=2000,
            temperature=0.9,
            top_p=0.95
        )
        
        # Parse the JSON response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found in response")
            
            json_str = json_match.group(0)
            questions_data = json.loads(json_str)
            
            # Validate and convert to MCQQuestion objects
            questions = []
            for i, q in enumerate(questions_data[:request.num_questions], 1):
                question = MCQQuestion(
                    id=i,
                    question=q.get("question", ""),
                    options=q.get("options", []),
                    correct_answer=q.get("correct_answer", "A"),
                    explanation=q.get("explanation", ""),
                    difficulty=q.get("difficulty", "medium")
                )
                questions.append(question)
            
            response = MCQGenerationResponse(
                topic=request.topic,
                num_questions=len(questions),
                questions=questions,
                status="success"
            )

            save_ai_response("mcq_responses", {
                "endpoint": "/api/mcq/generate",
                "topic": request.topic,
                "num_questions": request.num_questions,
                "difficulty": request.difficulty,
                "request": request.dict(),
                "response": {
                    "questions": [q.dict() for q in questions]
                },
                "status": "success"
            })

            return response
        
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse MCQ response: {str(e)}"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")


class PersistedMCQResponse(BaseModel):
    """Response model for fetching persisted MCQs"""
    id: str
    topic: str
    num_questions: int
    questions: List[MCQQuestion]
    created_at: str
    status: str


@app.get("/api/mcq/persisted", response_model=List[PersistedMCQResponse])
async def get_persisted_mcqs() -> List[PersistedMCQResponse]:
    """
    Fetch all persisted MCQ responses from Firestore
    
    Returns:
        List of persisted MCQ records
    
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        from firebase_client import get_firestore_client
        client = get_firestore_client()
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="Firestore not enabled or unavailable"
            )
        
        docs = client.collection("mcq_responses").stream()
        
        results = []
        for doc in docs:
            data = doc.to_dict()
            # Ensure questions exist and are properly formatted
            questions_data = data.get("response", {}).get("questions", [])
            questions = [MCQQuestion(**q) if isinstance(q, dict) else q for q in questions_data]
            
            result = PersistedMCQResponse(
                id=doc.id,
                topic=data.get("topic", ""),
                num_questions=data.get("num_questions", 0),
                questions=questions,
                created_at=data.get("created_at", ""),
                status=data.get("status", "success")
            )
            results.append(result)
        
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch persisted MCQs: {str(e)}"
        )


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
