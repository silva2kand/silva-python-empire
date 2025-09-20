from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio

app = FastAPI(title="AI Sherpa - Code Comprehension Service (Safe Mode)", version="1.0.0")

class ComprehensionRequest(BaseModel):
    code: str
    language: str
    question: Optional[str] = ""
    context: Optional[str] = ""

class ComprehensionOutput(BaseModel):
    summary: str
    key_functions: List[Dict[str, str]]
    potential_issues: List[str]
    dependencies: List[str]
    llm_analysis: str
    confidence_score: float

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    confidence: float

@app.get("/")
async def root():
    return {
        "service": "AI Sherpa Code Comprehension Service (Safe Mode)",
        "status": "running",
        "version": "1.0.0",
        "gpt4all_available": False
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpt4all_available": False
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant using safe mock responses."""
    try:
        message = request.message.lower()
        
        # Generate contextual responses based on message content
        if any(word in message for word in ['hello', 'hi', 'hey']):
            response = "Hello! I'm your AI coding assistant. I can help you with code analysis, debugging, and programming questions. What would you like to work on?"
        elif any(word in message for word in ['error', 'bug', 'problem', 'issue']):
            response = "I'd be happy to help you debug that issue! Could you share the specific error message or code that's causing problems? I can help analyze it and suggest solutions."
        elif any(word in message for word in ['code', 'function', 'class', 'method']):
            response = "I can help you with code analysis and improvements! Please share your code and I'll provide insights on structure, potential issues, and best practices."
        elif any(word in message for word in ['python', 'javascript', 'java', 'c++', 'c#']):
            response = f"Great! I can assist with {message.split()[0]} development. What specific aspect would you like help with - syntax, best practices, debugging, or something else?"
        elif 'help' in message:
            response = "I'm here to help! I can assist with:\n\n• Code analysis and review\n• Debugging and error resolution\n• Best practices and optimization\n• Language-specific questions\n• Architecture and design patterns\n\nWhat would you like to explore?"
        else:
            response = "I understand you're asking about: \"" + request.message + "\". While I'm currently running in safe mode, I can still help with general coding questions and provide guidance. Could you be more specific about what you'd like assistance with?"
        
        return ChatResponse(
            response=response,
            confidence=0.8
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/comprehend", response_model=ComprehensionOutput)
async def comprehend_code(request: ComprehensionRequest):
    """Analyze code using safe mock analysis."""
    try:
        # Basic analysis based on code content
        lines = request.code.split('\n')
        functions = [line.strip() for line in lines if 'def ' in line or 'function ' in line]
        
        return ComprehensionOutput(
            summary=f"Code analysis for {request.language} code with {len(lines)} lines.",
            key_functions=[{"name": func, "description": "Function detected in code"} for func in functions[:5]],
            potential_issues=["Consider adding error handling", "Review variable naming conventions"],
            dependencies=["Standard library imports detected"],
            llm_analysis="This code appears to be well-structured. Consider adding documentation and error handling for production use.",
            confidence_score=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)