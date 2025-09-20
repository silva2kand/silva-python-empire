from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
from shared.gpt4all_service import gpt4all_service
from agents.chat_integration import get_chat_integration, process_chat_with_agents

app = FastAPI(title="AI Sherpa - Code Comprehension Service", version="1.0.0")

# Initialize chat integration on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the multi-agent system on startup"""
    try:
        chat_integration = get_chat_integration()
        await chat_integration.initialize()
        print("Multi-agent chat integration initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize multi-agent system: {str(e)}")
        print("Falling back to basic chat functionality")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup multi-agent system on shutdown"""
    try:
        chat_integration = get_chat_integration()
        await chat_integration.shutdown()
        print("Multi-agent system shutdown complete")
    except Exception as e:
        print(f"Error during multi-agent system shutdown: {str(e)}")

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

@app.get("/")
async def root():
    return {
        "service": "AI Sherpa Code Comprehension Service",
        "status": "running",
        "version": "1.0.0",
        "gpt4all_available": gpt4all_service.is_available()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpt4all_available": gpt4all_service.is_available()
    }

@app.get("/agents/status")
async def get_agents_status():
    """
    Get the status of the multi-agent system.
    """
    try:
        chat_integration = get_chat_integration()
        status = await chat_integration.get_system_status()
        return {
            "multi_agent_system": status,
            "timestamp": status.get("timestamp")
        }
    except Exception as e:
        return {
            "multi_agent_system": {
                "status": "error",
                "error": str(e)
            },
            "timestamp": None
        }

@app.post("/comprehend", response_model=ComprehensionOutput)
async def comprehend_code(request: ComprehensionRequest):
    """
    Analyze and comprehend the provided code.
    
    This endpoint performs comprehensive code analysis including:
    - Code summary and purpose identification
    - Key function extraction and analysis
    - Potential issue detection
    - Dependency analysis
    - AI-powered insights using GPT4All
    """
    try:
        # Basic static analysis
        analysis_result = await analyze_code_structure(request.code, request.language)
        
        # Generate AI analysis using GPT4All
        prompt = gpt4all_service.create_code_comprehension_prompt(
            request.code, 
            request.language, 
            request.question
        )
        
        if request.context:
            prompt += f"\n\nAdditional Context:\n{request.context}"
        
        llm_analysis = gpt4all_service.generate_response(prompt, max_tokens=1024)
        
        # Calculate confidence score based on code complexity and analysis quality
        confidence_score = calculate_confidence_score(
            request.code, 
            analysis_result, 
            gpt4all_service.is_available()
        )
        
        return ComprehensionOutput(
            summary=analysis_result["summary"],
            key_functions=analysis_result["functions"],
            potential_issues=analysis_result["issues"],
            dependencies=analysis_result["dependencies"],
            llm_analysis=llm_analysis,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code comprehension failed: {str(e)}")

async def analyze_code_structure(code: str, language: str) -> Dict[str, Any]:
    """
    Perform basic static analysis of code structure.
    """
    lines = code.split('\n')
    
    # Basic analysis based on language
    if language.lower() in ['python', 'py']:
        return analyze_python_code(lines)
    elif language.lower() in ['javascript', 'js', 'typescript', 'ts']:
        return analyze_javascript_code(lines)
    elif language.lower() in ['java']:
        return analyze_java_code(lines)
    elif language.lower() in ['c', 'cpp', 'c++']:
        return analyze_c_code(lines)
    else:
        return analyze_generic_code(lines)

def analyze_python_code(lines: List[str]) -> Dict[str, Any]:
    """Analyze Python code structure."""
    functions = []
    classes = []
    imports = []
    issues = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Function detection
        if stripped.startswith('def '):
            func_name = stripped.split('(')[0].replace('def ', '')
            functions.append({
                "name": func_name,
                "line": i + 1,
                "type": "function"
            })
        
        # Class detection
        elif stripped.startswith('class '):
            class_name = stripped.split('(')[0].split(':')[0].replace('class ', '')
            classes.append({
                "name": class_name,
                "line": i + 1,
                "type": "class"
            })
        
        # Import detection
        elif stripped.startswith(('import ', 'from ')):
            imports.append(stripped)
        
        # Basic issue detection
        if 'TODO' in stripped or 'FIXME' in stripped:
            issues.append(f"Line {i + 1}: {stripped}")
        
        if 'print(' in stripped and 'debug' in stripped.lower():
            issues.append(f"Line {i + 1}: Debug print statement found")
    
    # Generate summary
    summary = f"Python code with {len(functions)} functions, {len(classes)} classes"
    if imports:
        summary += f", {len(imports)} imports"
    
    return {
        "summary": summary,
        "functions": functions + classes,
        "dependencies": imports,
        "issues": issues
    }

def analyze_javascript_code(lines: List[str]) -> Dict[str, Any]:
    """Analyze JavaScript/TypeScript code structure."""
    functions = []
    imports = []
    issues = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Function detection
        if 'function ' in stripped or '=>' in stripped:
            if 'function ' in stripped:
                func_name = stripped.split('function ')[1].split('(')[0].strip()
            else:
                func_name = stripped.split('=>')[0].split('=')[0].strip()
            
            functions.append({
                "name": func_name,
                "line": i + 1,
                "type": "function"
            })
        
        # Import detection
        elif stripped.startswith(('import ', 'const ', 'require(')):
            imports.append(stripped)
        
        # Basic issue detection
        if 'console.log' in stripped:
            issues.append(f"Line {i + 1}: Console log statement found")
        
        if 'TODO' in stripped or 'FIXME' in stripped:
            issues.append(f"Line {i + 1}: {stripped}")
    
    summary = f"JavaScript code with {len(functions)} functions"
    if imports:
        summary += f", {len(imports)} imports"
    
    return {
        "summary": summary,
        "functions": functions,
        "dependencies": imports,
        "issues": issues
    }

def analyze_java_code(lines: List[str]) -> Dict[str, Any]:
    """Analyze Java code structure."""
    functions = []
    classes = []
    imports = []
    issues = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Method detection
        if ('public ' in stripped or 'private ' in stripped or 'protected ' in stripped) and '(' in stripped:
            method_name = stripped.split('(')[0].split()[-1]
            functions.append({
                "name": method_name,
                "line": i + 1,
                "type": "method"
            })
        
        # Class detection
        elif stripped.startswith('class ') or stripped.startswith('public class '):
            class_name = stripped.split('class ')[1].split()[0]
            classes.append({
                "name": class_name,
                "line": i + 1,
                "type": "class"
            })
        
        # Import detection
        elif stripped.startswith('import '):
            imports.append(stripped)
        
        # Basic issue detection
        if 'System.out.println' in stripped:
            issues.append(f"Line {i + 1}: Debug print statement found")
    
    summary = f"Java code with {len(classes)} classes, {len(functions)} methods"
    
    return {
        "summary": summary,
        "functions": functions + classes,
        "dependencies": imports,
        "issues": issues
    }

def analyze_c_code(lines: List[str]) -> Dict[str, Any]:
    """Analyze C/C++ code structure."""
    functions = []
    includes = []
    issues = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Function detection (basic)
        if '(' in stripped and ')' in stripped and '{' in stripped and not stripped.startswith('//'):
            if any(keyword in stripped for keyword in ['int ', 'void ', 'char ', 'float ', 'double ']):
                func_name = stripped.split('(')[0].split()[-1]
                functions.append({
                    "name": func_name,
                    "line": i + 1,
                    "type": "function"
                })
        
        # Include detection
        elif stripped.startswith('#include'):
            includes.append(stripped)
        
        # Basic issue detection
        if 'printf(' in stripped and 'debug' in stripped.lower():
            issues.append(f"Line {i + 1}: Debug print statement found")
    
    summary = f"C/C++ code with {len(functions)} functions"
    
    return {
        "summary": summary,
        "functions": functions,
        "dependencies": includes,
        "issues": issues
    }

def analyze_generic_code(lines: List[str]) -> Dict[str, Any]:
    """Generic code analysis for unknown languages."""
    issues = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'TODO' in stripped or 'FIXME' in stripped:
            issues.append(f"Line {i + 1}: {stripped}")
    
    return {
        "summary": f"Generic code analysis - {len(lines)} lines",
        "functions": [],
        "dependencies": [],
        "issues": issues
    }

def calculate_confidence_score(code: str, analysis: Dict[str, Any], gpt4all_available: bool) -> float:
    """Calculate confidence score for the analysis."""
    base_score = 0.6
    
    # Increase confidence based on code length and complexity
    lines = len(code.split('\n'))
    if lines > 10:
        base_score += 0.1
    if lines > 50:
        base_score += 0.1
    
    # Increase confidence if functions/classes detected
    if analysis["functions"]:
        base_score += 0.1
    
    # Increase confidence if GPT4All is available
    if gpt4all_available:
        base_score += 0.1
    
    return min(base_score, 1.0)

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    confidence: float

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Chat with AI assistant for coding help and general questions.
    Enhanced with multi-agent system for better responses.
    """
    try:
        # Try to use multi-agent system first
        try:
            agent_result = await process_chat_with_agents(
                request.message, 
                request.conversation_history
            )
            
            if agent_result and agent_result.get("response"):
                return ChatResponse(
                    response=agent_result["response"],
                    confidence=agent_result.get("confidence", 0.8)
                )
        except Exception as agent_error:
            print(f"Multi-agent processing failed: {str(agent_error)}")
            print("Falling back to basic GPT4All processing")
        
        # Fallback to original GPT4All processing
        # Build context from conversation history
        context = ""
        if request.conversation_history:
            context = "Previous conversation:\n"
            for msg in request.conversation_history[-5:]:  # Keep last 5 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                context += f"{role.capitalize()}: {content}\n"
            context += "\nCurrent question:\n"
        
        # Create a comprehensive prompt for the AI
        prompt = f"""{context}User: {request.message}

As an AI coding assistant, please provide a helpful, accurate, and detailed response. 
If this is about code, provide examples when appropriate. 
If you're not sure about something, please say so.

A:"""
        
        # Get response from GPT4All
        response = gpt4all_service.generate_response(prompt, max_tokens=500)
        
        if not response:
            response = "I apologize, but I couldn't generate a response at the moment. Please try again."
        
        # Calculate confidence based on response quality
        confidence = 0.8 if len(response) > 50 else 0.6
        
        return ChatResponse(
            response=response,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)