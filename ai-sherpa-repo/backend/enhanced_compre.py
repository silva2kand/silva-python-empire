from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import ast
import re
import json
import os

# Lightweight model availability check
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False

app = FastAPI(title="AI Sherpa - Enhanced Code Analysis Service", version="2.0.0")

# Lightweight model configuration
MODEL_CONFIG = {
    "gpt4all": {
        "name": "mistral-7b-openorca.Q4_0.gguf",
        "available": GPT4ALL_AVAILABLE,
        "type": "local"
    },
    "mock_advanced": {
        "name": "Advanced Mock AI",
        "available": True,
        "type": "mock"
    },
    "mock_code": {
        "name": "Code Analysis AI",
        "available": True,
        "type": "mock"
    }
}

class AIModelManager:
    """Lightweight AI model manager"""
    
    def __init__(self):
        self.current_model = "mock_advanced"
        self.gpt4all_model = None
        
        # Try to initialize GPT4All if available
        if GPT4ALL_AVAILABLE and os.environ.get('DISABLE_GPT4ALL') != '1':
            try:
                self.gpt4all_model = GPT4All(MODEL_CONFIG["gpt4all"]["name"])
                print("GPT4All model loaded successfully")
            except Exception as e:
                print(f"GPT4All not available: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        available = []
        for model_id, config in MODEL_CONFIG.items():
            if config["available"]:
                available.append({
                    "id": model_id,
                    "name": config["name"],
                    "type": config["type"],
                    "active": model_id == self.current_model
                })
        return available
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model"""
        if model_id in MODEL_CONFIG and MODEL_CONFIG[model_id]["available"]:
            self.current_model = model_id
            return True
        return False
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using current model"""
        try:
            if self.current_model == "gpt4all" and self.gpt4all_model:
                return self.gpt4all_model.generate(prompt, max_tokens=max_tokens).strip()
            else:
                return self._generate_smart_mock_response(prompt)
        except Exception as e:
            print(f"Error with {self.current_model}: {e}")
            return self._generate_smart_mock_response(prompt)
    
    def _generate_smart_mock_response(self, prompt: str) -> str:
        """Generate intelligent mock responses based on model type"""
        prompt_lower = prompt.lower()
        
        if self.current_model == "mock_code":
            if "analyze" in prompt_lower and "code" in prompt_lower:
                return "This code looks well-structured! I can see you're using good practices. Here are some suggestions: 1) Consider adding more comments for complex logic, 2) You might want to add error handling for edge cases, 3) The function structure is clean and readable."
            elif "bug" in prompt_lower or "error" in prompt_lower:
                return "I can help debug this! Common issues to check: 1) Variable scope and naming, 2) Indentation errors, 3) Missing imports or dependencies, 4) Logic flow issues. Please share the specific error message for more targeted help."
            elif "optimize" in prompt_lower:
                return "For optimization, consider: 1) Using list comprehensions where appropriate, 2) Caching expensive operations, 3) Reducing nested loops, 4) Using built-in functions when possible. The current code structure provides a good foundation for improvements."
        
        # General responses
        if "code" in prompt_lower and "analysis" in prompt_lower:
            return "I can analyze your code for issues, suggest improvements, and explain functionality. Share your code and I'll provide detailed feedback on structure, performance, and best practices!"
        elif "model" in prompt_lower and ("switch" in prompt_lower or "change" in prompt_lower):
            return f"You can switch between models: {', '.join([m['name'] for m in self.get_available_models()])}. Each model has different strengths - try the Code Analysis AI for detailed code reviews!"
        elif "hello" in prompt_lower or "hi" in prompt_lower:
            return f"Hello! I'm your AI assistant running on {MODEL_CONFIG[self.current_model]['name']}. I can help with code analysis, debugging, optimization, and general programming questions. What would you like to work on?"
        elif "feature" in prompt_lower:
            return "Available features: 🔍 Code Analysis, 🐛 Bug Detection, ⚡ Performance Optimization, 🔄 Model Switching, 💡 Best Practice Suggestions, and 📊 Code Metrics. How can I help?"
        else:
            return f"I understand you're asking about: '{prompt[:80]}...' I'm running on {MODEL_CONFIG[self.current_model]['name']} and ready to help with your coding tasks. Could you provide more specific details?"

class CodeAnalyzer:
    """Advanced code analysis capabilities"""
    
    @staticmethod
    def analyze_python_code(code: str) -> Dict[str, Any]:
        """Comprehensive Python code analysis"""
        issues = []
        metrics = {}
        suggestions = []
        security_issues = []
        
        try:
            tree = ast.parse(code)
            
            # Extract code elements
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            # Calculate metrics
            lines = code.split('\n')
            metrics = {
                "lines_of_code": len(lines),
                "functions": len(functions),
                "classes": len(classes),
                "imports": len(imports),
                "blank_lines": len([line for line in lines if not line.strip()]),
                "comment_lines": len([line for line in lines if line.strip().startswith('#')])
            }
            
            # Code quality checks
            if len(functions) == 0 and len(classes) == 0 and len(lines) > 10:
                issues.append("Consider organizing code into functions or classes for better structure")
            
            # Line length check
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
            if long_lines:
                issues.append(f"Lines exceed 100 characters: {long_lines[:3]}{'...' if len(long_lines) > 3 else ''}")
            
            # Documentation check
            functions_without_docs = [f.name for f in functions if not ast.get_docstring(f)]
            if functions_without_docs:
                suggestions.append(f"Add docstrings to functions: {functions_without_docs[:3]}")
            
            # Security analysis
            if "eval(" in code or "exec(" in code:
                security_issues.append("Potential security risk: eval() or exec() usage")
            if "os.system" in code:
                security_issues.append("Potential security risk: os.system() usage")
            if "subprocess.call" in code and "shell=True" in code:
                security_issues.append("Potential security risk: subprocess with shell=True")
            
            # Best practices
            if "import *" in code:
                suggestions.append("Avoid wildcard imports (import *) - use specific imports")
            if re.search(r'\bprint\(', code) and len(functions) > 0:
                suggestions.append("Consider using logging instead of print statements")
            
            # Complexity estimation
            complexity_score = min(len(functions) * 3 + len(classes) * 5 + len([n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))]) * 2, 100)
            
        except SyntaxError as e:
            issues.append(f"Syntax Error at line {e.lineno}: {e.msg}")
            complexity_score = 0
        except Exception as e:
            issues.append(f"Analysis Error: {str(e)}")
            complexity_score = 0
        
        return {
            "issues": issues + security_issues,
            "metrics": metrics,
            "suggestions": suggestions,
            "security_issues": security_issues,
            "complexity_score": complexity_score
        }
    
    @staticmethod
    def analyze_javascript_code(code: str) -> Dict[str, Any]:
        """JavaScript code analysis"""
        issues = []
        suggestions = []
        
        # Modern JavaScript checks
        if "var " in code:
            suggestions.append("Use 'let' or 'const' instead of 'var' for better scoping")
        if "==" in code and "===" not in code:
            suggestions.append("Use strict equality (===) instead of loose equality (==)")
        if "console.log" in code:
            suggestions.append("Remove console.log statements before production")
        if "document.write" in code:
            issues.append("Avoid document.write - use modern DOM manipulation")
        
        # Security checks
        if "eval(" in code:
            issues.append("Security risk: eval() usage detected")
        if "innerHTML" in code:
            suggestions.append("Consider using textContent or safer DOM methods instead of innerHTML")
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {"lines_of_code": len(code.split('\n'))}
        }
    
    @staticmethod
    def get_code_language(code: str) -> str:
        """Enhanced language detection"""
        code_lower = code.lower()
        
        # Python indicators
        if any(keyword in code for keyword in ["def ", "import ", "print(", "if __name__", "class "]):
            return "python"
        
        # JavaScript indicators
        if any(keyword in code for keyword in ["function ", "const ", "let ", "=>", "console.log"]):
            return "javascript"
        
        # Other languages
        if "#include" in code or "int main" in code:
            return "c++"
        if "public class" in code or "System.out" in code:
            return "java"
        if "<?php" in code or "echo " in code:
            return "php"
        
        return "unknown"

# Initialize services
ai_manager = AIModelManager()
code_analyzer = CodeAnalyzer()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    model: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    model_used: str
    features_used: Optional[List[str]] = []

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = None
    analysis_type: Optional[str] = "full"

class CodeAnalysisResponse(BaseModel):
    language: str
    issues: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any]
    complexity_score: Optional[int] = None
    ai_analysis: str
    security_issues: Optional[List[str]] = []

class ModelSwitchRequest(BaseModel):
    model_id: str

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "AI Sherpa Enhanced Code Analysis Service",
        "status": "running",
        "version": "2.0.0",
        "features": ["code_analysis", "model_switching", "security_scanning", "performance_tips"],
        "current_model": ai_manager.current_model,
        "available_models": len(ai_manager.get_available_models())
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "current_model": ai_manager.current_model,
        "features_active": ["code_sniffer", "multi_model", "security_scan"]
    }

@app.get("/models")
async def get_available_models():
    return {
        "models": ai_manager.get_available_models(),
        "current": ai_manager.current_model,
        "switching_enabled": True
    }

@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    success = ai_manager.switch_model(request.model_id)
    if success:
        return {
            "success": True,
            "message": f"Switched to {MODEL_CONFIG[request.model_id]['name']}",
            "current_model": ai_manager.current_model,
            "model_type": MODEL_CONFIG[request.model_id]['type']
        }
    else:
        raise HTTPException(status_code=400, detail="Model not available")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        features_used = []
        
        # Model switching
        if request.model and request.model != ai_manager.current_model:
            if ai_manager.switch_model(request.model):
                features_used.append("model_switching")
        
        message = request.message
        
        # Code analysis detection
        if "```" in message or any(keyword in message.lower() for keyword in ["analyze code", "review code", "check code"]):
            features_used.append("code_analysis")
            
            # Extract code blocks
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', message, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
                language = code_analyzer.get_code_language(code)
                
                # Perform analysis
                if language == "python":
                    analysis = code_analyzer.analyze_python_code(code)
                elif language == "javascript":
                    analysis = code_analyzer.analyze_javascript_code(code)
                else:
                    analysis = {"issues": [], "suggestions": ["Language detection: " + language]}
                
                # Generate comprehensive response
                analysis_prompt = f"Analyze this {language} code and provide expert insights:\n\nCode Quality Assessment:\n- Issues: {len(analysis.get('issues', []))}\n- Suggestions: {len(analysis.get('suggestions', []))}\n- Complexity: {analysis.get('complexity_score', 'N/A')}\n\nProvide detailed feedback:"
                
                ai_response = ai_manager.generate_response(analysis_prompt, max_tokens=400)
                
                response = f"## 🔍 Code Analysis Results\n\n{ai_response}\n\n### 📊 Technical Details:\n"
                
                if analysis.get('issues'):
                    response += f"\n🔴 **Issues Found ({len(analysis['issues'])}):**\n"
                    for issue in analysis['issues'][:3]:
                        response += f"• {issue}\n"
                
                if analysis.get('suggestions'):
                    response += f"\n💡 **Suggestions ({len(analysis['suggestions'])}):**\n"
                    for suggestion in analysis['suggestions'][:3]:
                        response += f"• {suggestion}\n"
                
                if analysis.get('security_issues'):
                    response += f"\n🛡️ **Security Concerns:**\n"
                    for security in analysis['security_issues']:
                        response += f"• {security}\n"
                    features_used.append("security_scan")
                
                return ChatResponse(
                    response=response,
                    confidence=0.95,
                    model_used=ai_manager.current_model,
                    features_used=features_used
                )
        
        # Regular chat with context
        context = ""
        if request.conversation_history:
            context = "Context from previous messages:\n"
            for msg in request.conversation_history[-2:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:100]
                context += f"{role}: {content}\n"
        
        prompt = f"""{context}\nUser: {message}\n\nAs an AI coding assistant with advanced code analysis capabilities, provide helpful guidance. Current model: {MODEL_CONFIG[ai_manager.current_model]['name']}\n\nAssistant:"""
        
        response = ai_manager.generate_response(prompt, max_tokens=500)
        
        return ChatResponse(
            response=response,
            confidence=0.85,
            model_used=ai_manager.current_model,
            features_used=features_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    try:
        code = request.code
        language = request.language or code_analyzer.get_code_language(code)
        
        # Perform static analysis
        if language == "python":
            analysis = code_analyzer.analyze_python_code(code)
        elif language == "javascript":
            analysis = code_analyzer.analyze_javascript_code(code)
        else:
            analysis = {
                "issues": [],
                "suggestions": [f"Basic analysis for {language}"],
                "metrics": {"lines_of_code": len(code.split('\n'))}
            }
        
        # AI insights
        ai_prompt = f"Expert code review for {language}:\n\nMetrics: {analysis.get('metrics', {})}\nIssues: {len(analysis.get('issues', []))}\nComplexity: {analysis.get('complexity_score', 'N/A')}\n\nProvide professional assessment:"
        
        ai_analysis = ai_manager.generate_response(ai_prompt, max_tokens=300)
        
        return CodeAnalysisResponse(
            language=language,
            issues=analysis.get("issues", []),
            suggestions=analysis.get("suggestions", []),
            metrics=analysis.get("metrics", {}),
            complexity_score=analysis.get("complexity_score"),
            ai_analysis=ai_analysis,
            security_issues=analysis.get("security_issues", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting AI Sherpa Enhanced Service...")
    print(f"📊 Available models: {len(ai_manager.get_available_models())}")
    print(f"🔧 Current model: {ai_manager.current_model}")
    print("✅ Code analysis features ready!")
    uvicorn.run(app, host="0.0.0.0", port=8001)