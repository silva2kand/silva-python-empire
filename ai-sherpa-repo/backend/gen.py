from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from shared.gpt4all_service import gpt4all_service

app = FastAPI(title="AI Sherpa - Code Generation Service", version="1.0.0")

class GenerationRequest(BaseModel):
    description: str
    language: str
    requirements: Optional[str] = ""
    context: Optional[str] = ""
    style: Optional[str] = "clean"  # clean, minimal, verbose
    include_tests: Optional[bool] = False

class GenerationOutput(BaseModel):
    generated_code: str
    explanation: str
    suggestions: List[str]
    test_code: Optional[str] = None
    confidence_score: float
    language: str

@app.get("/")
async def root():
    return {
        "service": "AI Sherpa Code Generation Service",
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

@app.post("/generate", response_model=GenerationOutput)
async def generate_code(request: GenerationRequest):
    """
    Generate code based on the provided description and requirements.
    
    This endpoint creates code using AI assistance with:
    - Natural language description processing
    - Language-specific code generation
    - Best practices implementation
    - Optional test code generation
    - Code quality suggestions
    """
    try:
        # Create generation prompt
        prompt = gpt4all_service.create_code_generation_prompt(
            request.description,
            request.language,
            request.requirements
        )
        
        if request.context:
            prompt += f"\n\nExisting Context:\n{request.context}"
        
        if request.style:
            prompt += f"\n\nCode Style: {request.style}"
        
        # Generate code using GPT4All
        generated_response = gpt4all_service.generate_response(prompt, max_tokens=1024)
        
        # Parse the generated response
        parsed_result = parse_generation_response(generated_response, request.language)
        
        # Generate test code if requested
        test_code = None
        if request.include_tests:
            test_code = await generate_test_code(
                parsed_result["code"], 
                request.language, 
                request.description
            )
        
        # Calculate confidence score
        confidence_score = calculate_generation_confidence(
            request.description,
            parsed_result["code"],
            request.language,
            gpt4all_service.is_available()
        )
        
        # Generate suggestions
        suggestions = generate_code_suggestions(
            parsed_result["code"],
            request.language,
            request.style
        )
        
        return GenerationOutput(
            generated_code=parsed_result["code"],
            explanation=parsed_result["explanation"],
            suggestions=suggestions,
            test_code=test_code,
            confidence_score=confidence_score,
            language=request.language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

def parse_generation_response(response: str, language: str) -> Dict[str, str]:
    """
    Parse the AI-generated response to extract code and explanation.
    """
    # Look for code blocks
    code_start = response.find("```")
    if code_start != -1:
        # Find the end of the first code block
        code_start = response.find("\n", code_start) + 1
        code_end = response.find("```", code_start)
        
        if code_end != -1:
            code = response[code_start:code_end].strip()
            # Get explanation (text after the code block)
            explanation_start = code_end + 3
            explanation = response[explanation_start:].strip()
            
            # Clean up explanation
            if explanation.startswith("*Note:"):
                explanation = explanation.split("\n\n")[0] if "\n\n" in explanation else explanation
            
            return {
                "code": code,
                "explanation": explanation if explanation else "Generated code based on the provided description."
            }
    
    # Fallback: treat entire response as explanation with basic code template
    template_code = generate_code_template(language)
    return {
        "code": template_code,
        "explanation": response
    }

def generate_code_template(language: str) -> str:
    """
    Generate a basic code template for the specified language.
    """
    templates = {
        "python": '''def main():
    """Main function - implement your logic here."""
    pass

if __name__ == "__main__":
    main()''',
        
        "javascript": '''function main() {
    // Implement your logic here
    console.log("Hello, World!");
}

// Call the main function
main();''',
        
        "java": '''public class Main {
    public static void main(String[] args) {
        // Implement your logic here
        System.out.println("Hello, World!");
    }
}''',
        
        "cpp": '''#include <iostream>

int main() {
    // Implement your logic here
    std::cout << "Hello, World!" << std::endl;
    return 0;
}''',
        
        "c": '''#include <stdio.h>

int main() {
    // Implement your logic here
    printf("Hello, World!\\n");
    return 0;
}'''
    }
    
    return templates.get(language.lower(), "// Code template not available for this language")

async def generate_test_code(code: str, language: str, description: str) -> str:
    """
    Generate test code for the provided code.
    """
    test_prompt = f"""Generate comprehensive test code for the following {language} code:

Code to test:
```{language}
{code}
```

Description: {description}

Please provide:
1. Unit tests covering main functionality
2. Edge case tests
3. Error handling tests
4. Clear test descriptions

Use appropriate testing framework for {language}.
"""
    
    test_response = gpt4all_service.generate_response(test_prompt, max_tokens=512)
    
    # Extract test code from response
    test_start = test_response.find("```")
    if test_start != -1:
        test_start = test_response.find("\n", test_start) + 1
        test_end = test_response.find("```", test_start)
        if test_end != -1:
            return test_response[test_start:test_end].strip()
    
    return generate_test_template(language)

def generate_test_template(language: str) -> str:
    """
    Generate a basic test template for the specified language.
    """
    templates = {
        "python": '''import unittest

class TestMain(unittest.TestCase):
    def test_basic_functionality(self):
        # Add your test cases here
        self.assertTrue(True)
    
    def test_edge_cases(self):
        # Add edge case tests here
        pass

if __name__ == "__main__":
    unittest.main()''',
        
        "javascript": '''// Using Jest testing framework
describe('Main functionality', () => {
    test('basic functionality', () => {
        // Add your test cases here
        expect(true).toBe(true);
    });
    
    test('edge cases', () => {
        // Add edge case tests here
    });
});''',
        
        "java": '''import org.junit.Test;
import static org.junit.Assert.*;

public class MainTest {
    @Test
    public void testBasicFunctionality() {
        // Add your test cases here
        assertTrue(true);
    }
    
    @Test
    public void testEdgeCases() {
        // Add edge case tests here
    }
}'''
    }
    
    return templates.get(language.lower(), "// Test template not available for this language")

def calculate_generation_confidence(description: str, code: str, language: str, gpt4all_available: bool) -> float:
    """
    Calculate confidence score for the generated code.
    """
    base_score = 0.5
    
    # Increase confidence based on description clarity
    if len(description.split()) > 5:
        base_score += 0.1
    
    # Increase confidence based on code length and structure
    if len(code.split('\n')) > 3:
        base_score += 0.1
    
    # Check for basic code structure
    if any(keyword in code.lower() for keyword in ['def ', 'function ', 'class ', 'public ']):
        base_score += 0.1
    
    # Increase confidence if GPT4All is available
    if gpt4all_available:
        base_score += 0.2
    
    return min(base_score, 1.0)

def generate_code_suggestions(code: str, language: str, style: str) -> List[str]:
    """
    Generate suggestions for improving the generated code.
    """
    suggestions = []
    
    # Basic suggestions based on language
    if language.lower() == "python":
        if "def " in code and '"""' not in code:
            suggestions.append("Consider adding docstrings to your functions")
        if "try:" not in code and "except:" not in code:
            suggestions.append("Consider adding error handling with try-except blocks")
        if "import " not in code:
            suggestions.append("Consider if you need any imports for additional functionality")
    
    elif language.lower() in ["javascript", "typescript"]:
        if "function" in code and "//" not in code:
            suggestions.append("Consider adding comments to explain complex logic")
        if "console.log" in code:
            suggestions.append("Remove console.log statements before production")
    
    elif language.lower() == "java":
        if "public class" in code and "/**" not in code:
            suggestions.append("Consider adding Javadoc comments")
        if "System.out.println" in code:
            suggestions.append("Consider using a logging framework instead of System.out.println")
    
    # Style-based suggestions
    if style == "clean":
        suggestions.append("Follow clean code principles: meaningful names, small functions")
    elif style == "verbose":
        suggestions.append("Add detailed comments and documentation")
    elif style == "minimal":
        suggestions.append("Keep the code concise and focused")
    
    # General suggestions
    suggestions.extend([
        "Test your code thoroughly before deployment",
        "Consider code review with team members",
        "Follow your project's coding standards"
    ])
    
    return suggestions[:5]  # Return top 5 suggestions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)