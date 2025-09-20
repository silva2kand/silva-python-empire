try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False
    GPT4All = None

class GPT4AllService:
    """Shared service for GPT4All local LLM interactions."""
    
    def __init__(self):
        self.model = None
        self.model_name = "mistral-7b-openorca.Q4_0.gguf"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the GPT4All model if available."""
        if not GPT4ALL_AVAILABLE:
            print("GPT4All not available. Using mock responses for development.")
            return
        
        try:
            self.model = GPT4All(self.model_name)
            print(f"GPT4All model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Failed to load GPT4All model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if GPT4All is available and loaded."""
        return GPT4ALL_AVAILABLE and self.model is not None
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using GPT4All or mock response."""
        if not GPT4ALL_AVAILABLE or self.model is None:
            return self._generate_mock_response(prompt)
        
        try:
            response = self.model.generate(prompt, max_tokens=max_tokens)
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock responses for development when GPT4All is not available."""
        if "comprehend" in prompt.lower() or "analyze" in prompt.lower():
            return """**Code Analysis Summary:**

1. **Purpose**: This code appears to implement core functionality with proper error handling.
2. **Key Functions**: Main processing functions with input validation and output formatting.
3. **Potential Issues**: Consider adding more comprehensive error handling and input validation.
4. **Dependencies**: Standard library imports detected.

*Note: This is a mock response for development. Install GPT4All for actual AI analysis.*"""
        
        elif "generate" in prompt.lower() or "create" in prompt.lower():
            return """```python
# Generated code example
def example_function(param1, param2):
    \"\"\"Example function implementation.
    
    Args:
        param1: First parameter
        param2: Second parameter
    
    Returns:
        Processed result
    \"\"\"
    try:
        result = param1 + param2
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None
```

*Note: This is a mock response for development. Install GPT4All for actual code generation.*"""
        
        elif "security" in prompt.lower() or "vulnerability" in prompt.lower():
            return """**Security Analysis Summary:**

1. **Risk Level**: Medium
2. **Detected Issues**: No critical vulnerabilities found in automated scan.
3. **Recommendations**: 
   - Implement input validation
   - Add authentication checks
   - Use parameterized queries
4. **Best Practices**: Follow OWASP guidelines for secure coding.

*Note: This is a mock response for development. Install GPT4All for actual security analysis.*"""
        
        elif "research" in prompt.lower():
            return """**Research Summary:**

Based on the query, here are key findings:

1. **Overview**: The topic involves modern development practices and tools.
2. **Key Concepts**: Industry-standard approaches and methodologies.
3. **Best Practices**: Follow established patterns and conventions.
4. **Recommendations**: Consider scalability, maintainability, and performance.

*Note: This is a mock response for development. Install GPT4All for actual research capabilities.*"""
        
        else:
            return """This is a mock response from the GPT4All service. The actual model is not available in the current environment.

*Note: Install GPT4All and required dependencies for full functionality.*"""
    
    def create_code_comprehension_prompt(self, code: str, language: str, question: str = "") -> str:
        """Create a structured prompt for code comprehension."""
        prompt = f"""Analyze the following {language} code and provide a comprehensive analysis:

Code:
```{language}
{code}
```

Please provide:
1. A brief summary of what this code does
2. Key functions and their purposes
3. Potential issues or improvements
4. Dependencies and imports used
5. Overall code quality assessment
"""
        
        if question:
            prompt += f"\n\nSpecific Question: {question}"
        
        return prompt
    
    def create_code_generation_prompt(self, description: str, language: str, requirements: str = "") -> str:
        """Create a structured prompt for code generation."""
        prompt = f"""Generate {language} code based on the following description:

Description: {description}

Requirements:
- Write clean, well-documented code
- Include proper error handling
- Follow best practices for {language}
- Add appropriate comments
"""
        
        if requirements:
            prompt += f"\n\nAdditional Requirements:\n{requirements}"
        
        return prompt
    
    def create_research_prompt(self, query: str, context: str = "") -> str:
        """Create a structured prompt for research tasks."""
        prompt = f"""Research Query: {query}

Please provide:
1. Overview of the topic
2. Key concepts and definitions
3. Current best practices
4. Relevant tools and technologies
5. Recommendations and next steps
"""
        
        if context:
            prompt += f"\n\nContext: {context}"
        
        return prompt
    
    def create_security_analysis_prompt(self, code: str, language: str) -> str:
        """Create a structured prompt for security analysis."""
        prompt = f"""Perform a security analysis of the following {language} code:

Code:
```{language}
{code}
```

Please analyze for:
1. Common security vulnerabilities (OWASP Top 10)
2. Input validation issues
3. Authentication and authorization flaws
4. Data exposure risks
5. Injection vulnerabilities
6. Recommendations for fixes
"""
        
        return prompt
    
    def create_bug_prediction_prompt(self, code: str, language: str, git_history: str = "") -> str:
        """Create a structured prompt for bug prediction analysis."""
        prompt = f"""Analyze the following {language} code for potential bugs and issues:

Current Code ({language}):
```{language}
{code}
```

Git History Summary:
{git_history}

Please provide:
1. Bug risk assessment (Low/Medium/High)
2. Potential bug-prone areas in the code
3. Analysis based on change patterns
4. Recommendations for reducing bug risk
5. Suggested testing strategies
"""
        
        return prompt

# Global instance for shared use across services
gpt4all_service = GPT4AllService()