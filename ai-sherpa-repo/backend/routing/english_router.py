from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch
from typing import Optional, Dict, Any

class EnglishRouter:
    """English language router with LangChain integration for AI Sherpa"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"English router device set to use {self.device}")
        self.llm = None
        self.chain = None
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the English LangChain pipeline"""
        try:
            # Create HuggingFace pipeline for English generation
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256,  # GPT-2 style padding
                return_full_text=False
            )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(
                pipeline=hf_pipeline,
                model_kwargs={"temperature": 0.7, "max_length": 200}
            )
            
            # English-specific prompt template
            english_prompt = PromptTemplate(
                input_variables=["input"],
                template="""You are a helpful AI assistant. Provide clear, accurate, and useful responses in English.

Question: {input}

Answer:"""
            )
            
            # Create the chain with modern LangChain syntax
            self.chain = english_prompt | self.llm | StrOutputParser()
            
            print(f"✅ English router initialized with {self.model_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize English router: {e}")
            self.llm = None
            self.chain = None
    
    def is_english_text(self, text: str) -> bool:
        """Detect if text is primarily English"""
        # Simple heuristic: check for Latin characters and common English words
        latin_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return True  # Default to English for non-alphabetic text
        
        english_ratio = latin_chars / total_chars
        
        # Common English words for additional detection
        english_keywords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'what', 'where', 'when', 'why', 'how', 'who', 'which'
        }
        
        words = text.lower().split()
        english_keywords_found = sum(1 for word in words if word in english_keywords)
        
        return english_ratio > 0.7 or english_keywords_found > 0
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route and process English queries"""
        if not self.chain:
            return {
                "response": "English router not initialized",
                "language": "english",
                "error": True
            }
        
        if not self.is_english_text(query):
            return {
                "response": "Query does not appear to be in English",
                "language": "unknown",
                "should_route_to": "tamil"
            }
        
        try:
            # Generate response using LangChain
            response = self.chain.invoke({"input": query})
            
            # Clean up response
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            return {
                "response": response,
                "language": "english",
                "model_used": self.model_name,
                "router": "english"
            }
            
        except Exception as e:
            return {
                "response": f"English processing error: {str(e)}",
                "language": "english",
                "error": True
            }
    
    def get_english_system_prompts(self) -> Dict[str, str]:
        """Get various English system prompts for different contexts"""
        return {
            "general": "You are a helpful AI assistant. Please respond clearly and accurately in English.",
            "coding": "You are a programming assistant. Provide clear explanations and well-commented code in English.",
            "research": "You are a research assistant. Provide accurate, well-sourced information in English.",
            "media": "You are a media control assistant. Provide clear instructions for media operations in English.",
            "system": "You are a system administration assistant. Provide safe, accurate technical guidance in English."
        }
    
    def get_specialized_chains(self) -> Dict[str, Any]:
        """Get specialized chains for different domains"""
        if not self.llm:
            return {}
        
        chains = {}
        prompts = {
            "coding": PromptTemplate(
                input_variables=["input"],
                template="""You are an expert programming assistant. Provide clear, well-commented code and explanations.

Programming Question: {input}

Solution:"""
            ),
            "research": PromptTemplate(
                input_variables=["input"],
                template="""You are a research assistant. Provide accurate, detailed information with context.

Research Query: {input}

Research Response:"""
            ),
            "media": PromptTemplate(
                input_variables=["input"],
                template="""You are a media control assistant. Provide step-by-step instructions for media operations.

Media Request: {input}

Instructions:"""
            )
        }
        
        for domain, prompt in prompts.items():
            try:
                chains[domain] = prompt | self.llm | StrOutputParser()
            except Exception as e:
                print(f"⚠️ Failed to create {domain} chain: {e}")
        
        return chains
    
    def health_check(self) -> Dict[str, Any]:
        """Check English router health"""
        specialized_chains = self.get_specialized_chains()
        
        return {
            "status": "healthy" if self.chain else "unhealthy",
            "model_name": self.model_name,
            "device": self.device,
            "langchain_available": self.chain is not None,
            "specialized_chains": list(specialized_chains.keys()),
            "language": "english"
        }

# Global instance for easy import
english_router = EnglishRouter()

# LangChain chain for backward compatibility
english_chain = english_router.chain if english_router.chain else None