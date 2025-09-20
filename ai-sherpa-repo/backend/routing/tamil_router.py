from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch
from typing import Optional, Dict, Any

class TamilRouter:
    """Tamil language router with LangChain integration for AI Sherpa"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = None
        self.chain = None
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the Tamil LangChain pipeline"""
        try:
            # Create HuggingFace pipeline for Tamil-aware generation
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256  # GPT-2 style padding
            )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(
                pipeline=hf_pipeline,
                model_kwargs={"temperature": 0.7, "max_length": 200}
            )
            
            # Tamil-specific prompt template
            tamil_prompt = PromptTemplate(
                input_variables=["input"],
                template="""நீங்கள் ஒரு உதவிகரமான AI உதவியாளர். தமிழில் தெளிவாகவும் பயனுள்ளதாகவும் பதிலளிக்கவும்.

கேள்வி: {input}

பதில்:"""
            )
            
            # Create the chain with modern LangChain syntax
            self.chain = tamil_prompt | self.llm | StrOutputParser()
            
            print(f"✅ Tamil router initialized with {self.model_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Tamil router: {e}")
            self.llm = None
            self.chain = None
    
    def is_tamil_text(self, text: str) -> bool:
        """Detect if text contains Tamil characters"""
        tamil_range = range(0x0B80, 0x0BFF + 1)  # Tamil Unicode block
        tamil_keywords = {
            'வணக்கம்', 'எப்படி', 'என்ன', 'எங்கே', 'எப்போது', 'ஏன்', 'யார்',
            'செய்', 'இரு', 'வா', 'போ', 'பார்', 'கேள்', 'சொல்', 'தமிழ்', 'உதவி',
            'நன்றி', 'வாங்க', 'போங்க', 'இருக்கு', 'இல்ல', 'ஆமா', 'இல்லை'
        }
        
        # Check for Tamil Unicode characters
        tamil_chars = sum(1 for char in text if ord(char) in tamil_range)
        
        # Check for Tamil keywords
        words = text.split()
        tamil_keywords_found = sum(1 for word in words if word in tamil_keywords)
        
        return tamil_chars > 0 or tamil_keywords_found > 0
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route and process Tamil queries"""
        if not self.chain:
            return {
                "response": "Tamil router not initialized",
                "language": "tamil",
                "error": True
            }
        
        if not self.is_tamil_text(query):
            return {
                "response": "Query does not appear to be in Tamil",
                "language": "unknown",
                "should_route_to": "english"
            }
        
        try:
            # Generate response using LangChain
            response = self.chain.invoke({"input": query})
            
            # Clean up response
            if "பதில்:" in response:
                response = response.split("பதில்:")[-1].strip()
            
            return {
                "response": response,
                "language": "tamil",
                "model_used": self.model_name,
                "router": "tamil"
            }
            
        except Exception as e:
            return {
                "response": f"Tamil processing error: {str(e)}",
                "language": "tamil",
                "error": True
            }
    
    def get_tamil_system_prompts(self) -> Dict[str, str]:
        """Get various Tamil system prompts for different contexts"""
        return {
            "general": "நீங்கள் ஒரு உதவிகரமான AI உதவியாளர். தமிழில் பதிலளிக்கவும்.",
            "coding": "நீங்கள் ஒரு நிரலாக்க உதவியாளர். தமிழில் விளக்கி, ஆங்கிலத்தில் குறியீடு எழுதுங்கள்.",
            "research": "நீங்கள் ஒரு ஆராய்ச்சி உதவியாளர். தமிழில் தெளிவான தகவல் வழங்குங்கள்.",
            "media": "நீங்கள் ஒரு ஊடக கட்டுப்பாட்டு உதவியாளர். தமிழில் வழிகாட்டுங்கள்."
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check Tamil router health"""
        return {
            "status": "healthy" if self.chain else "unhealthy",
            "model_name": self.model_name,
            "device": self.device,
            "langchain_available": self.chain is not None,
            "language": "tamil"
        }

# Global instance for easy import
tamil_router = TamilRouter()

# LangChain chain for backward compatibility
tamil_chain = tamil_router.chain if tamil_router.chain else None