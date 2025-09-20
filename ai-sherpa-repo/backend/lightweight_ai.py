#!/usr/bin/env python3
"""
Lightweight AI Service for AI Sherpa
Integrates HuggingFace + LangChain with bilingual support
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import torch
import re
import json
import os
from typing import List, Dict, Optional

app = FastAPI(title="Lightweight AI Service", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class EmbeddingRequest(BaseModel):
    text: str
    
class RAGRequest(BaseModel):
    query: str
    context_docs: List[str] = []
    top_k: Optional[int] = 3

class BilingualRouter:
    """Routes queries based on language detection (Tamil/English)"""
    
    def __init__(self):
        self.tamil_pattern = re.compile(r'[\u0B80-\u0BFF]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        self.tamil_keywords = {
            'வணக்கம்', 'எப்படி', 'என்ன', 'எங்கே', 'எப்போது', 'ஏன்', 'யார்',
            'செய்', 'இரு', 'வா', 'போ', 'பார்', 'கேள்', 'சொல்', 'தமிழ்'
        }
        
    def detect_language(self, text: str) -> str:
        """Detect if text is Tamil or English"""
        tamil_count = len(self.tamil_pattern.findall(text))
        english_count = len(self.english_pattern.findall(text))
        tamil_keywords_found = sum(1 for word in text.split() if word in self.tamil_keywords)
        
        if tamil_count > english_count or tamil_keywords_found > 0:
            return "tamil"
        elif english_count > tamil_count:
            return "english"
        else:
            return "english"  # Default fallback
    
    def get_system_prompt(self, language: str) -> str:
        """Get appropriate system prompt based on language"""
        if language == "tamil":
            return "நீங்கள் ஒரு உதவிகரமான AI உதவியாளர். தமிழில் பதிலளிக்கவும்."
        return "You are a helpful AI assistant. Please respond in English."
            
    def route_prompt(self, prompt: str) -> str:
        lang = self.detect_language(prompt)
        system_prompt = self.get_system_prompt(lang)
        if lang == "tamil":
            return f"{system_prompt}\n\nகேள்வி: {prompt}\n\nபதில்:"
        return f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"

class LightweightAI:
    """Main lightweight AI service class with enhanced model management"""
    def __init__(self):
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"  # Default lightweight model
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.llm = None
        self.embedder = None
        self.router = BilingualRouter()
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_cache = {}  # Cache for loaded models
        self.embeddings = {}  # Store embeddings models
        self.langchain_models = {}  # Store LangChain model wrappers
        
        print(f"Initialized LightweightAI service using device: {self.device}")
        
        # Available lightweight models
        self.available_models = {
            "phi3-mini": {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "size": "~2.2GB",
                "description": "Lightweight, fast inference, good for general tasks"
            },
            "distilgpt2": {
                "name": "distilgpt2",
                "size": "~350MB",
                "description": "Very lightweight, fast, basic text generation"
            },
            "flan-t5-small": {
                "name": "google/flan-t5-small",
                "size": "~300MB",
                "description": "Instruction-tuned, good for simple tasks"
            }
        }
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize HuggingFace embeddings for RAG support"""
        try:
            # Use sentence-transformers for embeddings
            self.embeddings["default"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': self.device}
            )
            print("✅ Embeddings initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize embeddings: {e}")
            self.embeddings["default"] = None
        
    def load_model(self, model_key: str = "distilgpt2"):
        """Load the specified lightweight model"""
        try:
            if model_key in self.available_models:
                self.model_name = self.available_models[model_key]["name"]
            
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
            # Load embedding model for RAG
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            
            self.is_loaded = True
            print(f"Model {self.model_name} loaded successfully!")
            
            # Initialize LangChain pipeline for this model
            self._init_langchain_pipeline(model_key)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def _init_langchain_pipeline(self, model_key: str):
        """Initialize LangChain pipeline for the loaded model"""
        try:
            # Create HuggingFace pipeline for LangChain
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Create LangChain wrapper
            self.langchain_models[model_key] = HuggingFacePipeline(
                pipeline=hf_pipeline,
                model_kwargs={"temperature": 0.7, "max_length": 512}
            )
            
            print(f"✅ LangChain pipeline initialized for {model_key}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize LangChain pipeline: {e}")
            self.langchain_models[model_key] = None
    
    def generate_response(self, query: str, max_tokens: int = 512, temperature: float = 0.7) -> Dict:
        """Generate response using the loaded model with enhanced bilingual support"""
        if not self.is_loaded:
            return {"response": "Model not loaded. Please load a model first.", "language": "unknown"}
            
        try:
            # Detect language first
            detected_language = self.router.detect_language(query)
            
            # Route prompt based on language
            routed_prompt = self.router.route_prompt(query)
            
            # Generate response
            response = self.pipe(
                routed_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the input prompt from response
            if generated_text.startswith(routed_prompt):
                generated_text = generated_text[len(routed_prompt):].strip()
            
            return {
                "response": generated_text,
                "language": detected_language,
                "model_used": self.model_name
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "language": "unknown",
                "error": True
            }
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using sentence transformers"""
        if self.embedder is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded")
        
        try:
            embeddings = self.embedder.encode(text)
            return embeddings.tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
    
    def rag_search(self, query: str, documents: List[str], top_k: int = 3) -> Dict:
        """Perform RAG search on provided documents"""
        if self.embedder is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded")
        
        try:
            # Get query embedding
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            
            # Get document embeddings
            doc_embeddings = self.embedder.encode(documents, convert_to_tensor=True)
            
            # Calculate similarities
            scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            
            # Get top-k results
            top_indices = scores.topk(min(top_k, len(documents))).indices
            
            results = []
            for idx in top_indices:
                results.append({
                    'document': documents[idx],
                    'score': scores[idx].item(),
                    'index': idx.item()
                })
            
            # Generate response with context
            context = "\n".join([r['document'] for r in results[:2]])  # Use top 2 for context
            enhanced_query = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            response = self.generate_response(enhanced_query)
            
            return {
                'response': response,
                'relevant_docs': results,
                'context_used': context
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in RAG search: {str(e)}")

# Global AI instance
ai_service = LightweightAI()

@app.on_event("startup")
async def startup_event():
    """Load the default model on startup"""
    try:
        ai_service.load_model("distilgpt2")  # Start with the lightest model
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

@app.get("/")
async def root():
    return {"message": "Lightweight AI Service for AI Sherpa", "status": "running"}

@app.get("/models")
async def get_available_models():
    """Get list of available lightweight models"""
    return {
        "available_models": ai_service.available_models,
        "current_model": ai_service.model_name,
        "is_loaded": ai_service.is_loaded
    }

@app.post("/load_model/{model_key}")
async def load_model(model_key: str):
    """Load a specific model"""
    try:
        ai_service.load_model(model_key)
        return {"message": f"Model {model_key} loaded successfully", "model_name": ai_service.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: QueryRequest):
    """Generate text response with enhanced bilingual support"""
    if not ai_service.is_loaded:
        raise HTTPException(status_code=500, detail="No model loaded")
    
    result = ai_service.generate_response(
        request.query,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["response"])
    
    return {
        "response": result["response"],
        "detected_language": result["language"],
        "model": result.get("model_used", ai_service.model_name),
        "query_original": request.query
    }

@app.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """Get embeddings for text"""
    embeddings = ai_service.get_embeddings(request.text)
    return {
        "embeddings": embeddings,
        "dimension": len(embeddings)
    }

@app.post("/rag")
async def rag_search(request: RAGRequest):
    """Perform RAG search and generation"""
    result = ai_service.rag_search(request.query, request.context_docs, request.top_k)
    return result

@app.post("/langchain/generate")
async def langchain_generate(request: QueryRequest):
    """Generate response using LangChain pipeline"""
    if not ai_service.is_loaded:
        raise HTTPException(status_code=500, detail="No model loaded")
    
    current_model_key = next((k for k, v in ai_service.available_models.items() 
                             if v["name"] == ai_service.model_name), "distilgpt2")
    
    langchain_model = ai_service.langchain_models.get(current_model_key)
    if not langchain_model:
        raise HTTPException(status_code=500, detail="LangChain pipeline not available")
    
    try:
        # Detect language and route prompt
        detected_language = ai_service.router.detect_language(request.query)
        routed_prompt = ai_service.router.route_prompt(request.query)
        
        # Use LangChain for generation
        response = langchain_model.invoke(routed_prompt)
        
        # Clean up response
        if routed_prompt in response:
            response = response[len(routed_prompt):].strip()
        
        return {
            "response": response,
            "detected_language": detected_language,
            "model": ai_service.model_name,
            "method": "langchain",
            "query_original": request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain generation failed: {str(e)}")

@app.get("/embeddings")
async def get_embeddings_info():
    """Get information about available embeddings"""
    embeddings_info = {}
    for name, embedding_model in ai_service.embeddings.items():
        if embedding_model:
            embeddings_info[name] = {
                "model_name": getattr(embedding_model, 'model_name', 'unknown'),
                "available": True
            }
        else:
            embeddings_info[name] = {"available": False}
    
    return {
        "embeddings": embeddings_info,
        "default_available": ai_service.embeddings.get("default") is not None
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    current_model_key = next((k for k, v in ai_service.available_models.items() 
                             if v["name"] == ai_service.model_name), "distilgpt2")
    
    return {
        "status": "healthy",
        "model_loaded": ai_service.is_loaded,
        "model_name": ai_service.model_name if ai_service.is_loaded else None,
        "device": ai_service.device,
        "bilingual_support": True,
        "supported_languages": ["english", "tamil"],
        "cache_size": len(ai_service.models_cache),
        "langchain_available": current_model_key in ai_service.langchain_models,
        "embeddings_available": ai_service.embeddings.get("default") is not None,
        "features": {
            "generation": True,
            "embeddings": ai_service.embeddings.get("default") is not None,
            "langchain": current_model_key in ai_service.langchain_models,
            "bilingual_routing": True,
            "model_switching": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)