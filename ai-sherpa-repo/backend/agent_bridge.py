#!/usr/bin/env python3
"""
Agent Bridge Service for AI Sherpa
Integrates specialized HuggingFace agents with AI Sherpa backend
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import json
import requests
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add the huggingface project to Python path
huggingface_path = Path("C:/Users/Siva/Desktop/workspace/hugg and long")
sys.path.append(str(huggingface_path))

try:
    # Import your existing agents
    from project_scanner_agent import ProjectScannerAgent
    from local_model import LocalModelLoader
    from bilingual_router import BilingualRouter
    from rag_embeddings import RAGSystem
except ImportError as e:
    print(f"Warning: Could not import some agents: {e}")
    ProjectScannerAgent = None
    LocalModelLoader = None
    BilingualRouter = None
    RAGSystem = None

app = FastAPI(title="Agent Bridge Service", version="1.0.0")

class ProjectScanRequest(BaseModel):
    project_path: str
    analyze: Optional[bool] = True

class BilingualRequest(BaseModel):
    text: str
    target_language: Optional[str] = "auto"

class RAGQueryRequest(BaseModel):
    query: str
    documents: Optional[List[str]] = []
    top_k: Optional[int] = 3

class AgentTaskRequest(BaseModel):
    agent_type: str
    task: str
    parameters: Optional[Dict[str, Any]] = {}

class AgentBridge:
    """Bridge between HuggingFace agents and AI Sherpa"""
    
    def __init__(self):
        self.model_loader = None
        self.bilingual_router = None
        self.rag_system = None
        self.project_scanner = None
        self.lightweight_ai_url = "http://localhost:8003"
        
        # Initialize available agents
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all available agents"""
        try:
            # Initialize bilingual router
            if BilingualRouter:
                self.bilingual_router = BilingualRouter()
                print("Bilingual router initialized")
            
            # Initialize RAG system
            if RAGSystem:
                self.rag_system = RAGSystem()
                self.rag_system.load_knowledge_base()
                print("RAG system initialized")
            
            # Initialize model loader (lightweight)
            if LocalModelLoader:
                self.model_loader = LocalModelLoader()
                print("Model loader initialized")
                
        except Exception as e:
            print(f"Error initializing agents: {e}")
    
    def scan_project(self, project_path: str, analyze: bool = True) -> Dict:
        """Scan project using ProjectScannerAgent"""
        try:
            if not ProjectScannerAgent or not self.model_loader:
                return {"error": "Project scanner not available"}
            
            # Create project scanner instance
            scanner = ProjectScannerAgent(self.model_loader, project_path)
            
            # Scan project
            scan_results = scanner.scan_project()
            
            result = {
                "scan_results": scan_results,
                "project_path": project_path
            }
            
            # Add analysis if requested
            if analyze:
                try:
                    # Use lightweight AI service for analysis
                    analysis_prompt = f"""
                    Analyze this project structure:
                    - Total files: {len(scan_results.get('files', []))}
                    - Total directories: {len(scan_results.get('directories', []))}
                    - Languages: {json.dumps(scan_results.get('languages', {}), indent=2)}
                    - Total size: {scan_results.get('total_size', 0)} bytes
                    
                    Provide insights about the project type, technologies used, and recommendations.
                    """
                    
                    # Call lightweight AI service
                    response = requests.post(
                        f"{self.lightweight_ai_url}/generate",
                        json={"query": analysis_prompt, "max_tokens": 300}
                    )
                    
                    if response.status_code == 200:
                        analysis = response.json().get("response", "Analysis not available")
                        result["analysis"] = analysis
                    else:
                        result["analysis"] = "Analysis service unavailable"
                        
                except Exception as e:
                    result["analysis"] = f"Analysis error: {str(e)}"
            
            return result
            
        except Exception as e:
            return {"error": f"Project scan failed: {str(e)}"}
    
    def process_bilingual(self, text: str, target_language: str = "auto") -> Dict:
        """Process text with bilingual routing"""
        try:
            if not self.bilingual_router:
                return {"error": "Bilingual router not available"}
            
            detected_language = self.bilingual_router.detect_language(text)
            routed_prompt = self.bilingual_router.route_prompt(text, self.model_loader)
            
            return {
                "original_text": text,
                "detected_language": detected_language,
                "routed_prompt": routed_prompt,
                "target_language": target_language
            }
            
        except Exception as e:
            return {"error": f"Bilingual processing failed: {str(e)}"}
    
    def rag_query(self, query: str, documents: List[str] = None, top_k: int = 3) -> Dict:
        """Perform RAG query using local embeddings"""
        try:
            if not self.rag_system:
                return {"error": "RAG system not available"}
            
            # Add documents to knowledge base if provided
            if documents:
                for doc in documents:
                    self.rag_system.add_document(doc)
            
            # Retrieve relevant context
            relevant_docs = self.rag_system.retrieve_context(query, top_k)
            
            # Enrich prompt with context
            enriched_prompt = self.rag_system.enrich_prompt_with_context(query)
            
            # Generate response using lightweight AI service
            try:
                response = requests.post(
                    f"{self.lightweight_ai_url}/generate",
                    json={"query": enriched_prompt, "max_tokens": 400}
                )
                
                if response.status_code == 200:
                    ai_response = response.json().get("response", "No response generated")
                else:
                    ai_response = "AI service unavailable"
                    
            except Exception as e:
                ai_response = f"AI service error: {str(e)}"
            
            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "enriched_prompt": enriched_prompt,
                "response": ai_response
            }
            
        except Exception as e:
            return {"error": f"RAG query failed: {str(e)}"}
    
    def execute_agent_task(self, agent_type: str, task: str, parameters: Dict = None) -> Dict:
        """Execute a task using specified agent type"""
        if parameters is None:
            parameters = {}
            
        try:
            if agent_type == "project_scanner":
                project_path = parameters.get("project_path", ".")
                analyze = parameters.get("analyze", True)
                return self.scan_project(project_path, analyze)
                
            elif agent_type == "bilingual":
                text = parameters.get("text", task)
                target_language = parameters.get("target_language", "auto")
                return self.process_bilingual(text, target_language)
                
            elif agent_type == "rag":
                query = parameters.get("query", task)
                documents = parameters.get("documents", [])
                top_k = parameters.get("top_k", 3)
                return self.rag_query(query, documents, top_k)
                
            elif agent_type == "lightweight_ai":
                # Direct call to lightweight AI service
                try:
                    response = requests.post(
                        f"{self.lightweight_ai_url}/generate",
                        json={
                            "query": task,
                            "max_tokens": parameters.get("max_tokens", 512),
                            "temperature": parameters.get("temperature", 0.7)
                        }
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return {"error": "Lightweight AI service unavailable"}
                        
                except Exception as e:
                    return {"error": f"Lightweight AI service error: {str(e)}"}
            
            else:
                return {"error": f"Unknown agent type: {agent_type}"}
                
        except Exception as e:
            return {"error": f"Agent task execution failed: {str(e)}"}

# Global agent bridge instance
agent_bridge = AgentBridge()

@app.get("/")
async def root():
    return {"message": "Agent Bridge Service for AI Sherpa", "status": "running"}

@app.get("/agents")
async def get_available_agents():
    """Get list of available agents"""
    agents = {
        "project_scanner": {
            "available": ProjectScannerAgent is not None,
            "description": "Scans project files and analyzes structure"
        },
        "bilingual": {
            "available": agent_bridge.bilingual_router is not None,
            "description": "Tamil/English language detection and routing"
        },
        "rag": {
            "available": agent_bridge.rag_system is not None,
            "description": "Retrieval-Augmented Generation with local embeddings"
        },
        "lightweight_ai": {
            "available": True,
            "description": "Lightweight AI model service integration"
        }
    }
    
    return {"available_agents": agents}

@app.post("/scan_project")
async def scan_project(request: ProjectScanRequest):
    """Scan a project directory"""
    result = agent_bridge.scan_project(request.project_path, request.analyze)
    return result

@app.post("/bilingual")
async def process_bilingual(request: BilingualRequest):
    """Process text with bilingual routing"""
    result = agent_bridge.process_bilingual(request.text, request.target_language)
    return result

@app.post("/rag")
async def rag_query(request: RAGQueryRequest):
    """Perform RAG query"""
    result = agent_bridge.rag_query(request.query, request.documents, request.top_k)
    return result

@app.post("/execute")
async def execute_agent_task(request: AgentTaskRequest):
    """Execute a task using specified agent"""
    result = agent_bridge.execute_agent_task(
        request.agent_type, 
        request.task, 
        request.parameters
    )
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents_initialized": {
            "bilingual_router": agent_bridge.bilingual_router is not None,
            "rag_system": agent_bridge.rag_system is not None,
            "model_loader": agent_bridge.model_loader is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)