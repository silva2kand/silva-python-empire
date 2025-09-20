
#!/usr/bin/env python3
"""
AI Sherpa Multi-Agent Application - Fixed and Enhanced
LangChain + Tamil/English routing + Agent orchestration
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import routing modules
from routing.english_router import EnglishRouter
from routing.tamil_router import TamilRouter

# Import agent modules
from agents.agent_manager import AgentManager
from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.code_analysis_agent import CodeAnalysisAgent
from agents.task_coordinator_agent import TaskCoordinatorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class MultiAgentRequest(BaseModel):
    query: str
    agent_type: Optional[str] = "general"
    language: Optional[str] = "auto"
    context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    response: str
    agent_used: str
    language_detected: str
    processing_time: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class SystemHealthResponse(BaseModel):
    status: str
    services: Dict[str, Any]
    timestamp: str

class MultiAgentOrchestrator:
    """Enhanced multi-agent orchestrator with proper LangChain integration"""
    
    def __init__(self):
        self.routers = {
            "english": EnglishRouter(),
            "tamil": TamilRouter()
        }
        self.agent_manager = AgentManager()
        self.agent_types = ["coding", "research", "media", "system", "general"]
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing AI Sherpa Multi-Agent System...")
        
        # Initialize routers
        for name, router in self.routers.items():
            try:
                router.initialize()
                logger.info(f"✅ {name.title()} router initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {name} router: {e}")
        
        # Initialize agent manager
        try:
            asyncio.create_task(self.agent_manager.initialize())
            logger.info("✅ Agent manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent manager: {e}")
    
    async def process_query(self, request: MultiAgentRequest) -> AgentResponse:
        """Process query through appropriate router and agent"""
        start_time = datetime.now()
        
        try:
            # Detect language if auto
            if request.language == "auto":
                detected_language = self.detect_language(request.query)
            else:
                detected_language = request.language
            
            # Route to appropriate language router
            if detected_language == "tamil":
                router_response = await self.routers["tamil"].route_query(
                    request.query, request.agent_type, request.context
                )
            else:
                router_response = await self.routers["english"].route_query(
                    request.query, request.agent_type, request.context
                )
            
            # Process through agent manager if needed
            if request.agent_type in self.agent_types:
                agent_response = await self.agent_manager.process_request(
                    request.query, request.agent_type, router_response
                )
            else:
                agent_response = router_response
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                response=agent_response.get("response", "No response generated"),
                agent_used=agent_response.get("agent_used", "unknown"),
                language_detected=detected_language,
                processing_time=processing_time,
                confidence=agent_response.get("confidence", 0.8),
                metadata={
                    "router_used": detected_language,
                    "agent_type": request.agent_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                response=f"Error processing query: {str(e)}",
                agent_used="error_handler",
                language_detected=request.language or "unknown",
                processing_time=processing_time,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            # Simple Tamil detection (can be enhanced)
            tamil_chars = sum(1 for char in text if '஀' <= char <= '௿')
            if tamil_chars > len(text) * 0.3:  # 30% Tamil characters
                return "tamil"
            return "english"
        except Exception:
            return "english"
    
    async def get_system_health(self) -> SystemHealthResponse:
        """Get system health status"""
        services = {}
        
        # Check routers
        for name, router in self.routers.items():
            try:
                services[f"{name}_router"] = {
                    "status": "healthy" if hasattr(router, "chain") else "unhealthy",
                    "last_check": datetime.now().isoformat()
                }
            except Exception as e:
                services[f"{name}_router"] = {
                    "status": "error",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        # Check agent manager
        try:
            agent_status = await self.agent_manager.get_status()
            services["agent_manager"] = agent_status
        except Exception as e:
            services["agent_manager"] = {
                "status": "error",
                "error": str(e)
            }
        
        overall_status = "healthy" if all(
            s.get("status") == "healthy" for s in services.values()
        ) else "degraded"
        
        return SystemHealthResponse(
            status=overall_status,
            services=services,
            timestamp=datetime.now().isoformat()
        )

# Initialize FastAPI app
app = FastAPI(
    title="AI Sherpa Multi-Agent Backend",
    description="LangChain-powered multi-agent system with Tamil/English support",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = MultiAgentOrchestrator()

# API Routes
@app.post("/agent/process", response_model=AgentResponse)
async def process_agent_query(request: MultiAgentRequest):
    """Process query through multi-agent system"""
    return await orchestrator.process_query(request)

@app.get("/agent/health", response_model=SystemHealthResponse)
async def get_agent_health():
    """Get system health status"""
    return await orchestrator.get_system_health()

@app.get("/agent/types")
async def get_agent_types():
    """Get available agent types"""
    return {
        "agent_types": orchestrator.agent_types,
        "languages": ["english", "tamil", "auto"],
        "routers": list(orchestrator.routers.keys())
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Sherpa Multi-Agent Backend",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "LangChain Integration",
            "Tamil/English Support",
            "Multi-Agent Orchestration",
            "Code Analysis",
            "Research Capabilities",
            "Task Coordination"
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting AI Sherpa Multi-Agent Backend...")
    logger.info("🔗 Features: LangChain + Tamil/English + Multi-Agent")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info",
        reload=False
    )
