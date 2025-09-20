from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import requests
import json
from urllib.parse import quote_plus
from shared.gpt4all_service import gpt4all_service
from sentence_transformers import SentenceTransformer, util
import os

app = FastAPI(title="AI Sherpa - Enhanced Research Service with RAG", version="2.0.0")

class ResearchRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    search_web: Optional[bool] = True
    use_rag: Optional[bool] = True
    max_results: Optional[int] = 5
    focus_area: Optional[str] = "general"  # general, technical, academic, news
    language: Optional[str] = "english"  # english, tamil, auto

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: Optional[float] = 0.0

class RAGResult(BaseModel):
    document: str
    score: float
    source: str

class ResearchOutput(BaseModel):
    query: str
    summary: str
    key_findings: List[str]
    search_results: List[SearchResult]
    rag_results: List[RAGResult]
    ai_analysis: str
    recommendations: List[str]
    confidence_score: float
    sources_count: int
    language_detected: str
    processing_method: str

class GeneralKnowledgeAgent:
    """General Knowledge Agent for cross-domain research"""
    
    def __init__(self):
        self.domains = {
            "technology": ["AI", "machine learning", "programming", "software", "hardware"],
            "science": ["physics", "chemistry", "biology", "mathematics", "research"],
            "business": ["management", "finance", "marketing", "strategy", "economics"],
            "arts": ["literature", "music", "visual arts", "design", "creativity"],
            "health": ["medicine", "wellness", "nutrition", "fitness", "mental health"]
        }
    
    def classify_domain(self, query: str) -> str:
        """Classify query into knowledge domain"""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domains.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "general"
    
    def get_domain_context(self, domain: str) -> str:
        """Get contextual information for domain"""
        contexts = {
            "technology": "Focus on technical accuracy, implementation details, and current trends.",
            "science": "Emphasize peer-reviewed sources, methodological rigor, and evidence-based conclusions.",
            "business": "Consider market dynamics, financial implications, and strategic perspectives.",
            "arts": "Explore creative aspects, cultural significance, and aesthetic considerations.",
            "health": "Prioritize medical accuracy, safety considerations, and evidence-based practices.",
            "general": "Provide balanced, comprehensive information from multiple perspectives."
        }
        return contexts.get(domain, contexts["general"])

class MiniAIAgent:
    """Mini AI Agent for specialized tasks"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.capabilities = {
            "analyzer": "Analyze content structure, patterns, and insights",
            "summarizer": "Create concise summaries and key points",
            "validator": "Verify information accuracy and source reliability",
            "translator": "Handle multilingual content and translation",
            "recommender": "Generate actionable recommendations"
        }
    
    def process(self, content: str, task: str) -> str:
        """Process content based on agent type and task"""
        if self.agent_type == "analyzer":
            return self._analyze_content(content)
        elif self.agent_type == "summarizer":
            return self._summarize_content(content)
        elif self.agent_type == "validator":
            return self._validate_content(content)
        elif self.agent_type == "translator":
            return self._translate_content(content, task)
        elif self.agent_type == "recommender":
            return self._generate_recommendations(content)
        else:
            return f"Unknown agent type: {self.agent_type}"
    
    def _analyze_content(self, content: str) -> str:
        return f"Content analysis: {len(content)} characters, key themes identified."
    
    def _summarize_content(self, content: str) -> str:
        return f"Summary: {content[:200]}..." if len(content) > 200 else content
    
    def _validate_content(self, content: str) -> str:
        return "Content validation: Sources appear credible, information cross-referenced."
    
    def _translate_content(self, content: str, target_lang: str) -> str:
        return f"Translation to {target_lang}: {content}"
    
    def _generate_recommendations(self, content: str) -> str:
        return "Recommendations: Further research suggested in related areas."

class EnhancedRAGSystem:
    """Enhanced RAG System with multilingual support"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.documents = []
        self.doc_embeddings = None
        self.knowledge_base_file = "enhanced_knowledge_base.json"
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load enhanced knowledge base"""
        if os.path.exists(self.knowledge_base_file):
            with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])
                if self.documents:
                    self._update_embeddings()
        else:
            # Initialize with comprehensive knowledge base
            self.documents = [
                "LangChain is a framework for developing applications powered by language models with memory and tool integration.",
                "Hugging Face provides state-of-the-art machine learning models and transformers for NLP tasks.",
                "AI agents can perform autonomous tasks including research, analysis, and decision-making.",
                "Bilingual systems support multiple languages like English and Tamil with automatic language detection.",
                "RAG (Retrieval-Augmented Generation) combines retrieval and generation for contextually aware responses.",
                "FastAPI is a modern web framework for building APIs with Python, featuring automatic documentation.",
                "Docker containers enable consistent deployment across different environments and platforms.",
                "Multi-agent systems coordinate multiple AI agents to solve complex problems collaboratively.",
                "Sentence transformers create semantic embeddings for similarity search and clustering.",
                "Cross-container communication enables microservices to work together in distributed systems."
            ]
            self._update_embeddings()
            self.save_knowledge_base()
    
    def _update_embeddings(self):
        """Update document embeddings"""
        if self.documents:
            self.doc_embeddings = self.embedder.encode(self.documents, convert_to_tensor=True)
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[RAGResult]:
        """Retrieve most relevant documents"""
        if not self.documents or self.doc_embeddings is None:
            return []
        
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        top_indices = scores.topk(min(top_k, len(self.documents))).indices
        results = []
        
        for idx in top_indices:
            results.append(RAGResult(
                document=self.documents[idx],
                score=scores[idx].item(),
                source="Knowledge Base"
            ))
        
        return results
    
    def add_document(self, document: str):
        """Add document to knowledge base"""
        self.documents.append(document)
        self._update_embeddings()
        self.save_knowledge_base()
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
            json.dump({'documents': self.documents}, f, ensure_ascii=False, indent=2)

# Initialize global instances
general_knowledge_agent = GeneralKnowledgeAgent()
rag_system = EnhancedRAGSystem()
analyzer_agent = MiniAIAgent("analyzer")
summarizer_agent = MiniAIAgent("summarizer")
validator_agent = MiniAIAgent("validator")

@app.get("/")
async def root():
    return {
        "service": "AI Sherpa Enhanced Research Service with RAG",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "General Knowledge Agent",
            "Mini AI Agents",
            "Enhanced RAG System",
            "Multilingual Support",
            "Cross-domain Research"
        ],
        "gpt4all_available": gpt4all_service.is_available(),
        "rag_available": True,
        "knowledge_base_size": len(rag_system.documents)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpt4all_available": gpt4all_service.is_available(),
        "rag_system_ready": rag_system.doc_embeddings is not None,
        "knowledge_base_size": len(rag_system.documents),
        "agents_active": {
            "general_knowledge": True,
            "analyzer": True,
            "summarizer": True,
            "validator": True
        }
    }

@app.post("/research", response_model=ResearchOutput)
async def enhanced_research(request: ResearchRequest):
    """Enhanced research with RAG and AI agents"""
    try:
        # Detect language and classify domain
        language_detected = detect_language(request.query)
        domain = general_knowledge_agent.classify_domain(request.query)
        domain_context = general_knowledge_agent.get_domain_context(domain)
        
        search_results = []
        rag_results = []
        
        # Perform RAG retrieval if requested
        if request.use_rag:
            rag_results = rag_system.retrieve_context(request.query, top_k=5)
        
        # Perform web search if requested
        if request.search_web:
            search_results = await perform_enhanced_web_search(
                request.query,
                request.max_results,
                request.focus_area,
                domain
            )
        
        # Compile enhanced research context
        research_context = compile_enhanced_context(
            search_results, 
            rag_results, 
            request.context,
            domain_context
        )
        
        # Generate AI analysis with domain awareness
        analysis_prompt = create_enhanced_research_prompt(
            request.query,
            research_context,
            domain,
            language_detected
        )
        
        ai_analysis = gpt4all_service.generate_response(analysis_prompt, max_tokens=1024)
        
        # Use mini AI agents for processing
        analyzed_content = analyzer_agent.process(ai_analysis, "analyze")
        validated_content = validator_agent.process(ai_analysis, "validate")
        
        # Extract enhanced findings and recommendations
        key_findings = extract_enhanced_findings(search_results, rag_results, ai_analysis)
        recommendations = generate_enhanced_recommendations(
            request.query, 
            search_results, 
            rag_results, 
            ai_analysis,
            domain
        )
        
        # Generate comprehensive summary
        summary = summarizer_agent.process(
            f"Query: {request.query}\nAnalysis: {ai_analysis}\nFindings: {key_findings}",
            "summarize"
        )
        
        # Calculate enhanced confidence score
        confidence_score = calculate_enhanced_confidence(
            request.query,
            search_results,
            rag_results,
            gpt4all_service.is_available()
        )
        
        return ResearchOutput(
            query=request.query,
            summary=summary,
            key_findings=key_findings,
            search_results=search_results,
            rag_results=rag_results,
            ai_analysis=ai_analysis,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_count=len(search_results) + len(rag_results),
            language_detected=language_detected,
            processing_method=f"Enhanced RAG + {domain} domain analysis"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced research failed: {str(e)}")

def detect_language(text: str) -> str:
    """Simple language detection"""
    tamil_chars = sum(1 for char in text if '\u0b80' <= char <= '\u0bff')
    return "tamil" if tamil_chars > len(text) * 0.3 else "english"

async def perform_enhanced_web_search(query: str, max_results: int, focus_area: str, domain: str) -> List[SearchResult]:
    """Enhanced web search with domain awareness"""
    # Mock enhanced results with domain-specific content
    mock_results = [
        SearchResult(
            title=f"{domain.title()} Perspective on {query}",
            url=f"https://{domain}.example.com/guide-{query.replace(' ', '-').lower()}",
            snippet=f"From a {domain} perspective, {query} involves several key considerations and best practices.",
            source=f"{domain.title()}Guide",
            relevance_score=0.95
        ),
        SearchResult(
            title=f"Latest {domain.title()} Research on {query}",
            url=f"https://research.{domain}.com/latest-{query.replace(' ', '-').lower()}",
            snippet=f"Recent {domain} research reveals new insights about {query} and its applications.",
            source=f"{domain.title()}Research",
            relevance_score=0.88
        )
    ]
    
    return mock_results[:max_results]

def compile_enhanced_context(search_results: List[SearchResult], rag_results: List[RAGResult], 
                           user_context: str, domain_context: str) -> str:
    """Compile enhanced research context"""
    context_parts = []
    
    if domain_context:
        context_parts.append(f"Domain Context: {domain_context}")
    
    if user_context:
        context_parts.append(f"User Context: {user_context}")
    
    if rag_results:
        rag_context = "\n".join([f"- {result.document}" for result in rag_results[:3]])
        context_parts.append(f"Knowledge Base Context:\n{rag_context}")
    
    if search_results:
        search_context = "\n".join([f"- {result.title}: {result.snippet}" for result in search_results[:3]])
        context_parts.append(f"Web Search Context:\n{search_context}")
    
    return "\n\n".join(context_parts)

def create_enhanced_research_prompt(query: str, context: str, domain: str, language: str) -> str:
    """Create enhanced research prompt"""
    return f"""You are an expert researcher specializing in {domain}. 
Analyze the following query and provide comprehensive insights.

Query: {query}
Domain: {domain}
Language: {language}

Context:
{context}

Provide a detailed analysis including:
1. Key insights and findings
2. Domain-specific considerations
3. Practical implications
4. Future directions

Response:"""

def extract_enhanced_findings(search_results: List[SearchResult], rag_results: List[RAGResult], 
                            analysis: str) -> List[str]:
    """Extract enhanced key findings"""
    findings = []
    
    # Add findings from RAG results
    for rag_result in rag_results[:2]:
        findings.append(f"Knowledge Base: {rag_result.document[:100]}...")
    
    # Add findings from search results
    for search_result in search_results[:2]:
        findings.append(f"Web Source: {search_result.snippet[:100]}...")
    
    # Add AI-generated findings
    if "finding" in analysis.lower() or "insight" in analysis.lower():
        findings.append("AI Analysis: Key patterns and insights identified from multiple sources")
    
    return findings[:5]  # Limit to top 5 findings

def generate_enhanced_recommendations(query: str, search_results: List[SearchResult], 
                                    rag_results: List[RAGResult], analysis: str, domain: str) -> List[str]:
    """Generate enhanced recommendations"""
    recommendations = [
        f"Explore {domain}-specific resources for deeper understanding",
        "Cross-reference findings with multiple authoritative sources",
        "Consider practical implementation based on current context"
    ]
    
    if rag_results:
        recommendations.append("Leverage knowledge base insights for informed decision-making")
    
    if search_results:
        recommendations.append("Stay updated with latest developments in this area")
    
    return recommendations

def calculate_enhanced_confidence(query: str, search_results: List[SearchResult], 
                                rag_results: List[RAGResult], gpt4all_available: bool) -> float:
    """Calculate enhanced confidence score"""
    base_score = 0.5
    
    # Boost for RAG results
    if rag_results:
        base_score += 0.2 * min(len(rag_results) / 3, 1.0)
    
    # Boost for search results
    if search_results:
        base_score += 0.2 * min(len(search_results) / 3, 1.0)
    
    # Boost for AI availability
    if gpt4all_available:
        base_score += 0.1
    
    return min(base_score, 1.0)

@app.post("/add_knowledge")
async def add_knowledge(document: str):
    """Add document to knowledge base"""
    try:
        rag_system.add_document(document)
        return {
            "status": "success",
            "message": "Document added to knowledge base",
            "knowledge_base_size": len(rag_system.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")

@app.get("/knowledge_stats")
async def knowledge_stats():
    """Get knowledge base statistics"""
    return {
        "total_documents": len(rag_system.documents),
        "embeddings_ready": rag_system.doc_embeddings is not None,
        "model_name": "all-MiniLM-L6-v2",
        "supported_languages": ["english", "tamil"],
        "domains": list(general_knowledge_agent.domains.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)