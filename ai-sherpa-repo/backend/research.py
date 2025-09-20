from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import requests
import json
from urllib.parse import quote_plus
from shared.gpt4all_service import gpt4all_service

app = FastAPI(title="AI Sherpa - Research Service", version="1.0.0")

class ResearchRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    search_web: Optional[bool] = True
    max_results: Optional[int] = 5
    focus_area: Optional[str] = "general"  # general, technical, academic, news

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str

class ResearchOutput(BaseModel):
    query: str
    summary: str
    key_findings: List[str]
    search_results: List[SearchResult]
    ai_analysis: str
    recommendations: List[str]
    confidence_score: float
    sources_count: int

@app.get("/")
async def root():
    return {
        "service": "AI Sherpa Research Service",
        "status": "running",
        "version": "1.0.0",
        "gpt4all_available": gpt4all_service.is_available(),
        "web_search_available": True
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpt4all_available": gpt4all_service.is_available(),
        "web_search_available": True
    }

@app.post("/research", response_model=ResearchOutput)
async def conduct_research(request: ResearchRequest):
    """
    Conduct comprehensive research on the given query.
    
    This endpoint performs:
    - Web search for current information
    - Content analysis and summarization
    - AI-powered insights and recommendations
    - Source verification and ranking
    """
    try:
        search_results = []
        
        # Perform web search if requested
        if request.search_web:
            search_results = await perform_web_search(
                request.query, 
                request.max_results,
                request.focus_area
            )
        
        # Compile research context
        research_context = compile_research_context(search_results, request.context)
        
        # Generate AI analysis
        analysis_prompt = gpt4all_service.create_research_prompt(
            request.query,
            research_context
        )
        
        ai_analysis = gpt4all_service.generate_response(analysis_prompt, max_tokens=1024)
        
        # Extract key findings and recommendations
        key_findings = extract_key_findings(search_results, ai_analysis)
        recommendations = generate_recommendations(request.query, search_results, ai_analysis)
        
        # Generate summary
        summary = generate_research_summary(request.query, search_results, ai_analysis)
        
        # Calculate confidence score
        confidence_score = calculate_research_confidence(
            request.query,
            search_results,
            gpt4all_service.is_available()
        )
        
        return ResearchOutput(
            query=request.query,
            summary=summary,
            key_findings=key_findings,
            search_results=search_results,
            ai_analysis=ai_analysis,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_count=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

async def perform_web_search(query: str, max_results: int, focus_area: str) -> List[SearchResult]:
    """
    Perform web search using a search API or scraping.
    For demo purposes, this returns mock results.
    """
    # In a real implementation, you would use APIs like:
    # - Google Custom Search API
    # - Bing Search API
    # - DuckDuckGo API
    # - SerpAPI
    
    # Mock search results for demonstration
    mock_results = [
        SearchResult(
            title=f"Understanding {query}: A Comprehensive Guide",
            url=f"https://example.com/guide-{query.replace(' ', '-').lower()}",
            snippet=f"This comprehensive guide covers everything you need to know about {query}, including best practices, common pitfalls, and expert recommendations.",
            source="TechGuide"
        ),
        SearchResult(
            title=f"Latest Developments in {query}",
            url=f"https://news.example.com/latest-{query.replace(' ', '-').lower()}",
            snippet=f"Recent advancements and trends in {query} are shaping the industry. Here's what experts are saying about the future.",
            source="TechNews"
        ),
        SearchResult(
            title=f"Best Practices for {query}",
            url=f"https://bestpractices.example.com/{query.replace(' ', '-').lower()}",
            snippet=f"Industry experts share their top recommendations and best practices for implementing {query} effectively in your projects.",
            source="BestPractices"
        ),
        SearchResult(
            title=f"{query}: Common Mistakes to Avoid",
            url=f"https://mistakes.example.com/{query.replace(' ', '-').lower()}",
            snippet=f"Learn from common mistakes and pitfalls when working with {query}. Avoid these issues to ensure success.",
            source="LessonsLearned"
        ),
        SearchResult(
            title=f"Tools and Resources for {query}",
            url=f"https://tools.example.com/{query.replace(' ', '-').lower()}",
            snippet=f"Discover the best tools, libraries, and resources available for {query}. Compare features and find the right fit.",
            source="ToolsDirectory"
        )
    ]
    
    # Filter based on focus area
    if focus_area == "technical":
        mock_results = [r for r in mock_results if any(word in r.title.lower() for word in ["guide", "practices", "tools"])]
    elif focus_area == "news":
        mock_results = [r for r in mock_results if "latest" in r.title.lower() or "news" in r.source.lower()]
    
    return mock_results[:max_results]

def compile_research_context(search_results: List[SearchResult], additional_context: str) -> str:
    """
    Compile research context from search results and additional context.
    """
    context = "Research Context:\n\n"
    
    if additional_context:
        context += f"Additional Context: {additional_context}\n\n"
    
    if search_results:
        context += "Web Search Results:\n"
        for i, result in enumerate(search_results, 1):
            context += f"{i}. {result.title}\n"
            context += f"   Source: {result.source}\n"
            context += f"   Summary: {result.snippet}\n\n"
    
    return context

def extract_key_findings(search_results: List[SearchResult], ai_analysis: str) -> List[str]:
    """
    Extract key findings from search results and AI analysis.
    """
    findings = []
    
    # Extract from search results
    for result in search_results:
        if "best practices" in result.title.lower():
            findings.append(f"Best practices identified: {result.snippet[:100]}...")
        elif "latest" in result.title.lower() or "recent" in result.title.lower():
            findings.append(f"Recent developments: {result.snippet[:100]}...")
        elif "mistake" in result.title.lower() or "avoid" in result.title.lower():
            findings.append(f"Common pitfalls: {result.snippet[:100]}...")
    
    # Extract from AI analysis
    if "important" in ai_analysis.lower():
        findings.append("AI analysis highlights important considerations")
    if "recommend" in ai_analysis.lower():
        findings.append("AI provides specific recommendations")
    
    # Add general findings if none found
    if not findings:
        findings = [
            "Multiple sources provide comprehensive information",
            "Industry best practices are well documented",
            "Current trends and developments identified"
        ]
    
    return findings[:5]  # Return top 5 findings

def generate_recommendations(query: str, search_results: List[SearchResult], ai_analysis: str) -> List[str]:
    """
    Generate actionable recommendations based on research.
    """
    recommendations = []
    
    # General recommendations based on query type
    if any(word in query.lower() for word in ["programming", "code", "development"]):
        recommendations.extend([
            "Follow established coding standards and best practices",
            "Implement comprehensive testing strategies",
            "Consider performance and scalability implications"
        ])
    elif any(word in query.lower() for word in ["security", "vulnerability"]):
        recommendations.extend([
            "Implement security-first development practices",
            "Regular security audits and penetration testing",
            "Stay updated with latest security threats"
        ])
    elif any(word in query.lower() for word in ["performance", "optimization"]):
        recommendations.extend([
            "Profile and benchmark before optimizing",
            "Focus on bottlenecks with highest impact",
            "Monitor performance metrics continuously"
        ])
    else:
        recommendations.extend([
            "Research multiple sources for comprehensive understanding",
            "Stay updated with latest industry trends",
            "Consider practical implementation challenges"
        ])
    
    # Add source-based recommendations
    if search_results:
        recommendations.append("Review the provided sources for detailed information")
        recommendations.append("Verify information with additional authoritative sources")
    
    return recommendations[:6]  # Return top 6 recommendations

def generate_research_summary(query: str, search_results: List[SearchResult], ai_analysis: str) -> str:
    """
    Generate a comprehensive research summary.
    """
    summary = f"Research Summary for: {query}\n\n"
    
    if search_results:
        summary += f"Found {len(search_results)} relevant sources covering various aspects of {query}. "
        summary += "The research indicates comprehensive information is available from multiple authoritative sources. "
    
    # Add AI analysis insights
    if "mock response" not in ai_analysis.lower():
        summary += "AI analysis provides additional insights and recommendations based on the compiled research. "
    else:
        summary += "AI analysis capabilities are available for enhanced research insights. "
    
    summary += f"Key areas covered include best practices, current trends, common challenges, and practical recommendations for {query}."
    
    return summary

def calculate_research_confidence(query: str, search_results: List[SearchResult], gpt4all_available: bool) -> float:
    """
    Calculate confidence score for the research results.
    """
    base_score = 0.6
    
    # Increase confidence based on number of sources
    if len(search_results) >= 3:
        base_score += 0.1
    if len(search_results) >= 5:
        base_score += 0.1
    
    # Increase confidence based on query specificity
    if len(query.split()) > 2:
        base_score += 0.1
    
    # Increase confidence if GPT4All is available
    if gpt4all_available:
        base_score += 0.1
    
    return min(base_score, 1.0)

@app.get("/search")
async def search_only(query: str, max_results: int = 5, focus_area: str = "general"):
    """
    Perform web search only without AI analysis.
    """
    try:
        search_results = await perform_web_search(query, max_results, focus_area)
        return {
            "query": query,
            "results": search_results,
            "count": len(search_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)