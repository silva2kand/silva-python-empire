from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

app = FastAPI(title="AI Sherpa Dashboard", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the AI Sherpa dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Please ensure dashboard.html exists in the backend directory.</p>",
            status_code=404
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Sherpa Dashboard"}

@app.get("/services/status")
async def get_services_status():
    """Get status of all AI Sherpa services"""
    import aiohttp
    
    services = {
        "chat": "http://localhost:8001",
        "research": "http://localhost:8002", 
        "history": "http://localhost:8005",
        "security": "http://localhost:8006",
        "mapper": "http://localhost:8007"
    }
    
    status_results = {}
    
    async with aiohttp.ClientSession() as session:
        for service_name, service_url in services.items():
            try:
                async with session.get(f"{service_url}/health", timeout=5) as response:
                    if response.status == 200:
                        status_results[service_name] = {
                            "status": "online",
                            "url": service_url
                        }
                    else:
                        status_results[service_name] = {
                            "status": "error",
                            "url": service_url
                        }
            except Exception as e:
                status_results[service_name] = {
                    "status": "offline",
                    "url": service_url,
                    "error": str(e)
                }
    
    return status_results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)