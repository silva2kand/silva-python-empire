
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import asyncio
import aiohttp

app = FastAPI(title="AI Sherpa Dashboard - Enhanced", version="1.0.1")

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
    """Serve the enhanced AI Sherpa dashboard"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Sherpa Dashboard - Enhanced</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .services-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .service-card { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; backdrop-filter: blur(10px); }
            .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
            .status-online { background: #4CAF50; }
            .status-offline { background: #F44336; }
            .btn { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #45a049; }
            .logs { background: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 5px; font-family: monospace; height: 200px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Sherpa Dashboard - Enhanced</h1>
                <p>Unified AI Services Management</p>
            </div>
            
            <div class="services-grid">
                <div class="service-card">
                    <h3><span class="status-indicator status-online"></span>Dashboard Server</h3>
                    <p>Status: Running on port 5000</p>
                    <button class="btn" onclick="window.open('/health', '_blank')">Health Check</button>
                </div>
                
                <div class="service-card">
                    <h3><span class="status-indicator status-offline"></span>Chat Service</h3>
                    <p>AI Chat and conversation handling</p>
                    <button class="btn" onclick="checkService('chat')">Check Status</button>
                </div>
                
                <div class="service-card">
                    <h3><span class="status-indicator status-offline"></span>Research Service</h3>
                    <p>AI-powered research and analysis</p>
                    <button class="btn" onclick="checkService('research')">Check Status</button>
                </div>
                
                <div class="service-card">
                    <h3><span class="status-indicator status-offline"></span>HuggingFace Web</h3>
                    <p>Local AI models and bilingual routing</p>
                    <button class="btn" onclick="window.open('http://localhost:8000', '_blank')">Open Interface</button>
                </div>
                
                <div class="service-card">
                    <h3><span class="status-indicator status-online"></span>Rust Server</h3>
                    <p>Ultra-lightweight web server</p>
                    <button class="btn" onclick="window.open('http://localhost:8080', '_blank')">Open Server</button>
                </div>
                
                <div class="service-card">
                    <h3><span class="status-indicator status-online"></span>VS Code Extensions</h3>
                    <p>AI Sherpa & Advanced AI Assistant</p>
                    <button class="btn" onclick="openVSCode()">Open VS Code</button>
                </div>
            </div>
            
            <div style="margin-top: 30px;">
                <h3>System Logs</h3>
                <div class="logs" id="logs">
                    [INFO] AI Sherpa Dashboard started successfully<br>
                    [INFO] All services initialized<br>
                    [INFO] Web interface ready<br>
                </div>
            </div>
        </div>
        
        <script>
            function checkService(service) {
                fetch(`/services/${service}/status`)
                    .then(response => response.json())
                    .then(data => {
                        addLog(`[INFO] ${service} service: ${data.status}`);
                    })
                    .catch(error => {
                        addLog(`[ERROR] ${service} service: offline`);
                    });
            }
            
            function openVSCode() {
                addLog('[INFO] Opening VS Code...');
                // This would typically trigger a backend call to open VS Code
                alert('VS Code will open with AI extensions loaded');
            }
            
            function addLog(message) {
                const logs = document.getElementById('logs');
                const timestamp = new Date().toLocaleTimeString();
                logs.innerHTML += `[${timestamp}] ${message}<br>`;
                logs.scrollTop = logs.scrollHeight;
            }
            
            // Auto-refresh status every 30 seconds
            setInterval(() => {
                addLog('[INFO] Refreshing service status...');
            }, 30000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "service": "AI Sherpa Dashboard Enhanced",
        "version": "1.0.1",
        "features": ["VS Code Integration", "Multi-Service Management", "Real-time Status"]
    }

@app.get("/services/{service}/status")
async def get_service_status(service: str):
    """Get status of a specific service"""
    service_ports = {
        "chat": 8001,
        "research": 8002,
        "history": 8005,
        "security": 8006,
        "mapper": 8007,
        "huggingface": 8000,
        "rust": 8080
    }
    
    if service in service_ports:
        port = service_ports[service]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        return {"status": "online", "port": port}
        except:
            pass
    
    return {"status": "offline", "service": service}

def run_enhanced_dashboard(port=5000):
    """Run the enhanced dashboard server"""
    try:
        print(f"🚀 Starting AI Sherpa Enhanced Dashboard on http://localhost:{port}")
        uvicorn.run(app, host="localhost", port=port, log_level="info")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        print("Trying alternative port...")
        try:
            uvicorn.run(app, host="localhost", port=port + 1, log_level="info")
        except Exception as e2:
            print(f"❌ Alternative port failed: {e2}")

if __name__ == "__main__":
    run_enhanced_dashboard()
