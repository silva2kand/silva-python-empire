from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Set
import json
import asyncio
import os
import ast
import re
from pathlib import Path
from collections import defaultdict, deque
import time
from datetime import datetime

# Mock GPT4All service for development
class MockGPT4AllService:
    def __init__(self):
        print("GPT4All not available. Using mock responses for development")
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        return "Mock dependency analysis response for development purposes"

app = FastAPI(title="AI Sherpa - Dependency Mapping Service", version="1.0.0")
gpt_service = MockGPT4AllService()

# Connection Manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_data: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_data[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now(),
            "subscriptions": set()
        }
        print(f"Client {self.client_data[websocket]['client_id']} connected")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = self.client_data.get(websocket, {}).get("client_id", "unknown")
            if websocket in self.client_data:
                del self.client_data[websocket]
            print(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message to client: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict, subscription_filter: str = None):
        disconnected = []
        for connection in self.active_connections:
            try:
                # Check subscription filter
                if subscription_filter:
                    subscriptions = self.client_data.get(connection, {}).get("subscriptions", set())
                    if subscription_filter not in subscriptions:
                        continue
                
                await connection.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Request/Response Models
class DependencyMapRequest(BaseModel):
    project_path: str
    include_external: bool = True
    include_dev_dependencies: bool = False
    max_depth: int = 10
    file_extensions: List[str] = [".py", ".js", ".ts", ".jsx", ".tsx"]
    exclude_patterns: List[str] = ["node_modules", "__pycache__", ".git", ".venv", "venv"]

class DependencyNode(BaseModel):
    id: str
    name: str
    type: str  # file, module, package, class, function
    path: Optional[str] = None
    size: Optional[int] = None
    dependencies: List[str] = []
    dependents: List[str] = []
    metadata: Dict[str, Any] = {}
    risk_score: float = 0.0
    complexity_score: float = 0.0

class DependencyEdge(BaseModel):
    source: str
    target: str
    type: str  # import, call, inheritance, composition
    weight: float = 1.0
    metadata: Dict[str, Any] = {}

class DependencyGraph(BaseModel):
    nodes: List[DependencyNode]
    edges: List[DependencyEdge]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
    generated_at: str

@app.get("/")
async def root():
    return {"message": "AI Sherpa Dependency Mapping Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "dependency-mapping",
        "active_connections": len(manager.active_connections)
    }

# Core Classes
class DependencyMapper:
    def __init__(self, request: DependencyMapRequest):
        self.request = request
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: List[DependencyEdge] = []
        self.file_cache: Dict[str, Dict[str, Any]] = {}
    
    async def create_full_map(self) -> DependencyGraph:
        """Create a complete dependency map for the project."""
        start_time = time.time()
        
        # Scan project files
        files = self._scan_project_files()
        
        # Process each file
        for file_path in files:
            await self._process_file(file_path)
        
        # Calculate additional metrics
        self._calculate_risk_scores()
        self._calculate_complexity_scores()
        
        # Generate statistics
        statistics = self._generate_statistics()
        
        # Create metadata
        metadata = {
            "project_path": self.request.project_path,
            "scan_time": time.time() - start_time,
            "files_processed": len(files),
            "include_external": self.request.include_external,
            "max_depth": self.request.max_depth
        }
        
        return DependencyGraph(
            nodes=list(self.nodes.values()),
            edges=self.edges,
            metadata=metadata,
            statistics=statistics,
            generated_at=datetime.now().isoformat()
        )
    
    def _scan_project_files(self) -> List[str]:
        """Scan project directory for relevant files."""
        files = []
        
        for root, dirs, filenames in os.walk(self.request.project_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.request.exclude_patterns)]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                # Check file extension
                if any(filename.endswith(ext) for ext in self.request.file_extensions):
                    files.append(file_path)
        
        return files
    
    async def _process_file(self, file_path: str):
        """Process a single file to extract dependencies."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Create file node
            file_id = self._get_file_id(file_path)
            file_node = DependencyNode(
                id=file_id,
                name=os.path.basename(file_path),
                type="file",
                path=file_path,
                size=len(content),
                metadata={
                    "extension": Path(file_path).suffix,
                    "lines_of_code": len(content.split('\n')),
                    "last_modified": os.path.getmtime(file_path)
                }
            )
            
            self.nodes[file_id] = file_node
            
            # Extract dependencies based on file type
            if file_path.endswith('.py'):
                await self._process_python_file(file_path, content)
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                await self._process_javascript_file(file_path, content)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    async def _process_python_file(self, file_path: str, content: str):
        """Process Python file for dependencies."""
        try:
            tree = ast.parse(content)
            file_id = self._get_file_id(file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_import_dependency(file_id, alias.name, "import")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_import_dependency(file_id, node.module, "import_from")
                
                elif isinstance(node, ast.FunctionDef):
                    func_id = f"{file_id}::{node.name}"
                    func_node = DependencyNode(
                        id=func_id,
                        name=node.name,
                        type="function",
                        path=file_path,
                        metadata={
                            "line_number": node.lineno,
                            "args_count": len(node.args.args),
                            "is_async": isinstance(node, ast.AsyncFunctionDef)
                        }
                    )
                    self.nodes[func_id] = func_node
                    
                    # Add edge from file to function
                    self.edges.append(DependencyEdge(
                        source=file_id,
                        target=func_id,
                        type="contains"
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    class_id = f"{file_id}::{node.name}"
                    class_node = DependencyNode(
                        id=class_id,
                        name=node.name,
                        type="class",
                        path=file_path,
                        metadata={
                            "line_number": node.lineno,
                            "base_classes": [base.id for base in node.bases if isinstance(base, ast.Name)],
                            "methods_count": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        }
                    )
                    self.nodes[class_id] = class_node
                    
                    # Add edge from file to class
                    self.edges.append(DependencyEdge(
                        source=file_id,
                        target=class_id,
                        type="contains"
                    ))
        
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
    
    async def _process_javascript_file(self, file_path: str, content: str):
        """Process JavaScript/TypeScript file for dependencies."""
        file_id = self._get_file_id(file_path)
        
        # Simple regex-based parsing for imports with properly escaped patterns
        import_patterns = [
            r'import\s+.*?from\s+["\']([^"\'])+["\']',
            r'require\s*\(\s*["\']([^"\'])+["\']\s*\)',
            r'import\s*\(\s*["\']([^"\'])+["\']\s*\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                self._add_import_dependency(file_id, match, "import")
        
        # Extract function declarations
        function_patterns = [
            r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
            r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\(',
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*function\s*\(',
            r'async\s+function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                func_name = match if isinstance(match, str) else match[0]
                func_id = f"{file_id}::{func_name}"
                
                if func_id not in self.nodes:
                    func_node = DependencyNode(
                        id=func_id,
                        name=func_name,
                        type="function",
                        path=file_path,
                        metadata={"language": "javascript"}
                    )
                    self.nodes[func_id] = func_node
                    
                    # Add edge from file to function
                    self.edges.append(DependencyEdge(
                        source=file_id,
                        target=func_id,
                        type="contains"
                    ))
    
    def _add_import_dependency(self, source_id: str, target_module: str, import_type: str):
        """Add an import dependency."""
        # Skip relative imports that are too complex to resolve
        if target_module.startswith('.'):
            return
        
        # Create or get target node
        target_id = f"module::{target_module}"
        
        if target_id not in self.nodes:
            # Determine if it's external or internal
            is_external = not self._is_internal_module(target_module)
            
            if not self.request.include_external and is_external:
                return
            
            target_node = DependencyNode(
                id=target_id,
                name=target_module,
                type="module" if not is_external else "external_module",
                metadata={
                    "is_external": is_external,
                    "import_type": import_type
                }
            )
            self.nodes[target_id] = target_node
        
        # Add edge
        edge = DependencyEdge(
            source=source_id,
            target=target_id,
            type=import_type
        )
        self.edges.append(edge)
        
        # Update node dependencies
        if source_id in self.nodes:
            self.nodes[source_id].dependencies.append(target_id)
        if target_id in self.nodes:
            self.nodes[target_id].dependents.append(source_id)
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Simple heuristic: if it contains project path elements, it's internal
        project_name = os.path.basename(self.request.project_path).lower()
        return project_name in module_name.lower() or module_name.startswith(project_name)
    
    def _get_file_id(self, file_path: str) -> str:
        """Generate a unique ID for a file."""
        rel_path = os.path.relpath(file_path, self.request.project_path)
        return f"file::{rel_path.replace(os.sep, '/')}"
    
    def _calculate_risk_scores(self):
        """Calculate risk scores for nodes based on dependencies."""
        for node in self.nodes.values():
            # Risk factors: high dependency count, being a dependency of many others
            dependency_risk = min(len(node.dependencies) * 0.1, 1.0)
            dependent_risk = min(len(node.dependents) * 0.15, 1.0)
            
            # External dependencies add risk
            external_deps = sum(1 for dep_id in node.dependencies 
                              if dep_id in self.nodes and 
                              self.nodes[dep_id].metadata.get("is_external", False))
            external_risk = min(external_deps * 0.2, 1.0)
            
            node.risk_score = min((dependency_risk + dependent_risk + external_risk) / 3, 1.0)
    
    def _calculate_complexity_scores(self):
        """Calculate complexity scores for nodes."""
        for node in self.nodes.values():
            if node.type == "file":
                # File complexity based on size and dependencies
                size_factor = min((node.size or 0) / 10000, 1.0)  # Normalize to 10KB
                dep_factor = min(len(node.dependencies) / 20, 1.0)  # Normalize to 20 deps
                node.complexity_score = (size_factor + dep_factor) / 2
            
            elif node.type in ["function", "class"]:
                # Function/class complexity based on metadata
                if node.type == "function":
                    args_count = node.metadata.get("args_count", 0)
                    node.complexity_score = min(args_count / 10, 1.0)
                else:  # class
                    methods_count = node.metadata.get("methods_count", 0)
                    node.complexity_score = min(methods_count / 15, 1.0)
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the dependency graph."""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node in self.nodes.values():
            node_types[node.type] += 1
        
        for edge in self.edges:
            edge_types[edge.type] += 1
        
        # Calculate metrics
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        avg_dependencies = sum(len(node.dependencies) for node in self.nodes.values()) / total_nodes if total_nodes > 0 else 0
        avg_risk_score = sum(node.risk_score for node in self.nodes.values()) / total_nodes if total_nodes > 0 else 0
        avg_complexity = sum(node.complexity_score for node in self.nodes.values()) / total_nodes if total_nodes > 0 else 0
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "average_dependencies_per_node": round(avg_dependencies, 2),
            "average_risk_score": round(avg_risk_score, 3),
            "average_complexity_score": round(avg_complexity, 3),
            "density": round(total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0, 4)
        }

@app.post("/map", response_model=DependencyGraph)
async def create_dependency_map(request: DependencyMapRequest):
    """Create a comprehensive dependency map for a project."""
    try:
        if not os.path.exists(request.project_path):
            raise HTTPException(status_code=400, detail="Project path does not exist")
        
        # Start mapping process
        mapper = DependencyMapper(request)
        dependency_graph = await mapper.create_full_map()
        
        return dependency_graph
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dependency mapping failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)