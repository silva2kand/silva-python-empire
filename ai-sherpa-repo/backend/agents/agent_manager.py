"""Agent Manager for AI Sherpa Multi-Agent System"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from .base_agent import BaseAgent, TaskPriority, AgentStatus
from .research_agent import ResearchAgent
from .code_analysis_agent import CodeAnalysisAgent
from .task_coordinator_agent import TaskCoordinatorAgent

class SystemStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class AgentManager:
    """Central manager for the multi-agent system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.status = SystemStatus.INITIALIZING
        self.agents = {}
        self.coordinator = None
        self.task_history = []
        self.performance_metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0,
            "system_uptime": datetime.now()
        }
        self.active_tasks = {}
        self.task_counter = 0
        
        # Initialize the system
        asyncio.create_task(self._initialize_system())
    
    async def _initialize_system(self) -> bool:
        """Initialize the multi-agent system"""
        try:
            self.logger.info("Initializing AI Sherpa Multi-Agent System...")
            
            # Create and register specialized agents
            await self._create_agents()
            
            # Create and configure task coordinator
            await self._setup_coordinator()
            
            # Register all agents with coordinator
            await self._register_agents_with_coordinator()
            
            # Register all agents with message bus
            await self._register_agents_with_message_bus()
            
            # Perform system health check
            health_status = await self._perform_health_check()
            
            if health_status['overall_health'] == 'healthy':
                self.status = SystemStatus.READY
                self.logger.info("Multi-Agent System initialized successfully")
                return True
            else:
                self.status = SystemStatus.ERROR
                self.logger.error(f"System initialization failed: {health_status}")
                return False
                
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Failed to initialize system: {str(e)}")
            return False
    
    async def _create_agents(self) -> None:
        """Create and initialize all specialized agents"""
        try:
            # Create Research Agent
            research_agent = ResearchAgent()
            self.agents[research_agent.agent_id] = research_agent
            self.logger.info(f"Created Research Agent: {research_agent.agent_id}")
            
            # Create Code Analysis Agent
            code_agent = CodeAnalysisAgent()
            self.agents[code_agent.agent_id] = code_agent
            self.logger.info(f"Created Code Analysis Agent: {code_agent.agent_id}")
            
            # Additional agents can be added here
            # Example: DocumentationAgent, TestingAgent, etc.
            
        except Exception as e:
            self.logger.error(f"Failed to create agents: {str(e)}")
            raise
    
    async def _setup_coordinator(self) -> None:
        """Setup the task coordinator"""
        try:
            self.coordinator = TaskCoordinatorAgent()
            self.agents[self.coordinator.agent_id] = self.coordinator
            self.logger.info(f"Created Task Coordinator: {self.coordinator.agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to setup coordinator: {str(e)}")
            raise
    
    async def _register_agents_with_coordinator(self) -> None:
        """Register all agents with the task coordinator"""
        try:
            for agent_id, agent in self.agents.items():
                if agent_id != self.coordinator.agent_id:
                    success = self.coordinator.register_agent(agent)
                    if success:
                        self.logger.info(f"Registered agent {agent_id} with coordinator")
                    else:
                        self.logger.warning(f"Failed to register agent {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to register agents with coordinator: {str(e)}")
            raise
    
    async def _register_agents_with_message_bus(self) -> None:
        """Register all agents with the message bus"""
        try:
            from .communication import get_message_bus
            message_bus = get_message_bus()
            
            for agent_id, agent in self.agents.items():
                agent_info = {
                    "type": agent.__class__.__name__,
                    "capabilities": getattr(agent, 'capabilities', []),
                    "description": getattr(agent, 'description', f"{agent.__class__.__name__} instance")
                }
                
                await message_bus.register_agent(agent_id, agent_info)
                
                # Set message bus for heartbeat functionality
                if hasattr(agent, 'set_message_bus'):
                    agent.set_message_bus(message_bus)
                
                self.logger.info(f"Registered agent {agent_id} with message bus")
                
        except Exception as e:
            self.logger.error(f"Failed to register agents with message bus: {str(e)}")
            raise
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the system"""
        health_report = {
            "overall_health": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agent_health": {},
            "coordinator_health": {},
            "system_metrics": {}
        }
        
        try:
            # Check individual agent health
            healthy_agents = 0
            for agent_id, agent in self.agents.items():
                if agent_id != self.coordinator.agent_id:
                    agent_health = await self._check_agent_health(agent)
                    health_report["agent_health"][agent_id] = agent_health
                    
                    if agent_health["status"] == "healthy":
                        healthy_agents += 1
            
            # Check coordinator health
            if self.coordinator:
                coord_health = await self._check_agent_health(self.coordinator)
                health_report["coordinator_health"] = coord_health
            
            # Calculate overall health
            total_agents = len(self.agents) - 1  # Exclude coordinator
            if total_agents > 0:
                health_percentage = (healthy_agents / total_agents) * 100
                if health_percentage >= 80:
                    health_report["overall_health"] = "healthy"
                elif health_percentage >= 50:
                    health_report["overall_health"] = "degraded"
                else:
                    health_report["overall_health"] = "unhealthy"
            
            # Add system metrics
            health_report["system_metrics"] = {
                "total_agents": len(self.agents),
                "healthy_agents": healthy_agents,
                "coordinator_available": self.coordinator is not None,
                "system_status": self.status.value
            }
            
        except Exception as e:
            health_report["overall_health"] = "error"
            health_report["error"] = str(e)
        
        return health_report
    
    async def _check_agent_health(self, agent: BaseAgent) -> Dict[str, Any]:
        """Check the health of a specific agent"""
        try:
            # Test agent with a simple ping task
            ping_task = {
                "type": "health_check",
                "id": f"health_check_{agent.agent_id}_{datetime.now().timestamp()}",
                "timeout": 5.0
            }
            
            start_time = datetime.now()
            
            # Try to execute a simple task
            if agent.can_handle_task("health_check"):
                result = await asyncio.wait_for(agent.execute_task(ping_task), timeout=5.0)
                response_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "agent_status": agent.status.value,
                    "capabilities": len(agent.capabilities),
                    "last_checked": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "healthy",
                    "response_time": 0.0,
                    "agent_status": agent.status.value,
                    "capabilities": len(agent.capabilities),
                    "last_checked": datetime.now().isoformat(),
                    "note": "Agent doesn't handle health checks, but appears functional"
                }
                
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": "Agent health check timed out",
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for processing requests through the multi-agent system"""
        if self.status != SystemStatus.READY:
            return {
                "success": False,
                "error": f"System not ready. Current status: {self.status.value}",
                "system_status": self.status.value
            }
        
        try:
            self.status = SystemStatus.BUSY
            
            # Generate unique task ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{datetime.now().timestamp()}"
            
            # Prepare task for processing
            task = {
                "id": task_id,
                "original_request": request,
                "created_at": datetime.now().isoformat(),
                "priority": request.get("priority", "medium"),
                **request
            }
            
            # Add to active tasks
            self.active_tasks[task_id] = {
                "task": task,
                "status": "processing",
                "started_at": datetime.now()
            }
            
            # Determine processing strategy
            processing_strategy = await self._determine_processing_strategy(task)
            
            # Process the task based on strategy
            if processing_strategy == "single_agent":
                result = await self._process_with_single_agent(task)
            elif processing_strategy == "multi_agent":
                result = await self._process_with_multiple_agents(task)
            elif processing_strategy == "coordinated":
                result = await self._process_with_coordination(task)
            else:
                result = await self._process_with_fallback(task)
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = datetime.now()
            self.active_tasks[task_id]["result"] = result
            
            # Move to history
            self.task_history.append(self.active_tasks.pop(task_id))
            
            # Update performance metrics
            await self._update_performance_metrics(task, result)
            
            # Prepare response
            response = {
                "success": True,
                "task_id": task_id,
                "processing_strategy": processing_strategy,
                "result": result,
                "processed_at": datetime.now().isoformat()
            }
            
            self.status = SystemStatus.READY
            return response
            
        except Exception as e:
            self.status = SystemStatus.READY
            self.logger.error(f"Request processing failed: {str(e)}")
            
            # Update failed task
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
                self.task_history.append(self.active_tasks.pop(task_id))
            
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id if 'task_id' in locals() else None,
                "processed_at": datetime.now().isoformat()
            }
    
    async def _determine_processing_strategy(self, task: Dict[str, Any]) -> str:
        """Determine the best processing strategy for a task"""
        task_type = task.get("type", "").lower()
        complexity = task.get("complexity", "medium").lower()
        requires_multiple_skills = task.get("requires_multiple_skills", False)
        
        # Complex tasks or those requiring multiple skills need coordination
        if complexity == "high" or requires_multiple_skills:
            return "coordinated"
        
        # Tasks that can benefit from multiple perspectives
        multi_agent_tasks = [
            "comprehensive_analysis",
            "research_and_analysis",
            "code_review_and_research",
            "multi_perspective_evaluation"
        ]
        
        if any(ma_task in task_type for ma_task in multi_agent_tasks):
            return "multi_agent"
        
        # Simple, specialized tasks
        single_agent_tasks = [
            "web_search",
            "code_analysis",
            "syntax_check",
            "documentation_lookup",
            "simple_research"
        ]
        
        if any(sa_task in task_type for sa_task in single_agent_tasks):
            return "single_agent"
        
        # Default to single agent for unknown tasks
        return "single_agent"
    
    async def _process_with_single_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with the most suitable single agent"""
        task_type = task.get("type", "general")
        
        # Find the best agent for this task
        best_agent = await self._find_best_agent_for_task(task_type, task)
        
        if best_agent:
            result = await best_agent.execute_task(task)
            return {
                "strategy": "single_agent",
                "assigned_agent": best_agent.agent_id,
                "agent_name": best_agent.name,
                "result": result
            }
        else:
            return {
                "strategy": "single_agent",
                "error": f"No suitable agent found for task type: {task_type}",
                "available_agents": [agent.name for agent in self.agents.values()]
            }
    
    async def _process_with_multiple_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with multiple agents working in parallel"""
        task_type = task.get("type", "general")
        
        # Find multiple suitable agents
        suitable_agents = await self._find_multiple_agents_for_task(task_type, task)
        
        if len(suitable_agents) >= 2:
            # Execute task with multiple agents in parallel
            agent_tasks = []
            for agent in suitable_agents:
                agent_task = {
                    **task,
                    "agent_perspective": agent.name,
                    "multi_agent_mode": True
                }
                agent_tasks.append(agent.execute_task(agent_task))
            
            # Wait for all agents to complete
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process and combine results
            combined_result = await self._combine_agent_results(agent_results, suitable_agents)
            
            return {
                "strategy": "multi_agent",
                "participating_agents": [agent.agent_id for agent in suitable_agents],
                "agent_names": [agent.name for agent in suitable_agents],
                "individual_results": [
                    {"agent": agent.name, "result": result} 
                    for agent, result in zip(suitable_agents, agent_results)
                    if not isinstance(result, Exception)
                ],
                "combined_result": combined_result
            }
        else:
            # Fall back to single agent
            return await self._process_with_single_agent(task)
    
    async def _process_with_coordination(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process complex task using the task coordinator"""
        if not self.coordinator:
            return {
                "strategy": "coordinated",
                "error": "Task coordinator not available"
            }
        
        # Prepare coordination task
        coordination_task = {
            "type": "task_coordination",
            "main_task": task,
            "subtasks": await self._break_down_task(task),
            "dependencies": await self._analyze_task_dependencies(task)
        }
        
        # Execute through coordinator
        result = await self.coordinator.execute_task(coordination_task)
        
        return {
            "strategy": "coordinated",
            "coordinator": self.coordinator.agent_id,
            "coordination_result": result
        }
    
    async def _process_with_fallback(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing method"""
        # Try to find any available agent
        available_agents = [agent for agent in self.agents.values() if agent.is_available()]
        
        if available_agents:
            # Use the first available agent
            agent = available_agents[0]
            result = await agent.execute_task(task)
            
            return {
                "strategy": "fallback",
                "assigned_agent": agent.agent_id,
                "agent_name": agent.name,
                "result": result,
                "note": "Used fallback processing due to no specific strategy match"
            }
        else:
            return {
                "strategy": "fallback",
                "error": "No available agents for processing",
                "system_status": self.status.value
            }
    
    async def _find_best_agent_for_task(self, task_type: str, task: Dict[str, Any]) -> Optional[BaseAgent]:
        """Find the best agent for a specific task"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent != self.coordinator and agent.can_handle_task(task_type) and agent.is_available():
                # Calculate suitability score
                score = await self._calculate_agent_suitability_score(agent, task_type, task)
                suitable_agents.append((agent, score))
        
        if suitable_agents:
            # Sort by score and return the best agent
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        
        return None
    
    async def _find_multiple_agents_for_task(self, task_type: str, task: Dict[str, Any]) -> List[BaseAgent]:
        """Find multiple agents suitable for a task"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent != self.coordinator and agent.can_handle_task(task_type) and agent.is_available():
                suitable_agents.append(agent)
        
        # Return up to 3 agents for multi-agent processing
        return suitable_agents[:3]
    
    async def _calculate_agent_suitability_score(self, agent: BaseAgent, task_type: str, task: Dict[str, Any]) -> float:
        """Calculate how suitable an agent is for a specific task"""
        base_score = 1.0
        
        # Check if agent has specific capability for this task type
        if task_type in agent.capabilities:
            base_score += 0.5
        
        # Factor in agent performance metrics
        metrics = agent.get_performance_metrics()
        success_rate = metrics.get("success_rate", 100.0) / 100.0
        base_score *= success_rate
        
        # Factor in current workload (prefer less busy agents)
        if hasattr(agent, 'current_workload'):
            workload_penalty = min(0.3, agent.current_workload * 0.1)
            base_score *= (1.0 - workload_penalty)
        
        return base_score
    
    async def _combine_agent_results(self, agent_results: List[Any], agents: List[BaseAgent]) -> Dict[str, Any]:
        """Combine results from multiple agents"""
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(agent_results):
            if isinstance(result, Exception):
                failed_results.append({
                    "agent": agents[i].name if i < len(agents) else "unknown",
                    "error": str(result)
                })
            else:
                successful_results.append({
                    "agent": agents[i].name if i < len(agents) else "unknown",
                    "result": result
                })
        
        # Create combined analysis
        combined_analysis = {
            "total_agents": len(agents),
            "successful_responses": len(successful_results),
            "failed_responses": len(failed_results),
            "success_rate": (len(successful_results) / len(agents) * 100) if agents else 0
        }
        
        # If we have multiple successful results, try to synthesize them
        if len(successful_results) > 1:
            combined_analysis["synthesis"] = await self._synthesize_multiple_results(successful_results)
        elif len(successful_results) == 1:
            combined_analysis["primary_result"] = successful_results[0]["result"]
        
        return {
            "combined_analysis": combined_analysis,
            "individual_results": successful_results,
            "failed_results": failed_results,
            "combination_method": "multi_agent_synthesis"
        }
    
    async def _synthesize_multiple_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize multiple agent results into a coherent response"""
        synthesis = {
            "method": "consensus_based",
            "confidence_level": "medium",
            "key_findings": [],
            "consensus_points": [],
            "divergent_views": [],
            "recommended_action": ""
        }
        
        # Extract key information from each result
        all_findings = []
        for result_data in results:
            result = result_data.get("result", {})
            if isinstance(result, dict):
                # Extract findings, conclusions, or main content
                findings = result.get("findings", result.get("analysis", result.get("content", "")))
                if findings:
                    all_findings.append({
                        "agent": result_data.get("agent", "unknown"),
                        "findings": findings
                    })
        
        # Simple synthesis logic (can be enhanced with NLP)
        if all_findings:
            synthesis["key_findings"] = all_findings
            synthesis["consensus_points"] = ["Multiple agents provided analysis"]
            synthesis["recommended_action"] = "Review all agent perspectives for comprehensive understanding"
            
            # Calculate confidence based on agreement
            if len(all_findings) >= 2:
                synthesis["confidence_level"] = "high"
            else:
                synthesis["confidence_level"] = "medium"
        
        return synthesis
    
    async def _break_down_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down a complex task into subtasks"""
        task_type = task.get("type", "").lower()
        
        # Define common task breakdown patterns
        if "research" in task_type and "analysis" in task_type:
            return [
                {"id": "research_phase", "type": "web_search", "description": "Gather information"},
                {"id": "analysis_phase", "type": "data_analysis", "description": "Analyze gathered data"},
                {"id": "synthesis_phase", "type": "result_synthesis", "description": "Synthesize findings"}
            ]
        elif "code" in task_type:
            return [
                {"id": "code_review", "type": "code_analysis", "description": "Analyze code structure"},
                {"id": "optimization", "type": "code_optimization", "description": "Suggest improvements"},
                {"id": "documentation", "type": "documentation_generation", "description": "Generate documentation"}
            ]
        else:
            # Generic breakdown
            return [
                {"id": "preparation", "type": "task_preparation", "description": "Prepare for task execution"},
                {"id": "execution", "type": "task_execution", "description": "Execute main task"},
                {"id": "validation", "type": "result_validation", "description": "Validate results"}
            ]
    
    async def _analyze_task_dependencies(self, task: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze dependencies between subtasks"""
        # Simple dependency analysis
        return {
            "analysis_phase": ["research_phase"],
            "synthesis_phase": ["research_phase", "analysis_phase"],
            "optimization": ["code_review"],
            "documentation": ["code_review", "optimization"],
            "execution": ["preparation"],
            "validation": ["execution"]
        }
    
    async def _update_performance_metrics(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update system performance metrics"""
        self.performance_metrics["total_tasks_processed"] += 1
        
        if result.get("success", True):
            self.performance_metrics["successful_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        # Update average response time (simplified)
        if "processing_time" in result:
            current_avg = self.performance_metrics["average_response_time"]
            total_tasks = self.performance_metrics["total_tasks_processed"]
            new_time = result["processing_time"]
            
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_tasks - 1) + new_time) / total_tasks
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_status": self.status.value,
            "total_agents": len(self.agents),
            "available_agents": len([a for a in self.agents.values() if a.is_available()]),
            "coordinator_available": self.coordinator is not None,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_history),
            "performance_metrics": self.performance_metrics,
            "uptime": (datetime.now() - self.performance_metrics["system_uptime"]).total_seconds(),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_agent_details(self) -> Dict[str, Any]:
        """Get detailed information about all agents"""
        agent_details = {}
        
        for agent_id, agent in self.agents.items():
            agent_details[agent_id] = {
                "name": agent.name,
                "type": agent.__class__.__name__,
                "status": agent.status.value,
                "capabilities": agent.capabilities,
                "is_available": agent.is_available(),
                "performance_metrics": agent.get_performance_metrics()
            }
        
        return {
            "total_agents": len(self.agents),
            "agent_details": agent_details,
            "coordinator_id": self.coordinator.agent_id if self.coordinator else None
        }
    
    async def shutdown_system(self) -> Dict[str, Any]:
        """Gracefully shutdown the multi-agent system"""
        try:
            self.status = SystemStatus.SHUTDOWN
            
            # Wait for active tasks to complete (with timeout)
            if self.active_tasks:
                self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
                await asyncio.sleep(5)  # Give tasks time to complete
            
            # Shutdown all agents
            for agent_id, agent in self.agents.items():
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            self.logger.info("Multi-Agent System shutdown completed")
            
            return {
                "shutdown_successful": True,
                "final_metrics": self.performance_metrics,
                "shutdown_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            return {
                "shutdown_successful": False,
                "error": str(e),
                "shutdown_at": datetime.now().isoformat()
            }

# Global instance for the multi-agent system
_agent_manager_instance = None

def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance"""
    global _agent_manager_instance
    if _agent_manager_instance is None:
        _agent_manager_instance = AgentManager()
    return _agent_manager_instance

async def initialize_agent_system() -> bool:
    """Initialize the global agent system"""
    manager = get_agent_manager()
    return await manager._initialize_system()

async def process_agent_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a request through the agent system"""
    manager = get_agent_manager()
    return await manager.process_request(request)

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    manager = get_agent_manager()
    return manager.get_system_status()

def get_agent_details() -> Dict[str, Any]:
    """Get details about all agents"""
    manager = get_agent_manager()
    return manager.get_agent_details()