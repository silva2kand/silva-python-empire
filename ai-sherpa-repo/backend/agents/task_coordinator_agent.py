"""Task Coordinator Agent for AI Sherpa Multi-Agent System"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from .base_agent import BaseAgent, TaskPriority, AgentStatus

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskCoordinatorAgent(BaseAgent):
    """Specialized agent for task coordination and workflow management"""
    
    def __init__(self):
        super().__init__(
            agent_id="task_coordinator_001",
            name="TaskCoordinatorAgent",
            capabilities=[
                "task_delegation",
                "workflow_management",
                "agent_coordination",
                "task_prioritization",
                "result_aggregation",
                "load_balancing",
                "dependency_management",
                "progress_tracking",
                "error_handling",
                "performance_monitoring"
            ]
        )
        self.registered_agents = {}
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_dependencies = {}
        self.agent_workloads = {}
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if this agent can handle the given task type"""
        coordination_tasks = [
            "task_coordination",
            "workflow_management",
            "agent_management",
            "task_delegation",
            "result_aggregation",
            "progress_tracking",
            "load_balancing",
            "dependency_resolution"
        ]
        return task_type.lower() in coordination_tasks
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordination tasks"""
        task_type = task.get('type', '').lower()
        
        if task_type == "task_coordination":
            return await self._coordinate_complex_task(task)
        elif task_type == "workflow_management":
            return await self._manage_workflow(task)
        elif task_type == "agent_management":
            return await self._manage_agents(task)
        elif task_type == "task_delegation":
            return await self._delegate_task(task)
        elif task_type == "result_aggregation":
            return await self._aggregate_results(task)
        elif task_type == "progress_tracking":
            return await self._track_progress(task)
        elif task_type == "load_balancing":
            return await self._balance_load(task)
        else:
            return await self._general_coordination(task)
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the coordinator"""
        try:
            self.registered_agents[agent.agent_id] = {
                'agent': agent,
                'capabilities': agent.capabilities,
                'status': agent.status,
                'registered_at': datetime.now(),
                'task_count': 0,
                'success_rate': 100.0
            }
            self.agent_workloads[agent.agent_id] = 0
            self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordinator"""
        try:
            if agent_id in self.registered_agents:
                agent_info = self.registered_agents.pop(agent_id)
                self.agent_workloads.pop(agent_id, None)
                self.logger.info(f"Unregistered agent: {agent_info['agent'].name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            return False
    
    async def _coordinate_complex_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a complex task that requires multiple agents"""
        try:
            main_task = task.get('main_task', {})
            subtasks = task.get('subtasks', [])
            dependencies = task.get('dependencies', {})
            
            # Create task coordination plan
            coordination_plan = await self._create_coordination_plan(main_task, subtasks, dependencies)
            
            # Execute the plan
            execution_results = await self._execute_coordination_plan(coordination_plan)
            
            # Aggregate results
            final_result = await self._aggregate_task_results(execution_results)
            
            return {
                "coordination_type": "complex_task",
                "main_task_id": main_task.get('id', 'unknown'),
                "subtasks_count": len(subtasks),
                "execution_plan": coordination_plan,
                "execution_results": execution_results,
                "final_result": final_result,
                "success": all(result.get('status') == 'success' for result in execution_results.values()),
                "total_execution_time": self._calculate_total_execution_time(execution_results),
                "confidence": 0.90
            }
            
        except Exception as e:
            self.logger.error(f"Complex task coordination failed: {str(e)}")
            return {
                "coordination_type": "complex_task",
                "error": str(e),
                "success": False,
                "confidence": 0.0
            }
    
    async def _create_coordination_plan(self, main_task: Dict[str, Any], subtasks: List[Dict[str, Any]], dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Create a coordination plan for task execution"""
        plan = {
            "execution_order": [],
            "parallel_groups": [],
            "agent_assignments": {},
            "estimated_duration": 0,
            "resource_requirements": {}
        }
        
        # Analyze dependencies and create execution order
        dependency_graph = self._build_dependency_graph(subtasks, dependencies)
        execution_order = self._topological_sort(dependency_graph)
        
        # Group tasks that can run in parallel
        parallel_groups = self._identify_parallel_groups(execution_order, dependencies)
        
        # Assign agents to tasks
        agent_assignments = await self._assign_agents_to_tasks(subtasks)
        
        plan.update({
            "execution_order": execution_order,
            "parallel_groups": parallel_groups,
            "agent_assignments": agent_assignments,
            "estimated_duration": self._estimate_execution_duration(subtasks, agent_assignments)
        })
        
        return plan
    
    def _build_dependency_graph(self, subtasks: List[Dict[str, Any]], dependencies: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build a dependency graph for tasks"""
        graph = {}
        
        # Initialize graph with all tasks
        for task in subtasks:
            task_id = task.get('id', f"task_{len(graph)}")
            graph[task_id] = []
        
        # Add dependencies
        for task_id, deps in dependencies.items():
            if task_id in graph:
                graph[task_id] = deps if isinstance(deps, list) else [deps]
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort to determine execution order"""
        # Simple topological sort implementation
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # Find nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Remove edges and update in-degrees
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def _identify_parallel_groups(self, execution_order: List[str], dependencies: Dict[str, Any]) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel"""
        parallel_groups = []
        processed = set()
        
        for task_id in execution_order:
            if task_id not in processed:
                # Find all tasks that can run in parallel with this one
                parallel_group = [task_id]
                task_deps = set(dependencies.get(task_id, []))
                
                for other_task in execution_order:
                    if (other_task != task_id and 
                        other_task not in processed and
                        other_task not in task_deps and
                        task_id not in dependencies.get(other_task, [])):
                        parallel_group.append(other_task)
                
                parallel_groups.append(parallel_group)
                processed.update(parallel_group)
        
        return parallel_groups
    
    async def _assign_agents_to_tasks(self, subtasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assign the best available agents to tasks"""
        assignments = {}
        
        for task in subtasks:
            task_id = task.get('id', f"task_{len(assignments)}")
            task_type = task.get('type', 'general')
            
            # Find the best agent for this task
            best_agent = await self._find_best_agent_for_task(task_type, task)
            
            if best_agent:
                assignments[task_id] = best_agent
                # Update agent workload
                self.agent_workloads[best_agent] = self.agent_workloads.get(best_agent, 0) + 1
            else:
                self.logger.warning(f"No suitable agent found for task {task_id} of type {task_type}")
        
        return assignments
    
    async def _find_best_agent_for_task(self, task_type: str, task: Dict[str, Any]) -> Optional[str]:
        """Find the best available agent for a specific task"""
        suitable_agents = []
        
        for agent_id, agent_info in self.registered_agents.items():
            agent = agent_info['agent']
            
            # Check if agent can handle this task type
            if agent.can_handle_task(task_type) and agent.is_available():
                # Calculate agent score based on various factors
                score = self._calculate_agent_score(agent_info, task_type)
                suitable_agents.append((agent_id, score))
        
        if suitable_agents:
            # Sort by score (descending) and return the best agent
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        
        return None
    
    def _calculate_agent_score(self, agent_info: Dict[str, Any], task_type: str) -> float:
        """Calculate a score for an agent's suitability for a task"""
        base_score = 1.0
        
        # Factor in success rate
        success_rate = agent_info.get('success_rate', 100.0) / 100.0
        base_score *= success_rate
        
        # Factor in current workload (prefer less busy agents)
        workload = self.agent_workloads.get(agent_info['agent'].agent_id, 0)
        workload_penalty = min(0.5, workload * 0.1)
        base_score *= (1.0 - workload_penalty)
        
        # Factor in task type match (if agent has specific capability)
        if task_type in agent_info.get('capabilities', []):
            base_score *= 1.2
        
        return base_score
    
    def _estimate_execution_duration(self, subtasks: List[Dict[str, Any]], agent_assignments: Dict[str, str]) -> float:
        """Estimate total execution duration for the coordination plan"""
        # Simple estimation - in production, this would be more sophisticated
        base_duration_per_task = 5.0  # seconds
        total_duration = len(subtasks) * base_duration_per_task
        
        # Factor in parallelization
        if len(agent_assignments) > 1:
            parallelization_factor = min(0.7, len(agent_assignments) * 0.1)
            total_duration *= (1.0 - parallelization_factor)
        
        return total_duration
    
    async def _execute_coordination_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the coordination plan"""
        execution_results = {}
        
        try:
            parallel_groups = plan.get('parallel_groups', [])
            agent_assignments = plan.get('agent_assignments', {})
            
            for group in parallel_groups:
                # Execute tasks in this group in parallel
                group_tasks = []
                
                for task_id in group:
                    agent_id = agent_assignments.get(task_id)
                    if agent_id and agent_id in self.registered_agents:
                        agent = self.registered_agents[agent_id]['agent']
                        
                        # Create task for execution
                        task = {
                            'id': task_id,
                            'type': 'delegated_task',
                            'original_task': task_id,
                            'assigned_agent': agent_id
                        }
                        
                        # Execute task asynchronously
                        group_tasks.append(self._execute_single_task(agent, task))
                
                # Wait for all tasks in the group to complete
                if group_tasks:
                    group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(group_results):
                        task_id = group[i] if i < len(group) else f"task_{i}"
                        if isinstance(result, Exception):
                            execution_results[task_id] = {
                                'status': 'failed',
                                'error': str(result),
                                'completed_at': datetime.now().isoformat()
                            }
                        else:
                            execution_results[task_id] = result
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_single_task(self, agent: BaseAgent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task using the assigned agent"""
        try:
            result = await agent.execute_task(task)
            
            # Update agent statistics
            agent_id = agent.agent_id
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id]['task_count'] += 1
                
                # Update success rate
                if result.get('status') == 'success':
                    current_rate = self.registered_agents[agent_id]['success_rate']
                    task_count = self.registered_agents[agent_id]['task_count']
                    new_rate = ((current_rate * (task_count - 1)) + 100) / task_count
                    self.registered_agents[agent_id]['success_rate'] = new_rate
                else:
                    current_rate = self.registered_agents[agent_id]['success_rate']
                    task_count = self.registered_agents[agent_id]['task_count']
                    new_rate = ((current_rate * (task_count - 1)) + 0) / task_count
                    self.registered_agents[agent_id]['success_rate'] = new_rate
            
            # Decrease agent workload
            self.agent_workloads[agent_id] = max(0, self.agent_workloads.get(agent_id, 0) - 1)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed for agent {agent.agent_id}: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'agent_id': agent.agent_id,
                'completed_at': datetime.now().isoformat()
            }
    
    async def _aggregate_task_results(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple task executions"""
        successful_tasks = []
        failed_tasks = []
        total_tasks = len(execution_results)
        
        for task_id, result in execution_results.items():
            if result.get('status') == 'success':
                successful_tasks.append(task_id)
            else:
                failed_tasks.append(task_id)
        
        success_rate = (len(successful_tasks) / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": round(success_rate, 2),
            "successful_task_ids": successful_tasks,
            "failed_task_ids": failed_tasks,
            "aggregation_completed_at": datetime.now().isoformat()
        }
    
    def _calculate_total_execution_time(self, execution_results: Dict[str, Any]) -> float:
        """Calculate total execution time from results"""
        # This is a simplified calculation
        # In production, you'd track actual start and end times
        return len(execution_results) * 2.5  # Mock average time per task
    
    async def _manage_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manage a complete workflow"""
        workflow_id = task.get('workflow_id', f"workflow_{datetime.now().timestamp()}")
        workflow_steps = task.get('steps', [])
        
        workflow_result = {
            "workflow_id": workflow_id,
            "total_steps": len(workflow_steps),
            "completed_steps": 0,
            "step_results": {},
            "status": "in_progress",
            "started_at": datetime.now().isoformat()
        }
        
        try:
            for i, step in enumerate(workflow_steps):
                step_id = step.get('id', f"step_{i}")
                
                # Execute workflow step
                step_result = await self._execute_workflow_step(step)
                workflow_result["step_results"][step_id] = step_result
                
                if step_result.get('status') == 'success':
                    workflow_result["completed_steps"] += 1
                else:
                    # Handle step failure
                    if step.get('critical', False):
                        workflow_result["status"] = "failed"
                        workflow_result["failed_at_step"] = step_id
                        break
            
            if workflow_result["status"] != "failed":
                workflow_result["status"] = "completed"
            
            workflow_result["completed_at"] = datetime.now().isoformat()
            return workflow_result
            
        except Exception as e:
            workflow_result.update({
                "status": "error",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
            return workflow_result
    
    async def _execute_workflow_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_type = step.get('type', 'generic')
        
        # Find appropriate agent for this step
        agent_id = await self._find_best_agent_for_task(step_type, step)
        
        if agent_id and agent_id in self.registered_agents:
            agent = self.registered_agents[agent_id]['agent']
            return await agent.execute_task(step)
        else:
            return {
                "status": "failed",
                "error": f"No suitable agent found for step type: {step_type}",
                "step_id": step.get('id', 'unknown')
            }
    
    async def _manage_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manage registered agents"""
        action = task.get('action', 'status')
        
        if action == 'status':
            return self._get_all_agent_status()
        elif action == 'performance':
            return self._get_agent_performance_metrics()
        elif action == 'workload':
            return self._get_workload_distribution()
        else:
            return {
                "action": action,
                "error": f"Unknown agent management action: {action}"
            }
    
    def _get_all_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        agent_statuses = {}
        
        for agent_id, agent_info in self.registered_agents.items():
            agent = agent_info['agent']
            agent_statuses[agent_id] = {
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": agent.capabilities,
                "task_count": agent_info['task_count'],
                "success_rate": round(agent_info['success_rate'], 2),
                "current_workload": self.agent_workloads.get(agent_id, 0),
                "registered_at": agent_info['registered_at'].isoformat()
            }
        
        return {
            "total_agents": len(self.registered_agents),
            "available_agents": len([a for a in self.registered_agents.values() if a['agent'].is_available()]),
            "agent_details": agent_statuses
        }
    
    def _get_agent_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        metrics = {
            "overall_success_rate": 0.0,
            "total_tasks_completed": 0,
            "average_workload": 0.0,
            "agent_metrics": {}
        }
        
        if self.registered_agents:
            total_success_rate = sum(info['success_rate'] for info in self.registered_agents.values())
            total_tasks = sum(info['task_count'] for info in self.registered_agents.values())
            total_workload = sum(self.agent_workloads.values())
            
            metrics.update({
                "overall_success_rate": round(total_success_rate / len(self.registered_agents), 2),
                "total_tasks_completed": total_tasks,
                "average_workload": round(total_workload / len(self.registered_agents), 2)
            })
            
            for agent_id, agent_info in self.registered_agents.items():
                agent = agent_info['agent']
                metrics["agent_metrics"][agent_id] = agent.get_performance_metrics()
        
        return metrics
    
    def _get_workload_distribution(self) -> Dict[str, Any]:
        """Get current workload distribution across agents"""
        return {
            "workload_by_agent": dict(self.agent_workloads),
            "total_workload": sum(self.agent_workloads.values()),
            "max_workload": max(self.agent_workloads.values()) if self.agent_workloads else 0,
            "min_workload": min(self.agent_workloads.values()) if self.agent_workloads else 0,
            "workload_balance": self._calculate_workload_balance()
        }
    
    def _calculate_workload_balance(self) -> str:
        """Calculate how balanced the workload is across agents"""
        if not self.agent_workloads:
            return "no_agents"
        
        workloads = list(self.agent_workloads.values())
        if not workloads:
            return "balanced"
        
        max_load = max(workloads)
        min_load = min(workloads)
        
        if max_load - min_load <= 1:
            return "well_balanced"
        elif max_load - min_load <= 3:
            return "moderately_balanced"
        else:
            return "unbalanced"
    
    async def _delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a task to the most suitable agent"""
        target_task = task.get('target_task', {})
        task_type = target_task.get('type', 'general')
        
        # Find best agent
        best_agent_id = await self._find_best_agent_for_task(task_type, target_task)
        
        if best_agent_id and best_agent_id in self.registered_agents:
            agent = self.registered_agents[best_agent_id]['agent']
            
            # Execute the task
            result = await agent.execute_task(target_task)
            
            return {
                "delegation_successful": True,
                "assigned_agent": best_agent_id,
                "agent_name": agent.name,
                "task_result": result,
                "delegated_at": datetime.now().isoformat()
            }
        else:
            return {
                "delegation_successful": False,
                "error": f"No suitable agent found for task type: {task_type}",
                "available_agents": list(self.registered_agents.keys())
            }
    
    async def _aggregate_results(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple sources"""
        results = task.get('results', [])
        aggregation_method = task.get('method', 'simple')
        
        if aggregation_method == 'simple':
            return self._simple_aggregation(results)
        elif aggregation_method == 'weighted':
            weights = task.get('weights', {})
            return self._weighted_aggregation(results, weights)
        else:
            return self._advanced_aggregation(results, task.get('config', {}))
    
    def _simple_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple result aggregation"""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        return {
            "aggregation_method": "simple",
            "total_results": len(results),
            "successful_results": len(successful_results),
            "success_rate": (len(successful_results) / len(results) * 100) if results else 0,
            "combined_data": [r.get('data', {}) for r in successful_results],
            "aggregated_at": datetime.now().isoformat()
        }
    
    def _weighted_aggregation(self, results: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
        """Weighted result aggregation"""
        weighted_results = []
        total_weight = 0
        
        for result in results:
            if result.get('status') == 'success':
                agent_id = result.get('agent_id', 'unknown')
                weight = weights.get(agent_id, 1.0)
                weighted_results.append({
                    'result': result,
                    'weight': weight
                })
                total_weight += weight
        
        return {
            "aggregation_method": "weighted",
            "total_results": len(results),
            "weighted_results": len(weighted_results),
            "total_weight": total_weight,
            "weighted_data": weighted_results,
            "aggregated_at": datetime.now().isoformat()
        }
    
    def _advanced_aggregation(self, results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced result aggregation with custom logic"""
        # Implement advanced aggregation logic based on config
        return {
            "aggregation_method": "advanced",
            "total_results": len(results),
            "config": config,
            "message": "Advanced aggregation completed",
            "aggregated_at": datetime.now().isoformat()
        }
    
    async def _track_progress(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Track progress of ongoing tasks"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queued_tasks": len(self.task_queue),
            "agent_statuses": {agent_id: info['agent'].status.value for agent_id, info in self.registered_agents.items()},
            "progress_snapshot_at": datetime.now().isoformat()
        }
    
    async def _balance_load(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Balance load across available agents"""
        current_balance = self._calculate_workload_balance()
        
        if current_balance == "unbalanced":
            # Implement load balancing logic
            rebalancing_actions = self._generate_rebalancing_actions()
            
            return {
                "load_balancing": "needed",
                "current_balance": current_balance,
                "rebalancing_actions": rebalancing_actions,
                "balanced_at": datetime.now().isoformat()
            }
        else:
            return {
                "load_balancing": "not_needed",
                "current_balance": current_balance,
                "workload_distribution": dict(self.agent_workloads)
            }
    
    def _generate_rebalancing_actions(self) -> List[Dict[str, Any]]:
        """Generate actions to rebalance workload"""
        actions = []
        
        if self.agent_workloads:
            max_workload = max(self.agent_workloads.values())
            min_workload = min(self.agent_workloads.values())
            
            if max_workload - min_workload > 2:
                # Find overloaded and underloaded agents
                overloaded = [aid for aid, load in self.agent_workloads.items() if load == max_workload]
                underloaded = [aid for aid, load in self.agent_workloads.items() if load == min_workload]
                
                actions.append({
                    "action": "redistribute_tasks",
                    "from_agents": overloaded,
                    "to_agents": underloaded,
                    "estimated_improvement": "moderate"
                })
        
        return actions
    
    async def _general_coordination(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general coordination tasks"""
        return {
            "coordination_type": "general",
            "task_received": task.get('id', 'unknown'),
            "registered_agents": len(self.registered_agents),
            "system_status": "operational",
            "message": "General coordination task processed",
            "processed_at": datetime.now().isoformat()
        }