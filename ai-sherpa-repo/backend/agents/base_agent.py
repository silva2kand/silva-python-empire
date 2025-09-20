"""Base Agent Class for AI Sherpa Multi-Agent System"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    COMPLETED = "completed"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class BaseAgent(ABC):
    """Base class for all AI agents in the system"""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.current_task = None
        self.task_history = []
        self.logger = logging.getLogger(f"agent.{name}")
        self.created_at = datetime.now()
        self._heartbeat_task = None
        self._message_bus = None
        
    def set_message_bus(self, message_bus):
        """Set the message bus for communication"""
        self._message_bus = message_bus
        # Temporarily disable heartbeat to focus on core functionality
        # if self._heartbeat_task is None:
        #     self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
    
    async def _send_heartbeats(self):
        """Send periodic heartbeat messages to maintain agent registration"""
        while True:
            try:
                if self._message_bus:
                    from .communication import Message, MessageType, MessagePriority
                    heartbeat_message = Message(
                        id=f"heartbeat_{self.agent_id}_{datetime.now().timestamp()}",
                        sender_id=self.agent_id,
                        recipient_id="system",
                        message_type=MessageType.HEARTBEAT,
                        priority=MessagePriority.LOW,
                        content={"status": self.status.value, "timestamp": datetime.now().isoformat()},
                        created_at=datetime.now()
                    )
                    await self._message_bus.send_message(heartbeat_message)
                await asyncio.sleep(60)  # Send heartbeat every minute
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {str(e)}")
                await asyncio.sleep(60)
        
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task assigned to this agent"""
        pass
    
    @abstractmethod
    def can_handle_task(self, task_type: str) -> bool:
        """Check if this agent can handle the given task type"""
        pass
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with proper status management and error handling"""
        task_id = task.get('id', f"task_{datetime.now().timestamp()}")
        
        try:
            self.status = AgentStatus.WORKING
            self.current_task = task
            self.logger.info(f"Starting task {task_id}: {task.get('description', 'No description')}")
            
            # Process the task
            result = await self.process_task(task)
            
            # Add metadata to result
            result.update({
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'task_id': task_id,
                'completed_at': datetime.now().isoformat(),
                'status': 'success'
            })
            
            self.status = AgentStatus.COMPLETED
            self.task_history.append({
                'task_id': task_id,
                'task': task,
                'result': result,
                'completed_at': datetime.now(),
                'status': 'success'
            })
            
            self.logger.info(f"Completed task {task_id} successfully")
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            error_result = {
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'task_id': task_id,
                'error': str(e),
                'completed_at': datetime.now().isoformat(),
                'status': 'error'
            }
            
            self.task_history.append({
                'task_id': task_id,
                'task': task,
                'result': error_result,
                'completed_at': datetime.now(),
                'status': 'error'
            })
            
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            return error_result
            
        finally:
            self.current_task = None
            if self.status == AgentStatus.WORKING:
                self.status = AgentStatus.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and information"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'current_task': self.current_task,
            'tasks_completed': len([t for t in self.task_history if t['status'] == 'success']),
            'tasks_failed': len([t for t in self.task_history if t['status'] == 'error']),
            'created_at': self.created_at.isoformat()
        }
    
    def get_task_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get agent's task history"""
        history = self.task_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.status == AgentStatus.IDLE
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_tasks = len(self.task_history)
        successful_tasks = len([t for t in self.task_history if t['status'] == 'success'])
        failed_tasks = len([t for t in self.task_history if t['status'] == 'error'])
        
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': round(success_rate, 2),
            'average_task_time': self._calculate_average_task_time()
        }
    
    def _calculate_average_task_time(self) -> float:
        """Calculate average task completion time in seconds"""
        if not self.task_history:
            return 0.0
        
        # This is a simplified calculation - in a real implementation,
        # you'd track start and end times for each task
        return 2.5  # Mock average time