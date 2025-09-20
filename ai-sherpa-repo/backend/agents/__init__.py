"""AI Sherpa Multi-Agent System

This module provides a comprehensive multi-agent system for AI Sherpa,
including specialized agents for research, code analysis, and task coordination,
plus communication infrastructure and chat integration.
"""

from .base_agent import BaseAgent, AgentStatus, TaskPriority
from .research_agent import ResearchAgent
from .code_analysis_agent import CodeAnalysisAgent
from .task_coordinator_agent import TaskCoordinatorAgent
from .agent_manager import AgentManager, get_agent_manager
from .communication import (
    MessageBus, ResultAggregator, Message, MessageType, MessagePriority,
    get_message_bus, get_result_aggregator, send_agent_message, 
    broadcast_to_agents, request_from_agent, aggregate_agent_results
)
from .chat_integration import ChatAgentIntegration, get_chat_integration, process_chat_with_agents

__all__ = [
    'BaseAgent',
    'AgentStatus', 
    'TaskPriority',
    'ResearchAgent',
    'CodeAnalysisAgent', 
    'TaskCoordinatorAgent',
    'AgentManager',
    'get_agent_manager',
    'MessageBus',
    'ResultAggregator',
    'Message',
    'MessageType',
    'MessagePriority',
    'get_message_bus',
    'get_result_aggregator',
    'send_agent_message',
    'broadcast_to_agents',
    'request_from_agent',
    'aggregate_agent_results',
    'ChatAgentIntegration',
    'get_chat_integration',
    'process_chat_with_agents'
]