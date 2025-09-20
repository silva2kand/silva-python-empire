"""Chat Integration Service for AI Sherpa Multi-Agent System"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from .agent_manager import AgentManager, get_agent_manager
from .communication import (
    MessageType, MessagePriority, Message,
    get_message_bus, get_result_aggregator,
    send_agent_message, request_from_agent, aggregate_agent_results
)
from .base_agent import TaskPriority

class ChatAgentIntegration:
    """Integrates multi-agent system with chat interface"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agent_manager = get_agent_manager()
        self.message_bus = get_message_bus()
        self.result_aggregator = get_result_aggregator()
        
        # Chat processing strategies
        self.processing_strategies = {
            "single_agent": self._single_agent_processing,
            "multi_agent": self._multi_agent_processing,
            "coordinated": self._coordinated_processing,
            "research_focused": self._research_focused_processing,
            "code_focused": self._code_focused_processing
        }
        
        # Keywords for determining processing strategy
        self.strategy_keywords = {
            "research_focused": [
                "search", "find", "research", "lookup", "documentation", 
                "api", "library", "framework", "tutorial", "example",
                "how to", "what is", "explain", "learn", "guide"
            ],
            "code_focused": [
                "debug", "error", "bug", "fix", "optimize", "refactor", 
                "analyze", "review", "pattern", "performance", "security",
                "function", "class", "method", "variable", "syntax"
            ],
            "multi_agent": [
                "complex", "comprehensive", "detailed", "thorough", 
                "multiple", "various", "different approaches", "alternatives"
            ]
        }
    
    async def initialize(self):
        """Initialize the chat integration system"""
        try:
            # Initialize agent manager
            await self.agent_manager._initialize_system()
            
            # Register chat integration as a special agent
            await self.message_bus.register_agent(
                "chat_integration",
                {
                    "type": "chat_integration",
                    "capabilities": ["chat_processing", "strategy_selection", "result_synthesis"],
                    "description": "Integrates multi-agent system with chat interface"
                }
            )
            
            self.logger.info("Chat agent integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chat integration: {str(e)}")
            return False
    
    async def process_chat_message(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a chat message using the multi-agent system"""
        try:
            start_time = datetime.now()
            
            # Check if agent system is ready
            system_status = self.agent_manager.get_system_status()
            if system_status.get("status") != "ready":
                self.logger.warning("Agent system not ready, using fallback")
                return await self._fallback_processing(message, conversation_history)
            
            # Analyze the message to determine processing strategy
            strategy = self._determine_processing_strategy(message, conversation_history)
            
            self.logger.info(f"Processing chat message with strategy: {strategy}")
            
            # Process the message using the selected strategy
            if strategy in self.processing_strategies:
                processing_func = self.processing_strategies[strategy]
                result = await processing_func(message, conversation_history)
            else:
                # Fallback to single agent processing
                result = await self._single_agent_processing(message, conversation_history)
            
            # Add metadata to the result
            processing_time = (datetime.now() - start_time).total_seconds()
            result.update({
                "processing_strategy": strategy,
                "processing_time_seconds": processing_time,
                "agents_used": result.get("agents_used", []),
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing chat message: {str(e)}")
            # Try fallback processing on error
            try:
                return await self._fallback_processing(message, conversation_history)
            except:
                return {
                    "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                    "confidence": 0.1,
                    "error": str(e),
                    "processing_strategy": "error_fallback"
                }
    
    async def _fallback_processing(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Fallback processing using GPT4All service directly"""
        try:
            from shared.gpt4all_service import gpt4all_service
            
            # Use GPT4All service directly
            response = await asyncio.to_thread(
                gpt4all_service.generate_response, 
                message
            )
            
            return {
                "response": response,
                "confidence": 0.6,
                "agents_used": ["gpt4all_fallback"],
                "processing_strategy": "fallback_direct",
                "processing_details": {
                    "method": "direct_gpt4all",
                    "reason": "agent_system_unavailable"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fallback processing failed: {str(e)}")
            return {
                "response": "I'm having trouble processing your request right now. Please try again later.",
                "confidence": 0.1,
                "error": str(e),
                "processing_strategy": "emergency_fallback"
            }
    
    def _determine_processing_strategy(self, message: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Determine the best processing strategy for a message"""
        message_lower = message.lower()
        
        # Count keyword matches for each strategy
        strategy_scores = {}
        
        for strategy, keywords in self.strategy_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                strategy_scores[strategy] = score
        
        # Consider conversation history for context
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages
            for msg in recent_messages:
                content = msg.get('content', '').lower()
                for strategy, keywords in self.strategy_keywords.items():
                    additional_score = sum(0.5 for keyword in keywords if keyword in content)
                    strategy_scores[strategy] = strategy_scores.get(strategy, 0) + additional_score
        
        # Determine strategy based on scores
        if not strategy_scores:
            return "single_agent"  # Default strategy
        
        # Get the strategy with the highest score
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        # Apply some logic to prefer coordinated approach for complex queries
        if (strategy_scores.get("research_focused", 0) > 0 and 
            strategy_scores.get("code_focused", 0) > 0):
            return "coordinated"
        
        if strategy_scores.get("multi_agent", 0) > 2:
            return "multi_agent"
        
        return best_strategy
    
    async def _single_agent_processing(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process message using a single most appropriate agent"""
        try:
            # Determine which single agent to use
            agent_id = self._select_best_agent(message)
            
            # Prepare task for the agent
            task_data = {
                "type": "chat_query",
                "message": message,
                "conversation_history": conversation_history or [],
                "priority": TaskPriority.MEDIUM.value,
                "requires_response": True
            }
            
            # Send request to the selected agent
            response = await request_from_agent(
                "chat_integration", 
                agent_id, 
                task_data, 
                timeout=10.0
            )
            
            if response and response.get("success"):
                return {
                    "response": response.get("result", "No response generated"),
                    "confidence": response.get("confidence", 0.7),
                    "agents_used": [agent_id],
                    "processing_details": response.get("details", {})
                }
            else:
                return {
                    "response": "I couldn't process your request at the moment. Please try again.",
                    "confidence": 0.3,
                    "agents_used": [],
                    "error": "Agent processing failed"
                }
                
        except Exception as e:
            self.logger.error(f"Single agent processing error: {str(e)}")
            return {
                "response": "An error occurred while processing your request.",
                "confidence": 0.1,
                "agents_used": [],
                "error": str(e)
            }
    
    async def _multi_agent_processing(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process message using multiple agents in parallel"""
        try:
            # Select multiple agents for parallel processing
            agent_ids = self._select_multiple_agents(message)
            
            if not agent_ids:
                return await self._single_agent_processing(message, conversation_history)
            
            # Prepare task for all agents
            task_data = {
                "type": "chat_query",
                "message": message,
                "conversation_history": conversation_history or [],
                "priority": TaskPriority.MEDIUM.value,
                "requires_response": True
            }
            
            # Send requests to all selected agents in parallel
            tasks = []
            for agent_id in agent_ids:
                task = request_from_agent(
                    "chat_integration", 
                    agent_id, 
                    task_data, 
                    timeout=8.0
                )
                tasks.append(task)
            
            # Wait for all responses
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful responses
            successful_responses = []
            for i, response in enumerate(responses):
                if (not isinstance(response, Exception) and 
                    response and response.get("success")):
                    successful_responses.append(response)
            
            if not successful_responses:
                return {
                    "response": "I couldn't get responses from the agents. Please try again.",
                    "confidence": 0.2,
                    "agents_used": agent_ids,
                    "error": "No successful agent responses"
                }
            
            # Aggregate the results
            aggregated = await aggregate_agent_results(
                successful_responses, 
                strategy="consensus",
                config={"consensus_threshold": 0.4}
            )
            
            # Synthesize final response
            final_response = self._synthesize_multi_agent_response(successful_responses, aggregated)
            
            return {
                "response": final_response["response"],
                "confidence": final_response["confidence"],
                "agents_used": agent_ids,
                "processing_details": {
                    "successful_agents": len(successful_responses),
                    "aggregation_strategy": "consensus",
                    "individual_responses": len(successful_responses)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Multi-agent processing error: {str(e)}")
            return await self._single_agent_processing(message, conversation_history)
    
    async def _coordinated_processing(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process message using coordinated agent workflow"""
        try:
            # Use task coordinator to manage the workflow
            coordinator_id = "task_coordinator"
            
            # Prepare coordination task
            task_data = {
                "type": "coordinate_chat_query",
                "message": message,
                "conversation_history": conversation_history or [],
                "priority": TaskPriority.HIGH.value,
                "coordination_strategy": "sequential_with_synthesis",
                "requires_response": True
            }
            
            # Send request to task coordinator
            response = await request_from_agent(
                "chat_integration", 
                coordinator_id, 
                task_data, 
                timeout=15.0
            )
            
            if response and response.get("success"):
                return {
                    "response": response.get("result", "No coordinated response generated"),
                    "confidence": response.get("confidence", 0.8),
                    "agents_used": response.get("agents_involved", [coordinator_id]),
                    "processing_details": response.get("coordination_details", {})
                }
            else:
                # Fallback to multi-agent processing
                return await self._multi_agent_processing(message, conversation_history)
                
        except Exception as e:
            self.logger.error(f"Coordinated processing error: {str(e)}")
            return await self._multi_agent_processing(message, conversation_history)
    
    async def _research_focused_processing(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process message with focus on research capabilities"""
        try:
            # Use research agent primarily
            research_agent_id = "research_agent"
            
            # Prepare research task
            task_data = {
                "type": "research_query",
                "query": message,
                "conversation_history": conversation_history or [],
                "priority": TaskPriority.HIGH.value,
                "research_depth": "comprehensive",
                "requires_response": True
            }
            
            # Send request to research agent
            response = await request_from_agent(
                "chat_integration", 
                research_agent_id, 
                task_data, 
                timeout=12.0
            )
            
            if response and response.get("success"):
                return {
                    "response": response.get("result", "No research results found"),
                    "confidence": response.get("confidence", 0.7),
                    "agents_used": [research_agent_id],
                    "processing_details": {
                        "research_type": "focused",
                        "sources_consulted": response.get("sources", []),
                        "research_depth": "comprehensive"
                    }
                }
            else:
                # Fallback to single agent processing
                return await self._single_agent_processing(message, conversation_history)
                
        except Exception as e:
            self.logger.error(f"Research-focused processing error: {str(e)}")
            return await self._single_agent_processing(message, conversation_history)
    
    async def _code_focused_processing(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process message with focus on code analysis capabilities"""
        try:
            # Use code analysis agent primarily
            code_agent_id = "code_analysis_agent_001"
            
            # Prepare code analysis task
            task_data = {
                "type": "code_analysis_query",
                "query": message,
                "conversation_history": conversation_history or [],
                "priority": TaskPriority.HIGH.value,
                "analysis_depth": "comprehensive",
                "requires_response": True
            }
            
            # Send request to code analysis agent
            response = await request_from_agent(
                "chat_integration", 
                code_agent_id, 
                task_data, 
                timeout=12.0
            )
            
            if response and response.get("success"):
                return {
                    "response": response.get("result", "No code analysis results"),
                    "confidence": response.get("confidence", 0.8),
                    "agents_used": [code_agent_id],
                    "processing_details": {
                        "analysis_type": "focused",
                        "code_patterns_found": response.get("patterns", []),
                        "analysis_depth": "comprehensive"
                    }
                }
            else:
                # Fallback to single agent processing
                return await self._single_agent_processing(message, conversation_history)
                
        except Exception as e:
            self.logger.error(f"Code-focused processing error: {str(e)}")
            return await self._single_agent_processing(message, conversation_history)
    
    def _select_best_agent(self, message: str) -> str:
        """Select the best single agent for processing a message"""
        message_lower = message.lower()
        
        # Check for research-related keywords
        research_keywords = ["search", "find", "lookup", "documentation", "api", "how to", "what is"]
        if any(keyword in message_lower for keyword in research_keywords):
            return "research_agent_001"
        
        # Check for code-related keywords
        code_keywords = ["debug", "error", "code", "function", "class", "bug", "optimize"]
        if any(keyword in message_lower for keyword in code_keywords):
            return "code_analysis_agent_001"
        
        # Default to research agent for general queries
        return "research_agent_001"
    
    def _select_multiple_agents(self, message: str) -> List[str]:
        """Select multiple agents for parallel processing"""
        agents = []
        message_lower = message.lower()
        
        # Always include research agent for information gathering
        agents.append("research_agent_001")
        
        # Include code analysis agent if code-related
        code_keywords = ["code", "debug", "error", "function", "class", "programming"]
        if any(keyword in message_lower for keyword in code_keywords):
            agents.append("code_analysis_agent_001")
        
        # Include task coordinator for complex queries
        complex_keywords = ["complex", "comprehensive", "detailed", "multiple"]
        if any(keyword in message_lower for keyword in complex_keywords):
            agents.append("task_coordinator_001")
        
        return agents
    
    def _synthesize_multi_agent_response(self, responses: List[Dict[str, Any]], 
                                       aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize multiple agent responses into a coherent final response"""
        try:
            # Extract individual response texts
            response_texts = []
            total_confidence = 0
            
            for response in responses:
                if response.get("result"):
                    response_texts.append(response["result"])
                    total_confidence += response.get("confidence", 0.5)
            
            if not response_texts:
                return {
                    "response": "I couldn't generate a comprehensive response.",
                    "confidence": 0.2
                }
            
            # Simple synthesis: combine responses with clear attribution
            if len(response_texts) == 1:
                final_response = response_texts[0]
            else:
                # Create a structured response combining insights
                final_response = "Based on analysis from multiple specialized agents:\n\n"
                
                for i, text in enumerate(response_texts):
                    final_response += f"**Analysis {i+1}:** {text}\n\n"
                
                # Add a synthesis conclusion if aggregated data is available
                if aggregated.get("aggregated_result"):
                    final_response += "**Summary:** The agents generally agree on the key points and provide complementary insights."
            
            # Calculate average confidence
            avg_confidence = min(0.95, total_confidence / len(responses))
            
            return {
                "response": final_response,
                "confidence": avg_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Response synthesis error: {str(e)}")
            return {
                "response": "Multiple agents provided insights, but I had trouble synthesizing them.",
                "confidence": 0.4
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of the chat integration system"""
        try:
            agent_status = self.agent_manager.get_system_status()
            message_bus_metrics = self.message_bus.get_metrics()
            
            return {
                "chat_integration_status": "active",
                "agent_system_status": agent_status,
                "message_bus_metrics": message_bus_metrics,
                "available_strategies": list(self.processing_strategies.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {
                "chat_integration_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def shutdown(self):
        """Shutdown the chat integration system"""
        try:
            await self.message_bus.unregister_agent("chat_integration")
            await self.agent_manager.shutdown()
            self.logger.info("Chat integration system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

# Global instance
_chat_integration_instance = None

def get_chat_integration() -> ChatAgentIntegration:
    """Get the global chat integration instance"""
    global _chat_integration_instance
    if _chat_integration_instance is None:
        _chat_integration_instance = ChatAgentIntegration()
    return _chat_integration_instance

# Convenience function for external use
async def process_chat_with_agents(message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Process a chat message using the multi-agent system"""
    chat_integration = get_chat_integration()
    
    # Initialize if not already done
    if not hasattr(chat_integration, '_initialized'):
        await chat_integration.initialize()
        chat_integration._initialized = True
    
    return await chat_integration.process_chat_message(message, conversation_history)