"""Inter-Agent Communication System for AI Sherpa Multi-Agent System"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    RESULT_SHARE = "result_share"

class MessagePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class DeliveryStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class Message:
    """Represents a message in the inter-agent communication system"""
    id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    requires_response: bool = False
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        # Convert enums to strings
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['delivery_status'] = self.delivery_status.value
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        if self.delivered_at:
            data['delivered_at'] = self.delivered_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        # Convert string enums back to enum objects
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        data['delivery_status'] = DeliveryStatus(data['delivery_status'])
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        if data.get('delivered_at'):
            data['delivered_at'] = datetime.fromisoformat(data['delivered_at'])
        if data.get('acknowledged_at'):
            data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        return cls(**data)

class MessageBus:
    """Central message bus for inter-agent communication"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_queue_size = max_queue_size
        
        # Message storage and routing
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_queue_size))
        self.message_history: deque = deque(maxlen=max_queue_size * 2)
        self.pending_responses: Dict[str, Message] = {}
        
        # Agent registry and subscriptions
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, Dict[MessageType, Callable]] = defaultdict(dict)
        self.broadcast_subscriptions: Dict[str, List[str]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            "total_messages_sent": 0,
            "total_messages_delivered": 0,
            "total_messages_failed": 0,
            "average_delivery_time": 0.0,
            "active_conversations": 0,
            "queue_sizes": {},
            "last_updated": datetime.now()
        }
        
        # Background tasks
        self._cleanup_task = None
        self._heartbeat_task = None
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
    
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> bool:
        """Register an agent with the message bus"""
        try:
            self.registered_agents[agent_id] = {
                **agent_info,
                "registered_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "message_count": 0,
                "status": "active"
            }
            
            # Initialize message queue for the agent
            if agent_id not in self.message_queues:
                self.message_queues[agent_id] = deque(maxlen=self.max_queue_size)
            
            self.logger.info(f"Agent {agent_id} registered with message bus")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {str(e)}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the message bus"""
        try:
            if agent_id in self.registered_agents:
                # Mark agent as inactive
                self.registered_agents[agent_id]["status"] = "inactive"
                self.registered_agents[agent_id]["unregistered_at"] = datetime.now()
                
                # Clean up subscriptions
                for topic in list(self.broadcast_subscriptions.keys()):
                    if agent_id in self.broadcast_subscriptions[topic]:
                        self.broadcast_subscriptions[topic].remove(agent_id)
                
                self.logger.info(f"Agent {agent_id} unregistered from message bus")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            return False
    
    def register_message_handler(self, agent_id: str, message_type: MessageType, handler: Callable):
        """Register a message handler for an agent"""
        self.message_handlers[agent_id][message_type] = handler
        self.logger.debug(f"Registered {message_type.value} handler for agent {agent_id}")
    
    def subscribe_to_broadcasts(self, agent_id: str, topics: List[str]):
        """Subscribe an agent to broadcast topics"""
        for topic in topics:
            if agent_id not in self.broadcast_subscriptions[topic]:
                self.broadcast_subscriptions[topic].append(agent_id)
        self.logger.debug(f"Agent {agent_id} subscribed to topics: {topics}")
    
    async def send_message(self, message: Message) -> bool:
        """Send a message through the message bus"""
        try:
            # Validate message
            if not self._validate_message(message):
                return False
            
            # Check if recipient is registered and active
            if message.recipient_id not in self.registered_agents:
                self.logger.warning(f"Recipient {message.recipient_id} not registered")
                message.delivery_status = DeliveryStatus.FAILED
                return False
            
            if self.registered_agents[message.recipient_id]["status"] != "active":
                self.logger.warning(f"Recipient {message.recipient_id} is not active")
                message.delivery_status = DeliveryStatus.FAILED
                return False
            
            # Add message to recipient's queue
            recipient_queue = self.message_queues[message.recipient_id]
            
            # Check queue capacity
            if len(recipient_queue) >= self.max_queue_size:
                self.logger.warning(f"Message queue full for agent {message.recipient_id}")
                # Remove oldest low-priority message if possible
                self._make_queue_space(recipient_queue)
            
            # Insert message based on priority
            self._insert_message_by_priority(recipient_queue, message)
            
            # Update message status
            message.delivery_status = DeliveryStatus.DELIVERED
            message.delivered_at = datetime.now()
            
            # Add to history
            self.message_history.append(message)
            
            # Update metrics
            self.metrics["total_messages_sent"] += 1
            self.metrics["total_messages_delivered"] += 1
            self.registered_agents[message.sender_id]["message_count"] += 1
            
            # If message requires response, track it
            if message.requires_response:
                self.pending_responses[message.id] = message
            
            # Notify recipient if handler is registered
            await self._notify_message_handler(message)
            
            self.logger.debug(f"Message {message.id} delivered to {message.recipient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message {message.id}: {str(e)}")
            message.delivery_status = DeliveryStatus.FAILED
            self.metrics["total_messages_failed"] += 1
            return False
    
    async def send_broadcast(self, sender_id: str, topic: str, content: Dict[str, Any], 
                           priority: MessagePriority = MessagePriority.MEDIUM) -> List[str]:
        """Send a broadcast message to all subscribers of a topic"""
        delivered_to = []
        
        if topic in self.broadcast_subscriptions:
            subscribers = self.broadcast_subscriptions[topic]
            
            for subscriber_id in subscribers:
                if subscriber_id != sender_id:  # Don't send to sender
                    message = Message(
                        id=str(uuid.uuid4()),
                        sender_id=sender_id,
                        recipient_id=subscriber_id,
                        message_type=MessageType.BROADCAST,
                        priority=priority,
                        content={"topic": topic, **content},
                        created_at=datetime.now()
                    )
                    
                    if await self.send_message(message):
                        delivered_to.append(subscriber_id)
        
        self.logger.info(f"Broadcast from {sender_id} on topic '{topic}' delivered to {len(delivered_to)} agents")
        return delivered_to
    
    async def receive_messages(self, agent_id: str, max_messages: int = 10) -> List[Message]:
        """Receive messages for an agent"""
        if agent_id not in self.message_queues:
            return []
        
        messages = []
        queue = self.message_queues[agent_id]
        
        # Get up to max_messages from the queue
        for _ in range(min(max_messages, len(queue))):
            if queue:
                message = queue.popleft()
                messages.append(message)
                
                # Mark as acknowledged
                message.acknowledged_at = datetime.now()
                message.delivery_status = DeliveryStatus.ACKNOWLEDGED
        
        # Update heartbeat
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id]["last_heartbeat"] = datetime.now()
        
        return messages
    
    async def send_response(self, original_message: Message, response_content: Dict[str, Any]) -> bool:
        """Send a response to a message that requires response"""
        response_message = Message(
            id=str(uuid.uuid4()),
            sender_id=original_message.recipient_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=original_message.priority,
            content=response_content,
            created_at=datetime.now(),
            correlation_id=original_message.id
        )
        
        success = await self.send_message(response_message)
        
        # Remove from pending responses if successful
        if success and original_message.id in self.pending_responses:
            del self.pending_responses[original_message.id]
        
        return success
    
    async def request_response(self, sender_id: str, recipient_id: str, 
                             content: Dict[str, Any], timeout: float = 30.0,
                             priority: MessagePriority = MessagePriority.MEDIUM) -> Optional[Dict[str, Any]]:
        """Send a request and wait for response"""
        request_message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            priority=priority,
            content=content,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=timeout),
            requires_response=True
        )
        
        # Send the request
        if not await self.send_message(request_message):
            return None
        
        # Wait for response
        try:
            response = await self._wait_for_response(request_message.id, timeout)
            return response.content if response else None
        except asyncio.TimeoutError:
            self.logger.warning(f"Request {request_message.id} timed out")
            # Clean up pending response
            if request_message.id in self.pending_responses:
                del self.pending_responses[request_message.id]
            return None
    
    async def _wait_for_response(self, request_id: str, timeout: float) -> Optional[Message]:
        """Wait for a response to a specific request"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            # Check message history for response
            for message in reversed(self.message_history):
                if (message.message_type == MessageType.RESPONSE and 
                    message.correlation_id == request_id):
                    return message
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError()
    
    def _validate_message(self, message: Message) -> bool:
        """Validate a message before sending"""
        if not message.sender_id or not message.recipient_id:
            self.logger.error("Message missing sender or recipient ID")
            return False
        
        if message.sender_id not in self.registered_agents:
            self.logger.error(f"Sender {message.sender_id} not registered")
            return False
        
        if message.expires_at and message.expires_at <= datetime.now():
            self.logger.warning(f"Message {message.id} already expired")
            return False
        
        return True
    
    def _make_queue_space(self, queue: deque):
        """Make space in a full queue by removing low-priority messages"""
        # Find and remove the oldest low-priority message
        for i, message in enumerate(queue):
            if message.priority == MessagePriority.LOW:
                del queue[i]
                self.logger.debug("Removed low-priority message to make queue space")
                return
        
        # If no low-priority messages, remove oldest medium priority
        for i, message in enumerate(queue):
            if message.priority == MessagePriority.MEDIUM:
                del queue[i]
                self.logger.debug("Removed medium-priority message to make queue space")
                return
        
        # Last resort: remove oldest message
        if queue:
            queue.popleft()
            self.logger.warning("Removed oldest message to make queue space")
    
    def _insert_message_by_priority(self, queue: deque, message: Message):
        """Insert message into queue based on priority"""
        # For simplicity, just append (could implement proper priority queue)
        queue.append(message)
    
    async def _notify_message_handler(self, message: Message):
        """Notify registered message handler about new message"""
        recipient_id = message.recipient_id
        message_type = message.message_type
        
        if (recipient_id in self.message_handlers and 
            message_type in self.message_handlers[recipient_id]):
            
            handler = self.message_handlers[recipient_id][message_type]
            try:
                # Call handler asynchronously if it's a coroutine
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.logger.error(f"Message handler error for {recipient_id}: {str(e)}")
    
    async def _cleanup_expired_messages(self):
        """Background task to clean up expired messages"""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up expired messages from queues
                for agent_id, queue in self.message_queues.items():
                    expired_messages = []
                    for message in queue:
                        if message.expires_at and message.expires_at <= current_time:
                            expired_messages.append(message)
                    
                    for expired_message in expired_messages:
                        queue.remove(expired_message)
                        expired_message.delivery_status = DeliveryStatus.EXPIRED
                        self.logger.debug(f"Expired message {expired_message.id} removed from queue")
                
                # Clean up expired pending responses
                expired_responses = []
                for request_id, message in self.pending_responses.items():
                    if message.expires_at and message.expires_at <= current_time:
                        expired_responses.append(request_id)
                
                for request_id in expired_responses:
                    del self.pending_responses[request_id]
                
                # Update metrics
                self.metrics["queue_sizes"] = {agent_id: len(queue) for agent_id, queue in self.message_queues.items()}
                self.metrics["last_updated"] = current_time
                
                # Sleep for 30 seconds before next cleanup
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(30)
    
    async def _heartbeat_monitor(self):
        """Background task to monitor agent heartbeats"""
        while True:
            try:
                current_time = datetime.now()
                heartbeat_timeout = timedelta(minutes=5)
                
                for agent_id, agent_info in self.registered_agents.items():
                    if agent_info["status"] == "active":
                        last_heartbeat = agent_info["last_heartbeat"]
                        if current_time - last_heartbeat > heartbeat_timeout:
                            # Mark agent as inactive
                            agent_info["status"] = "inactive"
                            self.logger.warning(f"Agent {agent_id} marked inactive due to heartbeat timeout")
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get communication system metrics"""
        return {
            **self.metrics,
            "registered_agents": len(self.registered_agents),
            "active_agents": len([a for a in self.registered_agents.values() if a["status"] == "active"]),
            "total_queued_messages": sum(len(queue) for queue in self.message_queues.values()),
            "pending_responses": len(self.pending_responses),
            "broadcast_topics": len(self.broadcast_subscriptions)
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        if agent_id in self.registered_agents:
            agent_info = self.registered_agents[agent_id].copy()
            agent_info["queue_size"] = len(self.message_queues.get(agent_id, []))
            agent_info["subscribed_topics"] = [
                topic for topic, subscribers in self.broadcast_subscriptions.items()
                if agent_id in subscribers
            ]
            return agent_info
        return None
    
    async def shutdown(self):
        """Shutdown the message bus"""
        self.logger.info("Shutting down message bus...")
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # Mark all agents as inactive
        for agent_info in self.registered_agents.values():
            agent_info["status"] = "inactive"
        
        self.logger.info("Message bus shutdown complete")

class ResultAggregator:
    """Aggregates results from multiple agents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.aggregation_strategies = {
            "consensus": self._consensus_aggregation,
            "weighted": self._weighted_aggregation,
            "priority": self._priority_aggregation,
            "merge": self._merge_aggregation,
            "best_result": self._best_result_aggregation
        }
    
    async def aggregate_results(self, results: List[Dict[str, Any]], 
                              strategy: str = "consensus",
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate multiple agent results using specified strategy"""
        if not results:
            return {"error": "No results to aggregate"}
        
        if len(results) == 1:
            return {"aggregated_result": results[0], "strategy": "single_result"}
        
        config = config or {}
        
        if strategy in self.aggregation_strategies:
            aggregation_func = self.aggregation_strategies[strategy]
            return await aggregation_func(results, config)
        else:
            self.logger.warning(f"Unknown aggregation strategy: {strategy}")
            return await self._consensus_aggregation(results, config)
    
    async def _consensus_aggregation(self, results: List[Dict[str, Any]], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results based on consensus"""
        # Simple consensus: find common elements
        consensus_data = {}
        result_count = len(results)
        
        # Count occurrences of each key-value pair
        key_counts = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool)):
                        key_counts[key][str(value)] += 1
        
        # Include items that appear in majority of results
        threshold = config.get("consensus_threshold", 0.5)
        min_count = max(1, int(result_count * threshold))
        
        for key, value_counts in key_counts.items():
            for value, count in value_counts.items():
                if count >= min_count:
                    consensus_data[key] = value
                    break  # Take the first value that meets threshold
        
        return {
            "aggregated_result": consensus_data,
            "strategy": "consensus",
            "total_results": result_count,
            "consensus_threshold": threshold,
            "confidence": len(consensus_data) / max(1, len(key_counts))
        }
    
    async def _weighted_aggregation(self, results: List[Dict[str, Any]], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results using weighted average/combination"""
        weights = config.get("weights", [1.0] * len(results))
        
        if len(weights) != len(results):
            weights = [1.0] * len(results)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        weighted_result = {}
        
        # Combine numerical values using weighted average
        for i, result in enumerate(results):
            weight = weights[i]
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in weighted_result:
                            weighted_result[key] = 0
                        weighted_result[key] += value * weight
                    elif isinstance(value, str) and weight == max(weights):
                        # For strings, use the value from highest weighted result
                        weighted_result[key] = value
        
        return {
            "aggregated_result": weighted_result,
            "strategy": "weighted",
            "weights_used": weights,
            "total_results": len(results)
        }
    
    async def _priority_aggregation(self, results: List[Dict[str, Any]], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results based on priority order"""
        priorities = config.get("priorities", list(range(len(results))))
        
        # Sort results by priority (lower number = higher priority)
        sorted_results = sorted(zip(results, priorities), key=lambda x: x[1])
        
        # Start with highest priority result and merge others
        aggregated = sorted_results[0][0].copy() if sorted_results else {}
        
        for result, priority in sorted_results[1:]:
            if isinstance(result, dict):
                # Add keys that don't exist in aggregated result
                for key, value in result.items():
                    if key not in aggregated:
                        aggregated[key] = value
        
        return {
            "aggregated_result": aggregated,
            "strategy": "priority",
            "priority_order": [p for _, p in sorted_results],
            "total_results": len(results)
        }
    
    async def _merge_aggregation(self, results: List[Dict[str, Any]], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all results into a comprehensive result"""
        merged_result = {}
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in merged_result:
                        merged_result[key] = []
                    
                    # Store all values for each key
                    if isinstance(merged_result[key], list):
                        merged_result[key].append({"source": i, "value": value})
                    else:
                        # Convert to list if not already
                        old_value = merged_result[key]
                        merged_result[key] = [{"source": "previous", "value": old_value}, 
                                            {"source": i, "value": value}]
        
        return {
            "aggregated_result": merged_result,
            "strategy": "merge",
            "total_results": len(results),
            "note": "All values preserved with source information"
        }
    
    async def _best_result_aggregation(self, results: List[Dict[str, Any]], 
                                     config: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best result based on scoring criteria"""
        scoring_criteria = config.get("scoring_criteria", {})
        
        best_result = None
        best_score = float('-inf')
        best_index = -1
        
        for i, result in enumerate(results):
            score = self._calculate_result_score(result, scoring_criteria)
            if score > best_score:
                best_score = score
                best_result = result
                best_index = i
        
        return {
            "aggregated_result": best_result,
            "strategy": "best_result",
            "best_score": best_score,
            "best_result_index": best_index,
            "total_results": len(results)
        }
    
    def _calculate_result_score(self, result: Dict[str, Any], 
                              criteria: Dict[str, Any]) -> float:
        """Calculate a score for a result based on criteria"""
        score = 0.0
        
        # Default scoring: prefer results with more data
        if isinstance(result, dict):
            score += len(result) * 0.1
            
            # Check for specific quality indicators
            if "confidence" in result:
                score += float(result.get("confidence", 0)) * 0.5
            
            if "success" in result and result["success"]:
                score += 1.0
            
            # Apply custom criteria
            for criterion, weight in criteria.items():
                if criterion in result:
                    value = result[criterion]
                    if isinstance(value, (int, float)):
                        score += value * weight
                    elif isinstance(value, bool) and value:
                        score += weight
        
        return score

# Global instances
_message_bus_instance = None
_result_aggregator_instance = None

def get_message_bus() -> MessageBus:
    """Get the global message bus instance"""
    global _message_bus_instance
    if _message_bus_instance is None:
        _message_bus_instance = MessageBus()
    return _message_bus_instance

def get_result_aggregator() -> ResultAggregator:
    """Get the global result aggregator instance"""
    global _result_aggregator_instance
    if _result_aggregator_instance is None:
        _result_aggregator_instance = ResultAggregator()
    return _result_aggregator_instance

# Convenience functions
async def send_agent_message(sender_id: str, recipient_id: str, content: Dict[str, Any], 
                           message_type: MessageType = MessageType.REQUEST,
                           priority: MessagePriority = MessagePriority.MEDIUM) -> bool:
    """Send a message between agents"""
    message = Message(
        id=str(uuid.uuid4()),
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        priority=priority,
        content=content,
        created_at=datetime.now()
    )
    
    message_bus = get_message_bus()
    return await message_bus.send_message(message)

async def broadcast_to_agents(sender_id: str, topic: str, content: Dict[str, Any],
                            priority: MessagePriority = MessagePriority.MEDIUM) -> List[str]:
    """Broadcast a message to all agents subscribed to a topic"""
    message_bus = get_message_bus()
    return await message_bus.send_broadcast(sender_id, topic, content, priority)

async def request_from_agent(sender_id: str, recipient_id: str, content: Dict[str, Any],
                           timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Send a request to an agent and wait for response"""
    message_bus = get_message_bus()
    return await message_bus.request_response(sender_id, recipient_id, content, timeout)

async def aggregate_agent_results(results: List[Dict[str, Any]], strategy: str = "consensus",
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Aggregate results from multiple agents"""
    aggregator = get_result_aggregator()
    return await aggregator.aggregate_results(results, strategy, config)