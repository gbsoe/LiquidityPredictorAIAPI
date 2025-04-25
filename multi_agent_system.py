"""
Multi-Agent System for Self-Evolving Prediction Engine

This module implements a multi-agent system architecture that enables collective intelligence
for the prediction engine. Each agent specializes in a different aspect of prediction,
and they work together to improve the overall system performance.

Key features:
1. Collaborative filtering among specialized agents
2. Agent competition and selection (evolutionary algorithms)
3. Consensus mechanisms for aggregating predictions
4. Knowledge sharing and transfer learning between agents
5. Meta-learning for automatic agent adaptation
"""

import os
import json
import logging
import random
import time
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Callable, Optional, Set, Union
from dataclasses import dataclass, field
from copy import deepcopy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_agent_system")

# Constants
AGENT_HISTORY_SIZE = 100
MAX_KNOWLEDGE_SIZE = 1000
MAX_AGENTS = 25
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.7
MIN_FITNESS_THRESHOLD = 0.2
CONSENSUS_THRESHOLD = 0.67  # 2/3 majority

@dataclass
class PredictionTask:
    """Represents a prediction task"""
    task_id: str
    input_data: Dict[str, Any]
    target_variable: str
    timestamp: datetime = field(default_factory=datetime.now)
    true_outcome: Any = None
    predictions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "input_data": self.input_data,
            "target_variable": self.target_variable,
            "timestamp": self.timestamp.isoformat(),
            "true_outcome": self.true_outcome,
            "predictions": self.predictions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionTask':
        """Create from dictionary"""
        obj = cls(
            task_id=data["task_id"],
            input_data=data["input_data"],
            target_variable=data["target_variable"]
        )
        
        obj.timestamp = datetime.fromisoformat(data["timestamp"])
        obj.true_outcome = data.get("true_outcome")
        obj.predictions = data.get("predictions", {})
        obj.metadata = data.get("metadata", {})
        
        return obj

@dataclass
class Skill:
    """Represents a specific prediction skill"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    skill_type: str = "basic"  # basic, specialized, meta
    target_variables: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "skill_type": self.skill_type,
            "target_variables": self.target_variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            parameters=data.get("parameters", {}),
            skill_type=data.get("skill_type", "basic"),
            target_variables=data.get("target_variables", [])
        )
    
    def mutate(self, mutation_rate: float = DEFAULT_MUTATION_RATE) -> 'Skill':
        """
        Create a mutated copy of this skill
        
        Args:
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated skill
        """
        mutated = Skill(
            name=self.name,
            skill_type=self.skill_type,
            target_variables=self.target_variables.copy()
        )
        
        # Copy parameters with potential mutations
        mutated_params = {}
        for key, value in self.parameters.items():
            if random.random() < mutation_rate:
                # Mutate the parameter
                if isinstance(value, float):
                    # Perturb float value
                    mutated_params[key] = value * (0.8 + 0.4 * random.random())
                elif isinstance(value, int):
                    # Perturb integer value
                    mutated_params[key] = max(1, value + random.randint(-2, 2))
                elif isinstance(value, bool):
                    # Flip boolean
                    mutated_params[key] = not value
                elif isinstance(value, str) and len(self.parameters.get("options", [])) > 0:
                    # Choose different option if available
                    options = self.parameters.get("options", [])
                    mutated_params[key] = random.choice(options)
                else:
                    # No mutation for this type
                    mutated_params[key] = value
            else:
                # No mutation
                mutated_params[key] = value
                
        mutated.parameters = mutated_params
        return mutated

@dataclass
class AgentPerformance:
    """Tracks agent performance metrics"""
    successful_predictions: int = 0
    total_predictions: int = 0
    reward_collected: float = 0.0
    avg_prediction_error: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions
    
    def update(self, success: bool, reward: float, error: float) -> None:
        """
        Update performance metrics
        
        Args:
            success: Whether prediction was successful
            reward: Reward value
            error: Prediction error
        """
        self.total_predictions += 1
        if success:
            self.successful_predictions += 1
            
        self.reward_collected += reward
        
        # Update average error using moving average
        if self.total_predictions == 1:
            self.avg_prediction_error = error
        else:
            self.avg_prediction_error = (self.avg_prediction_error * (self.total_predictions - 1) + error) / self.total_predictions
            
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "successful_predictions": self.successful_predictions,
            "total_predictions": self.total_predictions,
            "reward_collected": self.reward_collected,
            "avg_prediction_error": self.avg_prediction_error,
            "success_rate": self.success_rate,
            "last_update": self.last_update.isoformat()
        }

@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge learned by an agent"""
    knowledge_id: str
    content: Any
    knowledge_type: str  # pattern, rule, heuristic, etc.
    confidence: float
    source: str  # agent_id or "external"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "knowledge_id": self.knowledge_id,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create from dictionary"""
        obj = cls(
            knowledge_id=data["knowledge_id"],
            content=data["content"],
            knowledge_type=data["knowledge_type"],
            confidence=data["confidence"],
            source=data["source"]
        )
        
        obj.timestamp = datetime.fromisoformat(data["timestamp"])
        obj.metadata = data.get("metadata", {})
        
        return obj

class Agent:
    """
    Autonomous agent that can make predictions and learn from experience
    """
    
    def __init__(self, agent_id: str, agent_type: str, skills: List[Skill] = None):
        """
        Initialize an agent
        
        Args:
            agent_id: Unique identifier
            agent_type: Type of agent (e.g., "apr_predictor", "tvl_predictor", etc.)
            skills: List of agent skills
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.skills = skills or []
        self.knowledge_base = []
        self.task_history = []
        self.performance = AgentPerformance()
        self.created_at = datetime.now()
        self.last_active = self.created_at
        self.specializations = set()
        self.collaborators = set()
        self.metadata = {}
        
    def predict(self, task: PredictionTask) -> Any:
        """
        Make a prediction for a task
        
        Args:
            task: Prediction task
            
        Returns:
            Prediction result
        """
        # This is a base implementation that would be overridden by specific agent types
        # For demonstration, we'll use a placeholder implementation
        
        self.last_active = datetime.now()
        
        # Add task to history
        self.task_history.append(task)
        if len(self.task_history) > AGENT_HISTORY_SIZE:
            self.task_history = self.task_history[-AGENT_HISTORY_SIZE:]
        
        # Calculate simple prediction based on skills
        prediction = None
        confidence = 0.0
        
        # Apply skills to make prediction
        if task.target_variable == "apr_change":
            # Simple APR prediction
            base_change = 0.0
            
            # Use skills to adjust prediction
            for skill in self.skills:
                if skill.name == "trend_analysis":
                    historical_trend = task.input_data.get('apr_change_7d', 0)
                    trend_weight = skill.parameters.get('trend_weight', 0.5)
                    base_change += historical_trend * trend_weight
                elif skill.name == "volume_impact":
                    volume_ratio = task.input_data.get('volume_liquidity_ratio', 0)
                    volume_factor = skill.parameters.get('volume_factor', 2.0)
                    base_change += volume_ratio * volume_factor
                elif skill.name == "category_bias":
                    category = task.input_data.get('category', 'Other')
                    category_factors = skill.parameters.get('category_factors', {})
                    base_change += category_factors.get(category, 0)
            
            # Add some randomness to simulation learning
            base_change += random.uniform(-1, 1)
            
            prediction = base_change
            confidence = 0.5 + (min(len(self.skills), 5) / 10)  # More skills = higher confidence
            
        elif task.target_variable == "tvl_change":
            # Simple TVL prediction
            base_change = 0.0
            
            # Use skills to adjust prediction
            for skill in self.skills:
                if skill.name == "tvl_momentum":
                    historical_trend = task.input_data.get('tvl_change_7d', 0)
                    momentum_factor = skill.parameters.get('momentum_factor', 0.7)
                    base_change += historical_trend * momentum_factor
                elif skill.name == "apr_correlation":
                    apr_change = task.input_data.get('apr_change_7d', 0)
                    correlation_strength = skill.parameters.get('correlation_strength', 0.3)
                    base_change += apr_change * correlation_strength
            
            # Add some randomness to simulation learning
            base_change += random.uniform(-2, 2)
            
            prediction = base_change
            confidence = 0.4 + (min(len(self.skills), 5) / 10)  # More skills = higher confidence
            
        else:
            # Unknown target variable
            prediction = 0.0
            confidence = 0.1
        
        # Store prediction in task
        task.predictions[self.agent_id] = {
            "value": prediction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    
    def learn(self, task: PredictionTask, reward: float = 0.0) -> None:
        """
        Learn from a completed task
        
        Args:
            task: Completed prediction task with true outcome
            reward: Reward value for reinforcement learning
        """
        if task.true_outcome is None:
            # Can't learn without true outcome
            return
            
        # Get our prediction for this task
        our_prediction = task.predictions.get(self.agent_id, {})
        prediction_value = our_prediction.get("value")
        
        if prediction_value is not None:
            # Calculate prediction error
            error = abs(prediction_value - task.true_outcome)
            
            # Update performance
            success = error < 1.0  # Arbitrary threshold for success
            self.performance.update(success, reward, error)
            
            # Learn from error - adjust skill parameters
            self._adapt_skills(task, error)
            
            # Extract and store knowledge
            self._extract_knowledge(task)
    
    def _adapt_skills(self, task: PredictionTask, error: float) -> None:
        """
        Adapt skills based on prediction error
        
        Args:
            task: Prediction task
            error: Prediction error
        """
        # Simple skill adaptation logic
        for skill in self.skills:
            if skill.name == "trend_analysis" and task.target_variable == "apr_change":
                # Adjust trend weight based on error
                current_weight = skill.parameters.get('trend_weight', 0.5)
                
                # If error is large, make a bigger adjustment
                adjustment = min(0.05, error / 20)
                
                # If our prediction was too high, decrease weight; if too low, increase
                pred_value = task.predictions[self.agent_id]["value"]
                if pred_value > task.true_outcome:
                    new_weight = max(0.1, current_weight - adjustment)
                else:
                    new_weight = min(0.9, current_weight + adjustment)
                    
                skill.parameters['trend_weight'] = new_weight
                
            elif skill.name == "tvl_momentum" and task.target_variable == "tvl_change":
                # Similar adaptation for TVL momentum
                current_factor = skill.parameters.get('momentum_factor', 0.7)
                adjustment = min(0.05, error / 20)
                
                pred_value = task.predictions[self.agent_id]["value"]
                if pred_value > task.true_outcome:
                    new_factor = max(0.2, current_factor - adjustment)
                else:
                    new_factor = min(0.9, current_factor + adjustment)
                    
                skill.parameters['momentum_factor'] = new_factor
    
    def _extract_knowledge(self, task: PredictionTask) -> None:
        """
        Extract knowledge from a completed task
        
        Args:
            task: Completed prediction task
        """
        # Simple knowledge extraction example
        if len(self.knowledge_base) >= MAX_KNOWLEDGE_SIZE:
            # Remove oldest item if at capacity
            self.knowledge_base = self.knowledge_base[1:]
            
        # Example: Learn about category performance
        category = task.input_data.get('category', 'Other')
        pred_value = task.predictions[self.agent_id]["value"]
        true_value = task.true_outcome
        error = abs(pred_value - true_value)
        
        # Create knowledge item for category behavior
        knowledge_id = f"category_{category}_{len(self.knowledge_base)}"
        
        knowledge = KnowledgeItem(
            knowledge_id=knowledge_id,
            content={
                "category": category,
                "predicted": pred_value,
                "actual": true_value,
                "error": error
            },
            knowledge_type="category_behavior",
            confidence=max(0, 1.0 - (error / 10)),  # Higher error = lower confidence
            source=self.agent_id
        )
        
        self.knowledge_base.append(knowledge)
    
    def add_skill(self, skill: Skill) -> None:
        """
        Add a new skill to the agent
        
        Args:
            skill: Skill to add
        """
        # Check if similar skill already exists
        for existing_skill in self.skills:
            if existing_skill.name == skill.name:
                # Update existing skill
                existing_skill.parameters.update(skill.parameters)
                return
                
        # Add new skill
        self.skills.append(skill)
    
    def share_knowledge(self, knowledge_filter: Callable = None) -> List[KnowledgeItem]:
        """
        Share knowledge with other agents
        
        Args:
            knowledge_filter: Optional filter function for knowledge items
            
        Returns:
            List of shareable knowledge items
        """
        if knowledge_filter:
            return [k for k in self.knowledge_base if knowledge_filter(k)]
        return self.knowledge_base
    
    def receive_knowledge(self, knowledge_items: List[KnowledgeItem]) -> None:
        """
        Receive knowledge from other agents
        
        Args:
            knowledge_items: Knowledge items to receive
        """
        # Filter out knowledge we already have
        existing_ids = {k.knowledge_id for k in self.knowledge_base}
        new_items = [k for k in knowledge_items if k.knowledge_id not in existing_ids]
        
        # Add new knowledge items
        self.knowledge_base.extend(new_items)
        
        # Trim knowledge base if needed
        if len(self.knowledge_base) > MAX_KNOWLEDGE_SIZE:
            # Sort by confidence and keep the most confident items
            self.knowledge_base.sort(key=lambda k: k.confidence, reverse=True)
            self.knowledge_base = self.knowledge_base[:MAX_KNOWLEDGE_SIZE]
    
    def calculate_fitness(self) -> float:
        """
        Calculate agent fitness for evolutionary selection
        
        Returns:
            Fitness score (0-1)
        """
        # Combine various factors for fitness
        performance_score = self.performance.success_rate
        
        # Recent activity factor
        days_since_active = (datetime.now() - self.last_active).days
        recency_factor = max(0, 1.0 - (days_since_active / 30))  # Decay over 30 days
        
        # Skill diversity factor
        skill_count = len(self.skills)
        skill_factor = min(1.0, skill_count / 5)  # Up to 5 skills
        
        # Knowledge factor
        knowledge_factor = min(1.0, len(self.knowledge_base) / 50)  # Up to 50 knowledge items
        
        # Combine factors with weights
        fitness = (
            performance_score * 0.5 +
            recency_factor * 0.2 +
            skill_factor * 0.15 +
            knowledge_factor * 0.15
        )
        
        return fitness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "skills": [skill.to_dict() for skill in self.skills],
            "knowledge_base": [k.to_dict() for k in self.knowledge_base],
            "performance": self.performance.to_dict(),
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "specializations": list(self.specializations),
            "collaborators": list(self.collaborators),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create from dictionary"""
        agent = cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            skills=[Skill.from_dict(s) for s in data.get("skills", [])]
        )
        
        # Restore knowledge base
        agent.knowledge_base = [KnowledgeItem.from_dict(k) for k in data.get("knowledge_base", [])]
        
        # Restore performance
        perf = data.get("performance", {})
        agent.performance.successful_predictions = perf.get("successful_predictions", 0)
        agent.performance.total_predictions = perf.get("total_predictions", 0)
        agent.performance.reward_collected = perf.get("reward_collected", 0.0)
        agent.performance.avg_prediction_error = perf.get("avg_prediction_error", 0.0)
        
        if "last_update" in perf:
            agent.performance.last_update = datetime.fromisoformat(perf["last_update"])
        
        # Restore timestamps
        agent.created_at = datetime.fromisoformat(data["created_at"])
        agent.last_active = datetime.fromisoformat(data["last_active"])
        
        # Restore sets
        agent.specializations = set(data.get("specializations", []))
        agent.collaborators = set(data.get("collaborators", []))
        
        # Restore metadata
        agent.metadata = data.get("metadata", {})
        
        return agent
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Agent(id={self.agent_id}, type={self.agent_type}, skills={len(self.skills)})"

class APRPredictionAgent(Agent):
    """Agent specialized in APR predictions"""
    
    def __init__(self, agent_id: str, skills: List[Skill] = None):
        """Initialize APR prediction agent"""
        default_skills = skills or [
            Skill(
                name="trend_analysis",
                parameters={"trend_weight": 0.5},
                skill_type="basic",
                target_variables=["apr_change"]
            ),
            Skill(
                name="volume_impact",
                parameters={"volume_factor": 2.0},
                skill_type="basic",
                target_variables=["apr_change"]
            ),
            Skill(
                name="category_bias",
                parameters={
                    "category_factors": {
                        "Major": 0.0,
                        "Meme": 1.5,
                        "DeFi": 0.8,
                        "Gaming": 0.5,
                        "Stablecoin": -0.5,
                        "Other": 0.2
                    }
                },
                skill_type="specialized",
                target_variables=["apr_change"]
            )
        ]
        
        super().__init__(agent_id, "apr_predictor", default_skills)
        self.specializations.add("apr_prediction")

class TVLPredictionAgent(Agent):
    """Agent specialized in TVL predictions"""
    
    def __init__(self, agent_id: str, skills: List[Skill] = None):
        """Initialize TVL prediction agent"""
        default_skills = skills or [
            Skill(
                name="tvl_momentum",
                parameters={"momentum_factor": 0.7},
                skill_type="basic",
                target_variables=["tvl_change"]
            ),
            Skill(
                name="apr_correlation",
                parameters={"correlation_strength": 0.3},
                skill_type="basic",
                target_variables=["tvl_change"]
            ),
            Skill(
                name="category_impact",
                parameters={
                    "category_factors": {
                        "Major": 1.0,
                        "Meme": 2.0,
                        "DeFi": 0.5,
                        "Gaming": 0.3,
                        "Stablecoin": 0.2,
                        "Other": 0.1
                    }
                },
                skill_type="specialized",
                target_variables=["tvl_change"]
            )
        ]
        
        super().__init__(agent_id, "tvl_predictor", default_skills)
        self.specializations.add("tvl_prediction")

class RiskAssessmentAgent(Agent):
    """Agent specialized in risk assessment"""
    
    def __init__(self, agent_id: str, skills: List[Skill] = None):
        """Initialize risk assessment agent"""
        default_skills = skills or [
            Skill(
                name="volatility_analysis",
                parameters={"volatility_weight": 0.6},
                skill_type="basic",
                target_variables=["risk_score"]
            ),
            Skill(
                name="liquidity_assessment",
                parameters={"liquidity_threshold": 1_000_000},
                skill_type="basic",
                target_variables=["risk_score"]
            )
        ]
        
        super().__init__(agent_id, "risk_assessor", default_skills)
        self.specializations.add("risk_assessment")

class MetaAgent(Agent):
    """Agent that coordinates other agents"""
    
    def __init__(self, agent_id: str, skills: List[Skill] = None):
        """Initialize meta agent"""
        default_skills = skills or [
            Skill(
                name="agent_weighting",
                parameters={"weight_update_rate": 0.1},
                skill_type="meta",
                target_variables=["meta_prediction"]
            ),
            Skill(
                name="confidence_calibration",
                parameters={"calibration_factor": 0.8},
                skill_type="meta",
                target_variables=["meta_prediction"]
            )
        ]
        
        super().__init__(agent_id, "meta", default_skills)
        self.agent_weights = {}  # agent_id -> weight
        self.specializations.add("meta_prediction")
    
    def predict(self, task: PredictionTask, agent_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a meta-prediction by aggregating other agents' predictions
        
        Args:
            task: Prediction task
            agent_predictions: Dictionary of agent predictions
            
        Returns:
            Meta-prediction
        """
        self.last_active = datetime.now()
        
        # Add task to history
        self.task_history.append(task)
        if len(self.task_history) > AGENT_HISTORY_SIZE:
            self.task_history = self.task_history[-AGENT_HISTORY_SIZE:]
        
        # If no agent predictions, return default
        if not agent_predictions:
            default_prediction = {
                "prediction": 0.0,
                "confidence": 0.1
            }
            
            task.predictions[self.agent_id] = {
                "value": default_prediction["prediction"],
                "confidence": default_prediction["confidence"],
                "timestamp": datetime.now().isoformat()
            }
            
            return default_prediction
        
        # Compute weighted average of predictions
        total_weight = 0.0
        weighted_sum = 0.0
        
        for agent_id, prediction in agent_predictions.items():
            # Get agent weight (default to 1.0)
            weight = self.agent_weights.get(agent_id, 1.0)
            
            # Adjust weight by prediction confidence
            confidence = prediction.get("confidence", 0.5)
            adjusted_weight = weight * confidence
            
            # Update weighted sum
            value = prediction.get("prediction", 0.0)
            weighted_sum += value * adjusted_weight
            total_weight += adjusted_weight
            
        # Calculate final prediction
        if total_weight > 0:
            final_prediction = weighted_sum / total_weight
        else:
            final_prediction = 0.0
            
        # Calculate consensus confidence
        agent_values = [p.get("prediction", 0.0) for p in agent_predictions.values()]
        avg_prediction = sum(agent_values) / len(agent_values)
        variance = sum((p - avg_prediction) ** 2 for p in agent_values) / len(agent_values)
        
        # Higher variance = lower confidence
        consensus_factor = max(0.1, 1.0 - (variance / 10.0))
        
        # Adjust by number of agents (more agents = higher confidence, up to a point)
        num_agents_factor = min(1.0, len(agent_predictions) / 5.0)
        
        meta_confidence = consensus_factor * num_agents_factor
        
        # Store prediction in task
        task.predictions[self.agent_id] = {
            "value": final_prediction,
            "confidence": meta_confidence,
            "timestamp": datetime.now().isoformat(),
            "agent_weights": {agent_id: self.agent_weights.get(agent_id, 1.0) for agent_id in agent_predictions}
        }
        
        return {
            "prediction": final_prediction,
            "confidence": meta_confidence
        }
    
    def update_weights(self, task: PredictionTask) -> None:
        """
        Update agent weights based on prediction accuracy
        
        Args:
            task: Completed prediction task
        """
        if task.true_outcome is None:
            return
            
        # Calculate errors for each agent
        agent_errors = {}
        
        for agent_id, prediction_data in task.predictions.items():
            if agent_id == self.agent_id:
                continue  # Skip self
                
            prediction = prediction_data.get("value", 0.0)
            error = abs(prediction - task.true_outcome)
            agent_errors[agent_id] = error
        
        # Update weights based on errors
        for agent_id, error in agent_errors.items():
            current_weight = self.agent_weights.get(agent_id, 1.0)
            
            # Calculate weight adjustment (more error = lower weight)
            adjustment = 0.1 * min(1.0, error / 5.0)
            
            # Update weight
            new_weight = max(0.1, current_weight * (1.0 - adjustment))
            self.agent_weights[agent_id] = new_weight

class MultiAgentSystem:
    """
    Coordinates multiple agents for prediction tasks
    """
    
    def __init__(self):
        """Initialize the multi-agent system"""
        self.agents = {}  # agent_id -> Agent
        self.task_history = []
        self.knowledge_repository = []
        self.meta_agent = None
        self.last_evolutionary_step = None
        self.system_metrics = {
            "tasks_completed": 0,
            "evolution_cycles": 0,
            "avg_prediction_error": 0.0,
            "knowledge_items": 0
        }
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the system
        
        Args:
            agent: Agent to add
        """
        self.agents[agent.agent_id] = agent
        
        # If this is a meta agent, store it separately
        if agent.agent_type == "meta":
            self.meta_agent = agent
    
    def create_default_agents(self) -> None:
        """Create default set of agents"""
        # APR prediction agents (with varying skills)
        for i in range(3):
            agent_id = f"apr_agent_{i+1}"
            agent = APRPredictionAgent(agent_id)
            self.add_agent(agent)
        
        # TVL prediction agents
        for i in range(3):
            agent_id = f"tvl_agent_{i+1}"
            agent = TVLPredictionAgent(agent_id)
            self.add_agent(agent)
        
        # Risk assessment agents
        for i in range(2):
            agent_id = f"risk_agent_{i+1}"
            agent = RiskAssessmentAgent(agent_id)
            self.add_agent(agent)
        
        # Create meta-agent if needed
        if self.meta_agent is None:
            meta_agent = MetaAgent("meta_agent_1")
            self.add_agent(meta_agent)
            self.meta_agent = meta_agent
    
    def predict(self, input_data: Dict[str, Any], target_variable: str) -> Dict[str, Any]:
        """
        Generate prediction using the multi-agent system
        
        Args:
            input_data: Input data for prediction
            target_variable: Target variable to predict
            
        Returns:
            Prediction result
        """
        # Create a prediction task
        task_id = f"task_{len(self.task_history) + 1}"
        task = PredictionTask(
            task_id=task_id,
            input_data=input_data,
            target_variable=target_variable
        )
        
        # Select agents that specialize in this target variable
        relevant_agents = []
        for agent in self.agents.values():
            is_relevant = False
            for skill in agent.skills:
                if target_variable in skill.target_variables:
                    is_relevant = True
                    break
            
            if is_relevant:
                relevant_agents.append(agent)
        
        # If no relevant agents, use all non-meta agents
        if not relevant_agents:
            relevant_agents = [a for a in self.agents.values() if a.agent_type != "meta"]
        
        # Get predictions from each agent
        agent_predictions = {}
        for agent in relevant_agents:
            # Skip meta agents for now
            if agent.agent_type == "meta":
                continue
                
            prediction = agent.predict(task)
            agent_predictions[agent.agent_id] = prediction
        
        # If we have a meta agent, use it to generate final prediction
        final_prediction = None
        if self.meta_agent:
            final_prediction = self.meta_agent.predict(task, agent_predictions)
        else:
            # Simple average if no meta agent
            predictions = [p.get("prediction", 0.0) for p in agent_predictions.values()]
            confidences = [p.get("confidence", 0.5) for p in agent_predictions.values()]
            
            if predictions:
                avg_prediction = sum(predictions) / len(predictions)
                avg_confidence = sum(confidences) / len(confidences)
                
                final_prediction = {
                    "prediction": avg_prediction,
                    "confidence": avg_confidence
                }
            else:
                final_prediction = {
                    "prediction": 0.0,
                    "confidence": 0.1
                }
        
        # Add task to history
        self.task_history.append(task)
        
        # Return final prediction with some metadata
        result = {
            "value": final_prediction["prediction"],
            "confidence": final_prediction["confidence"],
            "task_id": task_id,
            "agent_count": len(relevant_agents),
            "meta_agent": self.meta_agent is not None,
            "created_at": datetime.now().isoformat()
        }
        
        return result
    
    def update_with_outcome(self, task_id: str, true_outcome: Any) -> None:
        """
        Update the system with the true outcome of a prediction task
        
        Args:
            task_id: ID of the task
            true_outcome: True outcome value
        """
        # Find the task
        task = None
        for t in self.task_history:
            if t.task_id == task_id:
                task = t
                break
                
        if task is None:
            logger.error(f"Task not found: {task_id}")
            return
            
        # Update task with true outcome
        task.true_outcome = true_outcome
        
        # Update system metrics
        self.system_metrics["tasks_completed"] += 1
        
        # Calculate prediction error
        meta_prediction = task.predictions.get(self.meta_agent.agent_id if self.meta_agent else "", {})
        meta_value = meta_prediction.get("value", 0.0)
        prediction_error = abs(meta_value - true_outcome)
        
        # Update average error
        if self.system_metrics["tasks_completed"] == 1:
            self.system_metrics["avg_prediction_error"] = prediction_error
        else:
            prev_avg = self.system_metrics["avg_prediction_error"]
            count = self.system_metrics["tasks_completed"]
            self.system_metrics["avg_prediction_error"] = (prev_avg * (count - 1) + prediction_error) / count
        
        # Update agents with outcome
        for agent_id, prediction_data in task.predictions.items():
            if agent_id in self.agents:
                # Calculate reward based on prediction accuracy
                agent_value = prediction_data.get("value", 0.0)
                agent_error = abs(agent_value - true_outcome)
                
                # Simple reward function: higher for lower error
                reward = max(0, 1.0 - (agent_error / 5.0))
                
                # Let agent learn from outcome
                self.agents[agent_id].learn(task, reward)
        
        # Special handling for meta agent
        if self.meta_agent:
            self.meta_agent.update_weights(task)
        
        # Check if it's time for evolutionary step
        if (self.last_evolutionary_step is None or 
            (datetime.now() - self.last_evolutionary_step).total_seconds() > 3600):  # Once per hour
            self._evolutionary_step()
    
    def _evolutionary_step(self) -> None:
        """Perform evolutionary step to improve agent population"""
        # Update timestamp
        self.last_evolutionary_step = datetime.now()
        self.system_metrics["evolution_cycles"] += 1
        
        # Calculate fitness for each agent
        agent_fitness = {}
        for agent_id, agent in self.agents.items():
            # Skip meta agent
            if agent.agent_type == "meta":
                continue
                
            fitness = agent.calculate_fitness()
            agent_fitness[agent_id] = fitness
        
        # Sort agents by fitness
        sorted_agents = sorted(
            [(agent_id, fitness) for agent_id, fitness in agent_fitness.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check if we have too many agents
        if len(sorted_agents) <= MAX_AGENTS:
            # No need to remove any agents
            pass
        else:
            # Remove the least fit agents
            for agent_id, fitness in sorted_agents[MAX_AGENTS:]:
                if fitness < MIN_FITNESS_THRESHOLD:
                    logger.info(f"Removing unfit agent: {agent_id} with fitness {fitness:.3f}")
                    del self.agents[agent_id]
        
        # Recalculate after removals
        sorted_agents = [(agent_id, fitness) for agent_id, fitness in sorted_agents 
                        if agent_id in self.agents and fitness >= MIN_FITNESS_THRESHOLD]
        
        # Create new agents through crossover and mutation
        new_agents = []
        
        # Only do crossover if we have enough agents
        if len(sorted_agents) >= 2:
            # Number of new agents to create
            n_new = min(3, MAX_AGENTS - len(self.agents))
            
            for i in range(n_new):
                # Select parents with probability proportional to fitness
                parent1_id = self._select_parent([a[0] for a in sorted_agents], [a[1] for a in sorted_agents])
                parent2_id = self._select_parent([a[0] for a in sorted_agents], [a[1] for a in sorted_agents])
                
                # Ensure different parents
                attempt = 0
                while parent2_id == parent1_id and attempt < 5:
                    parent2_id = self._select_parent([a[0] for a in sorted_agents], [a[1] for a in sorted_agents])
                    attempt += 1
                
                if parent1_id is not None and parent2_id is not None:
                    # Create child through crossover and mutation
                    parent1 = self.agents[parent1_id]
                    parent2 = self.agents[parent2_id]
                    
                    child = self._create_child_agent(parent1, parent2)
                    new_agents.append(child)
        
        # Add all new agents
        for agent in new_agents:
            logger.info(f"Adding new agent: {agent.agent_id}")
            self.add_agent(agent)
        
        # Share knowledge between agents
        self._share_knowledge()
    
    def _select_parent(self, agent_ids: List[str], fitness_values: List[float]) -> Optional[str]:
        """
        Select parent using fitness proportional selection
        
        Args:
            agent_ids: List of agent IDs
            fitness_values: Corresponding fitness values
            
        Returns:
            Selected agent ID
        """
        if not agent_ids:
            return None
            
        # Ensure fitness values are positive
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            adj_fitness = [f - min_fitness + 0.01 for f in fitness_values]
        else:
            adj_fitness = fitness_values
            
        # Calculate selection probabilities
        total_fitness = sum(adj_fitness)
        if total_fitness <= 0:
            # If all fitness values are 0, use uniform selection
            return random.choice(agent_ids)
            
        # Fitness proportional selection
        probabilities = [f / total_fitness for f in adj_fitness]
        return random.choices(agent_ids, weights=probabilities, k=1)[0]
    
    def _create_child_agent(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        Create a child agent through crossover and mutation
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child agent
        """
        # Decide child agent type
        if random.random() < 0.5:
            agent_type = parent1.agent_type
        else:
            agent_type = parent2.agent_type
            
        # Generate new ID
        new_id = f"{agent_type}_{random.randint(1000, 9999)}"
        
        # Create child based on agent type
        if agent_type == "apr_predictor":
            child = APRPredictionAgent(new_id, [])
        elif agent_type == "tvl_predictor":
            child = TVLPredictionAgent(new_id, [])
        elif agent_type == "risk_assessor":
            child = RiskAssessmentAgent(new_id, [])
        else:
            # Fallback to generic agent
            child = Agent(new_id, agent_type, [])
        
        # Crossover skills
        all_skills = parent1.skills + parent2.skills
        
        # Select some skills from parents
        n_skills = min(5, len(all_skills))
        selected_skills = random.sample(all_skills, n_skills)
        
        # Add skills with mutation
        for skill in selected_skills:
            # Decide whether to mutate
            if random.random() < DEFAULT_MUTATION_RATE:
                mutated_skill = skill.mutate()
                child.add_skill(mutated_skill)
            else:
                child.add_skill(deepcopy(skill))
        
        # Inherit some knowledge from parents
        parent1_knowledge = parent1.share_knowledge()
        parent2_knowledge = parent2.share_knowledge()
        
        # Select random subset of knowledge
        all_knowledge = parent1_knowledge + parent2_knowledge
        n_knowledge = min(20, len(all_knowledge))
        selected_knowledge = random.sample(all_knowledge, n_knowledge)
        
        child.receive_knowledge(selected_knowledge)
        
        # Record parents in metadata
        child.metadata["parent1"] = parent1.agent_id
        child.metadata["parent2"] = parent2.agent_id
        child.metadata["generation"] = max(
            parent1.metadata.get("generation", 0),
            parent2.metadata.get("generation", 0)
        ) + 1
        
        return child
    
    def _share_knowledge(self) -> None:
        """Share knowledge between agents"""
        # Collect all knowledge
        all_knowledge = []
        for agent in self.agents.values():
            knowledge = agent.share_knowledge()
            all_knowledge.extend(knowledge)
        
        # Remove duplicates
        knowledge_dict = {}
        for item in all_knowledge:
            if item.knowledge_id not in knowledge_dict:
                knowledge_dict[item.knowledge_id] = item
                
        # Update central repository
        self.knowledge_repository = list(knowledge_dict.values())
        self.system_metrics["knowledge_items"] = len(self.knowledge_repository)
        
        # Distribute knowledge to agents
        for agent in self.agents.values():
            # Select relevant knowledge for this agent
            relevant_knowledge = []
            
            for item in self.knowledge_repository:
                # Knowledge is relevant if it matches the agent's specializations
                is_relevant = False
                
                if item.knowledge_type == "category_behavior":
                    category = item.content.get("category")
                    if "apr_prediction" in agent.specializations and item.knowledge_type == "category_behavior":
                        is_relevant = True
                    elif "tvl_prediction" in agent.specializations and item.knowledge_type == "category_behavior":
                        is_relevant = True
                
                if is_relevant:
                    relevant_knowledge.append(item)
            
            # Share relevant knowledge with the agent
            if relevant_knowledge:
                agent.receive_knowledge(relevant_knowledge)
    
    def save_state(self, filepath: str) -> None:
        """
        Save system state to a file
        
        Args:
            filepath: Path to save to
        """
        # Create state dictionary
        state = {
            "agents": {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
            "meta_agent_id": self.meta_agent.agent_id if self.meta_agent else None,
            "system_metrics": self.system_metrics,
            "last_evolutionary_step": self.last_evolutionary_step.isoformat() if self.last_evolutionary_step else None,
            "knowledge_repository": [k.to_dict() for k in self.knowledge_repository],
            "task_history": [t.to_dict() for t in self.task_history[-100:]]  # Save only recent tasks
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved multi-agent system state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    @classmethod
    def load_state(cls, filepath: str) -> 'MultiAgentSystem':
        """
        Load system state from a file
        
        Args:
            filepath: Path to load from
            
        Returns:
            MultiAgentSystem instance
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Create new system
            system = cls()
            
            # Load agents
            for agent_id, agent_data in state["agents"].items():
                agent_type = agent_data["agent_type"]
                
                if agent_type == "apr_predictor":
                    agent = APRPredictionAgent.from_dict(agent_data)
                elif agent_type == "tvl_predictor":
                    agent = TVLPredictionAgent.from_dict(agent_data)
                elif agent_type == "risk_assessor":
                    agent = RiskAssessmentAgent.from_dict(agent_data)
                elif agent_type == "meta":
                    agent = MetaAgent.from_dict(agent_data)
                else:
                    agent = Agent.from_dict(agent_data)
                    
                system.add_agent(agent)
            
            # Set meta agent
            if state["meta_agent_id"] and state["meta_agent_id"] in system.agents:
                system.meta_agent = system.agents[state["meta_agent_id"]]
            
            # Load system metrics
            system.system_metrics = state["system_metrics"]
            
            # Load evolutionary timestamp
            if state["last_evolutionary_step"]:
                system.last_evolutionary_step = datetime.fromisoformat(state["last_evolutionary_step"])
            
            # Load knowledge repository
            system.knowledge_repository = [KnowledgeItem.from_dict(k) for k in state["knowledge_repository"]]
            
            # Load task history
            system.task_history = [PredictionTask.from_dict(t) for t in state["task_history"]]
            
            logger.info(f"Loaded multi-agent system with {len(system.agents)} agents")
            
            return system
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            raise e

# Example usage
if __name__ == "__main__":
    # Create multi-agent system
    mas = MultiAgentSystem()
    mas.create_default_agents()
    
    # Example prediction task
    input_data = {
        "category": "Meme",
        "tvl_change_7d": 5.2,
        "apr_change_7d": 3.1,
        "volume_liquidity_ratio": 0.15
    }
    
    # Generate prediction
    result = mas.predict(input_data, "apr_change")
    print(f"Prediction: {result['value']:.2f} with confidence {result['confidence']:.2f}")
    
    # Simulate receiving the true outcome later
    task_id = result["task_id"]
    true_outcome = 2.8
    mas.update_with_outcome(task_id, true_outcome)
    
    # Run a few more predictions and updates
    for i in range(5):
        # Vary the input data
        input_data = {
            "category": random.choice(["Meme", "Major", "DeFi", "Stablecoin"]),
            "tvl_change_7d": random.uniform(-10, 10),
            "apr_change_7d": random.uniform(-5, 5),
            "volume_liquidity_ratio": random.uniform(0.05, 0.3)
        }
        
        # Predict different targets
        target = random.choice(["apr_change", "tvl_change"])
        
        result = mas.predict(input_data, target)
        print(f"Prediction {i+2}: {result['value']:.2f} with confidence {result['confidence']:.2f}")
        
        # Simulate true outcome (for demonstration)
        true_outcome = result['value'] + random.uniform(-1, 1)
        mas.update_with_outcome(result["task_id"], true_outcome)
    
    # Trigger an evolutionary step
    mas._evolutionary_step()
    
    # Print system metrics
    print("\nSystem Metrics:")
    for key, value in mas.system_metrics.items():
        print(f"{key}: {value}")
    
    # Print agent fitness scores
    print("\nAgent Fitness Scores:")
    for agent_id, agent in mas.agents.items():
        if agent.agent_type != "meta":
            fitness = agent.calculate_fitness()
            print(f"{agent_id}: {fitness:.3f}")
    
    # Save the system state
    mas.save_state("multi_agent_system.pkl")