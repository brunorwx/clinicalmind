# app/application/process_query_use_case.py
"""Use case for processing a query through the agent system."""
from typing import Dict, Any, AsyncIterator
from app.domain.query.value_objects import TrialId, Question, Answer
from app.domain.query.interfaces import QuerySynthesizer, Agent
from app.domain.safety.interfaces import SafetyChecker
from app.domain.shared.exceptions import SafetyBlockedError, InvalidInputError


class ProcessQueryUseCase:
    """Orchestrates query processing through safety and agent workflows."""
    
    def __init__(
        self,
        safety_checker: SafetyChecker,
        agents: Dict[str, Agent],
        synthesizer: QuerySynthesizer
    ):
        self.safety = safety_checker
        self.agents = agents
        self.synthesizer = synthesizer
    
    async def execute(
        self,
        trial_id: TrialId,
        question: Question,
        user_id: str,
        graph_executor = None  # LangGraph executor for streaming events
    ) -> Answer:
        """Execute the complete query workflow.
        
        1. Validate inputs
        2. Check safety
        3. Run agents (via graph)
        4. Synthesize answer
        
        Returns Answer object with synthesized response and sources.
        Raises SafetyBlockedError if safety check fails.
        """
        try:
            # Validate inputs
            if not trial_id:
                raise InvalidInputError("Trial ID is required")
            if not question:
                raise InvalidInputError("Question is required")
            if not user_id:
                raise InvalidInputError("User ID is required")
            
            # Check safety first
            safety_result = await self.safety.check(question, user_id)
            
            if safety_result.is_blocked():
                flag = await self.safety.create_flag(
                    safety_result,
                    question,
                    user_id
                )
                raise SafetyBlockedError(
                    safety_result.level,
                    safety_result.reason
                )
            
            # If we have a graph executor, use it; otherwise skip agent execution
            agent_outputs = {}
            retrieved_chunks = []
            
            if graph_executor:
                # Graph executor will handle agent orchestration
                agent_outputs, retrieved_chunks = await graph_executor(
                    trial_id,
                    question,
                    user_id
                )
            
            # Synthesize final answer
            answer = await self.synthesizer.synthesize(
                trial_id,
                question,
                agent_outputs,
                retrieved_chunks
            )
            
            return answer
            
        except (SafetyBlockedError, InvalidInputError):
            raise
        except Exception as e:
            raise SafetyBlockedError("error", f"Query processing failed: {str(e)}")
