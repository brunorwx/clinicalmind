# app/infrastructure/safety_checker.py
"""Safety checker implementation."""
import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.domain.safety.interfaces import SafetyChecker
from app.domain.query.value_objects import Question
from app.domain.safety.value_objects import SafetyClassification, SafetyFlag
from app.config import settings


class OpenAISafetyChecker(SafetyChecker):
    """OpenAI-based safety checker."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    
    async def check(
        self,
        question: Question,
        user_id: str
    ) -> SafetyClassification:
        """Check if a question is safe using OpenAI."""
        safety_prompt = """You are a safety classifier for a clinical research system.

Classify the following query as:
- "safe": Can be answered normally
- "needs_review": Contains sensitive clinical info, flag for human review
- "blocked": Potentially harmful, should not be processed

Query: {query}

Respond with ONLY a JSON object like:
{{"level": "safe", "reason": "routine query"}}
"""
        
        resp = await self.llm.ainvoke(
            SystemMessage(content=safety_prompt.format(query=question.text))
        )
        
        try:
            import json
            data = json.loads(resp.content)
            return SafetyClassification(
                level=data.get("level", "safe"),
                reason=data.get("reason", "")
            )
        except Exception:
            return SafetyClassification(
                level="safe",
                reason="Default: treated as safe"
            )
    
    async def create_flag(
        self,
        classification: SafetyClassification,
        question: Question,
        user_id: str
    ) -> SafetyFlag:
        """Create a flag for a safety issue."""
        flag = SafetyFlag(
            id=str(uuid.uuid4()),
            classification=classification,
            query_text=question.text,
            user_id=user_id,
            created_at=datetime.utcnow().isoformat()
        )
        return flag
