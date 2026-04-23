# ClinicalMind DDD Refactoring - Architecture Guide

## Overview

ClinicalMind has been refactored to follow **Domain-Driven Design (DDD)** and clean code principles. The architecture is now organized into clear layers with explicit separation of concerns.

## New Directory Structure

```
app/
├── domain/                     # Pure business logic (no dependencies)
│   ├── shared/
│   │   └── exceptions.py      # Domain-level exceptions
│   ├── query/                 # Query processing domain
│   │   ├── value_objects.py   # TrialId, Question, Answer, Source
│   │   ├── repositories.py    # ChunkRepository interface
│   │   └── interfaces.py      # LLMProvider, Agent, QuerySynthesizer
│   ├── documents/             # Document management domain
│   │   └── value_objects.py   # DocumentChunk, DocumentMetadata
│   └── safety/                # Safety domain
│       ├── value_objects.py   # SafetyClassification, SafetyFlag
│       └── interfaces.py      # SafetyChecker interface
│
├── application/               # Use cases & orchestration
│   ├── process_query_use_case.py      # Core business workflow
│   └── upload_document_use_case.py    # Document ingestion workflow
│
├── infrastructure/            # External integrations
│   ├── llm_providers.py       # OpenAI LLM provider implementation
│   ├── query_synthesizer.py   # Query synthesizer implementation
│   ├── safety_checker.py      # Safety checker implementation
│   └── repositories/
│       └── chunk_repository.py  # PostgreSQL chunk repository
│
├── agent/                     # LangGraph orchestration (simplified)
│   ├── graph.py              # Graph building only
│   ├── state.py              # Unified agent state
│   ├── contexts/             # State contexts by agent
│   │   ├── query_context.py
│   │   ├── rag_context.py
│   │   ├── data_context.py
│   │   └── safety_context.py
│   ├── agents/               # Simple node executors
│   │   ├── supervisor.py
│   │   ├── rag_agent.py
│   │   ├── data_agent.py
│   │   └── safety_agent.py
│   └── tools/                # Agent tools
│
├── presentation/             # HTTP layer (routers)
│   └── routers/
│       ├── query.py         # Query endpoint
│       └── documents.py     # Upload endpoint
│
├── container.py             # Dependency injection container
├── main.py                  # FastAPI app
├── config.py                # Configuration
└── database.py              # Database setup
```

## Key Improvements

### 1. **Domain Layer** ✅

- **Value Objects**: `TrialId`, `Question`, `Answer`, `Source`, `SafetyClassification`
  - Immutable, with validation in `__post_init__`
  - Represent concepts from the domain language
  
- **Repositories**: Abstract `ChunkRepository` interface
  - Concrete implementation in infrastructure layer
  - Enables testing with mock repositories
  
- **Interfaces**: Define contracts for external dependencies
  - `LLMProvider` - abstract LLM interactions
  - `SafetyChecker` - abstract safety checking
  - `QuerySynthesizer` - abstract answer synthesis

### 2. **Application Layer** ✅

- **Use Cases**: Explicit business workflows
  - `ProcessQueryUseCase`: Full query orchestration
  - `UploadDocumentUseCase`: Document validation & ingestion
  - No HTTP concerns, fully testable
  
### 3. **Infrastructure Layer** ✅

- **Concrete Implementations**:
  - `OpenAILLMProvider`: LangChain ChatOpenAI wrapper
  - `OpenAISafetyChecker`: Safety classification
  - `DefaultQuerySynthesizer`: Answer synthesis
  - `PostgresChunkRepository`: Database access

### 4. **Presentation Layer** ✅

- **HTTP-only routers**:
  - Minimal HTTP logic
  - Delegates to use cases
  - No business logic in routers

### 5. **Simplified State** ✅

**Before**: `AgentState` had 40+ fields (god object)
**After**:

- Unified `AgentState` with organized subsections (cleaner)
- Separate context types for different agents
- Core context (messages, trial_id, user_id, question) kept minimal

## Benefits

✅ **Single Responsibility**: Each class has one reason to change
✅ **Testability**: Dependencies injected, easy to mock
✅ **Flexibility**: Swap OpenAI for Claude, Llama, etc.
✅ **Language-Driven**: Code reads like business requirements
✅ **Clear Contracts**: Interfaces define expected behavior
✅ **Reduced Coupling**: Domain doesn't depend on framework
✅ **Explicit Workflows**: Use cases show exactly what happens

## Usage Examples

### Processing a Query (Use Case)

```python
from app.container import Container
from app.domain.query.value_objects import TrialId, Question

use_case = Container.get_process_query_use_case()

answer = await use_case.execute(
    trial_id=TrialId("TRIAL-ABC123"),
    question=Question("How many patients in treatment arm?"),
    user_id="user123"
)

print(answer.text)
for source in answer.sources:
    print(f"  - {source.document} p.{source.page}")
```

### Injecting Dependencies

```python
from app.container import Container

# Get singleton instances with all deps injected
llm = Container.get_llm_provider()
safety = Container.get_safety_checker()
synthesizer = Container.get_synthesizer()
```

### Adding a New Agent

```python
from app.domain.query.interfaces import Agent
from app.domain.query.value_objects import TrialId, Question

class MyNewAgent(Agent):
    async def execute(self, trial_id: TrialId, question: Question, context):
        # Your agent logic here
        return {"my_output": "result"}
```

### Swapping LLM Providers

```python
from app.domain.query.interfaces import LLMProvider

class AnthropicLLMProvider(LLMProvider):
    async def synthesize_answer(self, trial_id, question, agent_outputs):
        # Use Claude instead of OpenAI
        pass

# Wire into container instead of OpenAILLMProvider
```

## Next Steps (Optional Improvements)

1. **Result Types**: Add `Result[T, E]` for better error handling

   ```python
   class Result(Generic[T, E]):
       ok: T | None
       err: E | None
   ```

2. **Event Bus**: Add domain events for side effects

   ```python
   class SafetyFlagCreated(DomainEvent):
       flag: SafetyFlag
   ```

3. **Dependency Injector Library**: Use a proper DI library

   ```bash
   pip install python-dependency-injector
   ```

4. **Policy Layer**: Add authorization/access control

   ```python
   class TrialAccessPolicy:
       async def can_user_access_trial(self, user_id, trial_id):
           pass
   ```

## Migration Notes

The refactoring maintains **backward compatibility** with existing endpoints:

- `/api/v1/query` still works (same interface)
- `/api/v1/documents/upload` still works (same interface)
- LangGraph orchestration unchanged
- Database models unchanged

The changes are **internal architecture only** - no breaking API changes.

## Testing Strategy

With DDD structure, testing becomes much simpler:

```python
# Test use case with mock dependencies
mock_checker = MockSafetyChecker()
mock_repo = MockChunkRepository()

use_case = ProcessQueryUseCase(
    safety_checker=mock_checker,
    agents={},
    synthesizer=mock_synthesizer
)

answer = await use_case.execute(trial_id, question, user_id)
assert answer.text == "expected answer"
```

## References

- **Domain-Driven Design**: Eric Evans, "Domain-Driven Design" (Blue Book)
- **Clean Architecture**: Robert C. Martin, "Clean Architecture"
- **Hexagonal Architecture**: Alistair Cockburn
