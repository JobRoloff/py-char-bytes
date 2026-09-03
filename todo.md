Phase 1: Schemas & Types    --> Define contracts and Pydantic models
Phase 2: Service Stubs     --> Create skeleton interfaces (raising NotImplementedError)
Phase 3: Test Suite        --> Write failing unit/integration tests
Phase 4: Implementations   --> Build out low-level services to make tests pass
Phase 5: Pipeline & Evals  --> Wire async orchestrator, CLI, and benchmark runner