"""Per-phase implementations for `CraftSession.craft()`.

The orchestrator stays in `session.py`; each phase block is moved here
as a free function that takes the session and an explicit kwargs bundle.
This keeps `craft()` skim-readable and makes each phase independently
unit-testable.
"""
