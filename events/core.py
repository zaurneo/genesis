"""
Minimal event tracking for Genesis multi-agent system.

Simple, focused on the actual problem: finding which agent failed 
and why report sections are missing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import uuid


class EventType(Enum):
    """Only the events we actually need."""
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    TOOL_CALLED = "tool_called"
    TOOL_FAILED = "tool_failed"
    OUTPUT_INVALID = "output_invalid"


@dataclass
class Event:
    """Simple event record."""
    type: EventType
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def __repr__(self):
        """Clean representation for debugging."""
        return f"<{self.timestamp.isoformat()} | {self.type.value} | {self.agent_id}>"


class EventStream:
    """Dead simple event tracking."""
    
    def __init__(self):
        self._events: List[Event] = []
    
    def publish(self, event: Event) -> None:
        """Just store the event."""
        self._events.append(event)
        
        # Optional: print for immediate visibility during debugging
        if event.type == EventType.AGENT_FAILED:
            print(f"❌ {event.agent_id} failed: {event.error}")
        elif event.type == EventType.OUTPUT_INVALID:
            print(f"⚠️ {event.agent_id} produced invalid output: {event.data.get('reason')}")
    
    def get_events(self, session_id: Optional[str] = None) -> List[Event]:
        """Get all events, optionally filtered by session."""
        if session_id:
            return [e for e in self._events if e.session_id == session_id]
        return self._events.copy()
    
    def query(self, 
              event_type: Optional[EventType] = None,
              agent_id: Optional[str] = None,
              session_id: Optional[str] = None) -> List[Event]:
        """Simple query by type, agent, or session."""
        results = self._events
        
        if event_type:
            results = [e for e in results if e.type == event_type]
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if session_id:
            results = [e for e in results if e.session_id == session_id]
            
        return results
    
    def reset(self) -> None:
        """Clear all events - useful for testing."""
        self._events.clear()


# Singleton
_event_stream = EventStream()

def get_event_stream() -> EventStream:
    """Get the global event stream."""
    return _event_stream


# Configurable expectations - modify these based on your workflow
DEFAULT_EXPECTED_AGENTS = ['stock_data_agent', 'stock_analyzer_agent', 'stock_reporter_agent']
DEFAULT_EXPECTED_SECTIONS = ['Executive Summary', 'Key Findings', 'Technical Analysis', 'Recommendation']


def contains_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Check which keywords are missing from text.
    
    Args:
        text: Text to search in
        keywords: Keywords to look for (case-insensitive)
    
    Returns:
        List of keywords that were NOT found in the text
    """
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() not in text_lower]


def diagnose_missing_sections(
    session_id: str,
    expected_agents: Optional[List[str]] = None,
    expected_sections: Optional[List[str]] = None
) -> str:
    """
    Simple diagnostic: what happened and what's missing?
    
    This is the ACTUAL function you need - tells you exactly
    which agents ran, which failed, and what's missing.
    
    Args:
        session_id: Session to diagnose
        expected_agents: Agents that should have run (uses defaults if None)
        expected_sections: Sections that should be in final output (uses defaults if None)
    
    Returns:
        Formatted diagnostic report
    """
    events = get_event_stream().get_events(session_id)
    
    if not events:
        return f"No events found for session {session_id}"
    
    # Use provided expectations or defaults
    expected_agents = expected_agents or DEFAULT_EXPECTED_AGENTS
    expected_sections = expected_sections or DEFAULT_EXPECTED_SECTIONS
    
    # What actually happened
    started: Set[str] = set()
    completed: Set[str] = set()
    failed: Dict[str, str] = {}
    invalid_outputs: List[Dict[str, Any]] = []
    
    for event in events:
        if event.type == EventType.AGENT_STARTED:
            started.add(event.agent_id)
        elif event.type == EventType.AGENT_COMPLETED:
            completed.add(event.agent_id)
        elif event.type == EventType.AGENT_FAILED:
            failed[event.agent_id] = event.error or "Unknown error"
        elif event.type == EventType.OUTPUT_INVALID:
            invalid_outputs.append({
                'agent': event.agent_id,
                'reason': event.data.get('reason', 'Unknown'),
                'missing_sections': event.data.get('missing_sections', [])
            })
    
    # Build simple report
    report = [
        f"🔍 Diagnostic Report for Session: {session_id}",
        f"📊 Events: {len(events)}",
        ""
    ]
    
    # Which agents never even started?
    never_started = set(expected_agents) - started
    if never_started:
        report.append("❌ Agents that never started:")
        for agent in never_started:
            report.append(f"  • {agent}")
        report.append("")
    
    # Which agents started but didn't complete?
    started_not_completed = started - completed - set(failed.keys())
    if started_not_completed:
        report.append("⏳ Agents that started but didn't complete:")
        for agent in started_not_completed:
            report.append(f"  • {agent}")
        report.append("")
    
    # Which agents failed?
    if failed:
        report.append("❌ Failed agents:")
        for agent, error in failed.items():
            report.append(f"  • {agent}: {error}")
        report.append("")
    
    # What output problems?
    if invalid_outputs:
        report.append("📝 Output problems:")
        for output in invalid_outputs:
            report.append(f"  • {output['agent']}: {output['reason']}")
            if output['missing_sections']:
                report.append(f"    Missing: {', '.join(output['missing_sections'])}")
        report.append("")
    
    # Root cause
    if failed:
        report.append(f"🎯 Root cause: {list(failed.keys())[0]} failed")
    elif never_started:
        report.append(f"🎯 Root cause: Workflow stopped before reaching {list(never_started)[0]}")
    elif invalid_outputs:
        report.append(f"🎯 Root cause: {invalid_outputs[0]['agent']} produced incomplete output")
    else:
        report.append("🎯 Root cause: Unknown - add more event tracking")
    
    return "\n".join(report)


# Configurable validation expectations
VALIDATION_KEYWORDS = {
    'stock_data_agent': ['price', 'volume', 'data'],
    'stock_analyzer_agent': ['analysis', 'trend', 'indicator'],
    'stock_reporter_agent': ['Executive Summary', 'Key Findings', 'Technical Analysis', 'Recommendation']
}


def validate_output(
    content: str, 
    agent_id: str, 
    session_id: str,
    min_length: int = 100,
    expected_keywords: Optional[List[str]] = None
) -> Optional[Event]:
    """
    Simple output validation - returns event if output is invalid.
    
    Use this after each agent to check if they produced what they should.
    
    Args:
        content: The output content to validate
        agent_id: Which agent produced this
        session_id: Current session
        min_length: Minimum acceptable content length
        expected_keywords: Keywords to check for (uses defaults if None)
    
    Returns:
        Event if output is invalid, None if valid
    """
    # Use provided keywords or defaults
    if expected_keywords is None:
        expected_keywords = VALIDATION_KEYWORDS.get(agent_id, [])
    
    # Check for missing keywords
    missing = contains_keywords(content, expected_keywords)
    
    # Check length
    too_short = len(content) < min_length
    
    if missing or too_short:
        return Event(
            type=EventType.OUTPUT_INVALID,
            agent_id=agent_id,
            session_id=session_id,
            data={
                'reason': 'Missing expected sections' if missing else 'Output too short',
                'missing_sections': missing,
                'output_length': len(content),
                'expected_length': min_length
            }
        )
    
    return None  # Output is valid