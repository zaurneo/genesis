"""
Comprehensive diagnostics system for Genesis multi-agent workflow.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

# Import event tracking
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from events import get_event_stream, Event, EventType, diagnose_missing_sections
    _events_available = True
except ImportError:
    _events_available = False
    print("⚠️ Event tracking not available for diagnostics")


class WorkflowDiagnostics:
    """Comprehensive diagnostic system for multi-agent workflows."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.checkpoints = []
        self.custom_metrics = {}
        self.expected_flow = [
            'supervisor_start',
            'stock_data_agent_start',
            'stock_data_agent_complete',
            'stock_analyzer_agent_start',
            'stock_analyzer_agent_complete',
            'stock_reporter_agent_start',
            'stock_reporter_agent_complete',
            'supervisor_complete'
        ]
    
    def add_checkpoint(self, name: str, data: Optional[Dict[str, Any]] = None):
        """Add a diagnostic checkpoint."""
        checkpoint = {
            'name': name,
            'timestamp': datetime.now(),
            'data': data or {}
        }
        self.checkpoints.append(checkpoint)
        
        if _events_available:
            event_stream = get_event_stream()
            event_stream.publish(Event(
                type=EventType.AGENT_COMPLETED,
                agent_id=f"checkpoint_{name}",
                session_id=self.session_id,
                data=checkpoint['data']
            ))
        
        print(f"✅ Checkpoint: {name} at {checkpoint['timestamp'].strftime('%H:%M:%S')}")
    
    def track_metric(self, metric_name: str, value: Any):
        """Track a custom metric."""
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = []
        self.custom_metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now()
        })
    
    def run_full_diagnostics(self) -> str:
        """Run comprehensive diagnostics and return formatted report."""
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append("🔍 COMPREHENSIVE WORKFLOW DIAGNOSTICS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Session ID: {self.session_id}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. Basic Event Analysis
        if _events_available:
            report_lines.append("📊 EVENT ANALYSIS")
            report_lines.append("-"*40)
            report_lines.extend(self._analyze_events())
            report_lines.append("")
        
        # 2. Checkpoint Analysis
        report_lines.append("🏁 CHECKPOINT ANALYSIS")
        report_lines.append("-"*40)
        report_lines.extend(self._analyze_checkpoints())
        report_lines.append("")
        
        # 3. Performance Metrics
        report_lines.append("⚡ PERFORMANCE METRICS")
        report_lines.append("-"*40)
        report_lines.extend(self._analyze_performance())
        report_lines.append("")
        
        # 4. File System Analysis
        report_lines.append("📁 FILE SYSTEM ANALYSIS")
        report_lines.append("-"*40)
        report_lines.extend(self._analyze_files())
        report_lines.append("")
        
        # 5. Workflow Validation
        report_lines.append("✔️ WORKFLOW VALIDATION")
        report_lines.append("-"*40)
        report_lines.extend(self._validate_workflow())
        report_lines.append("")
        
        # 6. Custom Metrics
        if self.custom_metrics:
            report_lines.append("📈 CUSTOM METRICS")
            report_lines.append("-"*40)
            report_lines.extend(self._format_custom_metrics())
            report_lines.append("")
        
        # 7. Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-"*40)
        report_lines.extend(self._generate_recommendations())
        
        report_lines.append("="*80)
        
        return "\n".join(report_lines)
    
    def _analyze_events(self) -> List[str]:
        """Analyze events from event stream."""
        lines = []
        event_stream = get_event_stream()
        events = event_stream.get_events(self.session_id)
        
        if not events:
            lines.append("❌ No events found for this session")
            return lines
        
        # Count by type
        event_counts = {}
        agent_performance = {}
        
        for event in events:
            # Count event types
            event_type = event.type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Track agent performance
            agent = event.agent_id
            if agent not in agent_performance:
                agent_performance[agent] = {
                    'started': 0,
                    'completed': 0,
                    'failed': 0,
                    'first_event': event.timestamp,
                    'last_event': event.timestamp
                }
            
            if event.type == EventType.AGENT_STARTED:
                agent_performance[agent]['started'] += 1
            elif event.type == EventType.AGENT_COMPLETED:
                agent_performance[agent]['completed'] += 1
            elif event.type == EventType.AGENT_FAILED:
                agent_performance[agent]['failed'] += 1
            
            agent_performance[agent]['last_event'] = event.timestamp
        
        lines.append(f"Total Events: {len(events)}")
        lines.append("\nEvent Type Distribution:")
        for event_type, count in sorted(event_counts.items()):
            lines.append(f"  • {event_type}: {count}")
        
        lines.append("\nAgent Performance:")
        for agent, perf in agent_performance.items():
            duration = (perf['last_event'] - perf['first_event']).total_seconds()
            status = "✅" if perf['completed'] > 0 else "❌" if perf['failed'] > 0 else "⏳"
            lines.append(f"  {status} {agent}:")
            lines.append(f"     Started: {perf['started']}, Completed: {perf['completed']}, Failed: {perf['failed']}")
            lines.append(f"     Duration: {duration:.2f}s")
        
        # Timeline
        lines.append("\nEvent Timeline (last 10):")
        for event in events[-10:]:
            lines.append(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.type.value} - {event.agent_id}")
        
        return lines
    
    def _analyze_checkpoints(self) -> List[str]:
        """Analyze checkpoints."""
        lines = []
        
        if not self.checkpoints:
            lines.append("❌ No checkpoints recorded")
            return lines
        
        lines.append(f"Total Checkpoints: {len(self.checkpoints)}")
        
        # Calculate time between checkpoints
        lines.append("\nCheckpoint Timeline:")
        for i, cp in enumerate(self.checkpoints):
            if i > 0:
                time_diff = (cp['timestamp'] - self.checkpoints[i-1]['timestamp']).total_seconds()
                lines.append(f"  ↓ {time_diff:.2f}s")
            lines.append(f"  • {cp['timestamp'].strftime('%H:%M:%S')} - {cp['name']}")
            if cp['data']:
                lines.append(f"    Data: {json.dumps(cp['data'], indent=6)}")
        
        return lines
    
    def _analyze_performance(self) -> List[str]:
        """Analyze performance metrics."""
        lines = []
        
        if not self.checkpoints:
            lines.append("❌ No performance data available")
            return lines
        
        # Total workflow time
        if len(self.checkpoints) >= 2:
            total_time = (self.checkpoints[-1]['timestamp'] - self.checkpoints[0]['timestamp']).total_seconds()
            lines.append(f"Total Workflow Time: {total_time:.2f}s")
            
            # Average time between checkpoints
            time_diffs = []
            for i in range(1, len(self.checkpoints)):
                diff = (self.checkpoints[i]['timestamp'] - self.checkpoints[i-1]['timestamp']).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                avg_time = sum(time_diffs) / len(time_diffs)
                max_time = max(time_diffs)
                min_time = min(time_diffs)
                
                lines.append(f"Average Checkpoint Interval: {avg_time:.2f}s")
                lines.append(f"Longest Operation: {max_time:.2f}s")
                lines.append(f"Shortest Operation: {min_time:.2f}s")
        
        return lines
    
    def _analyze_files(self) -> List[str]:
        """Analyze output files."""
        lines = []
        
        output_dir = Path("output")
        if not output_dir.exists():
            lines.append("❌ Output directory not found")
            return lines
        
        # Count files by type
        file_types = {}
        total_size = 0
        newest_file = None
        newest_time = None
        
        for file in output_dir.iterdir():
            if file.is_file():
                ext = file.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                
                stat = file.stat()
                total_size += stat.st_size
                
                if newest_time is None or stat.st_mtime > newest_time:
                    newest_time = stat.st_mtime
                    newest_file = file.name
        
        lines.append(f"Total Files: {sum(file_types.values())}")
        lines.append(f"Total Size: {total_size / (1024*1024):.2f} MB")
        
        lines.append("\nFile Types:")
        for ext, count in sorted(file_types.items()):
            lines.append(f"  • {ext}: {count} files")
        
        if newest_file:
            lines.append(f"\nNewest File: {newest_file}")
            lines.append(f"Created: {datetime.fromtimestamp(newest_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return lines
    
    def _validate_workflow(self) -> List[str]:
        """Validate workflow execution."""
        lines = []
        
        # Check expected vs actual flow
        actual_flow = [cp['name'] for cp in self.checkpoints]
        
        lines.append("Expected Flow:")
        for step in self.expected_flow:
            if step in actual_flow:
                lines.append(f"  ✅ {step}")
            else:
                lines.append(f"  ❌ {step} (missing)")
        
        # Additional validation
        if _events_available:
            lines.append("\nValidation Checks:")
            
            # Use the built-in diagnostic
            diagnostic = diagnose_missing_sections(self.session_id)
            if "Root cause:" in diagnostic:
                lines.append("  " + diagnostic.split("Root cause:")[-1].strip())
        
        return lines
    
    def _format_custom_metrics(self) -> List[str]:
        """Format custom metrics."""
        lines = []
        
        for metric_name, values in self.custom_metrics.items():
            lines.append(f"\n{metric_name}:")
            for v in values[-5:]:  # Last 5 values
                lines.append(f"  • {v['value']} at {v['timestamp'].strftime('%H:%M:%S')}")
        
        return lines
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check for failures
        if _events_available:
            events = get_event_stream().get_events(self.session_id)
            failures = [e for e in events if e.type == EventType.AGENT_FAILED]
            
            if failures:
                recommendations.append("🔴 Critical: Agent failures detected")
                for f in failures:
                    recommendations.append(f"   Fix: {f.agent_id} - {f.error}")
        
        # Check performance
        if self.checkpoints and len(self.checkpoints) >= 2:
            total_time = (self.checkpoints[-1]['timestamp'] - self.checkpoints[0]['timestamp']).total_seconds()
            if total_time > 60:
                recommendations.append("🟡 Performance: Workflow took over 1 minute")
                recommendations.append("   Consider: Optimizing data processing or model training")
        
        # Check completeness
        missing_checkpoints = [cp for cp in self.expected_flow if cp not in [c['name'] for c in self.checkpoints]]
        if missing_checkpoints:
            recommendations.append("🟡 Completeness: Missing expected checkpoints")
            recommendations.append(f"   Add: {', '.join(missing_checkpoints)}")
        
        if not recommendations:
            recommendations.append("✅ All systems functioning normally")
        
        return recommendations
    
    def save_diagnostic_report(self, filename: Optional[str] = None):
        """Save diagnostic report to file."""
        if not filename:
            filename = f"diagnostic_report_{self.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = Path("output") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        report = self.run_full_diagnostics()
        filepath.write_text(report)
        
        print(f"\n💾 Diagnostic report saved to: {filepath}")
        return str(filepath)


# Convenience functions
def create_diagnostics(session_id: Optional[str] = None) -> WorkflowDiagnostics:
    """Create a new diagnostics instance."""
    return WorkflowDiagnostics(session_id)


def quick_diagnostic(session_id: str) -> None:
    """Run a quick diagnostic check."""
    diag = WorkflowDiagnostics(session_id)
    print(diag.run_full_diagnostics())