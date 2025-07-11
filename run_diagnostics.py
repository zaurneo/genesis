#!/usr/bin/env python3
"""
Standalone diagnostic runner for Genesis workflow analysis.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from tools.diagnostics import WorkflowDiagnostics, quick_diagnostic
from events import get_event_stream


def main():
    parser = argparse.ArgumentParser(description='Run Genesis workflow diagnostics')
    parser.add_argument('--session', '-s', help='Session ID to analyze')
    parser.add_argument('--last', '-l', action='store_true', help='Analyze last session')
    parser.add_argument('--save', '-o', help='Save report to file')
    parser.add_argument('--live', action='store_true', help='Live monitoring mode')
    
    args = parser.parse_args()
    
    if args.live:
        # Live monitoring mode
        print("[REFRESH] LIVE MONITORING MODE (Press Ctrl+C to exit)")
        print("="*60)
        
        # Enable verbose event tracking
        event_stream = get_event_stream()
        event_stream.verbose = True
        
        print("Waiting for events...")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n\nExiting live monitoring...")
    
    elif args.last:
        # Find last session
        event_stream = get_event_stream()
        all_events = event_stream.get_events()
        
        if not all_events:
            print("[ERROR] No events found")
            return
        
        # Get unique sessions
        sessions = list(set(e.session_id for e in all_events if e.session_id))
        if not sessions:
            print("[ERROR] No sessions found")
            return
        
        # Get last session (most recent event)
        last_session = max(sessions, key=lambda s: max(
            e.timestamp for e in all_events if e.session_id == s
        ))
        
        print(f"Analyzing last session: {last_session}")
        diag = WorkflowDiagnostics(last_session)
        report = diag.run_full_diagnostics()
        print(report)
        
        if args.save:
            diag.save_diagnostic_report(args.save)
    
    elif args.session:
        # Analyze specific session
        diag = WorkflowDiagnostics(args.session)
        report = diag.run_full_diagnostics()
        print(report)
        
        if args.save:
            diag.save_diagnostic_report(args.save)
    
    else:
        # Interactive mode
        print("[SEARCH] GENESIS DIAGNOSTIC TOOL")
        print("="*40)
        print("1. Analyze specific session")
        print("2. Analyze last session")
        print("3. Live monitoring")
        print("4. Show all sessions")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            session_id = input("Enter session ID: ").strip()
            quick_diagnostic(session_id)
        elif choice == "2":
            main_args = ['--last']
            sys.argv[1:] = main_args
            main()
        # ... implement other options


if __name__ == "__main__":
    main()