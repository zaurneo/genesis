# conversation_viewer.py - Enhanced for Hierarchical 6-Agent Development System
"""
Enhanced conversation viewer for the hierarchical 6-agent development system.
Features Technical Lead authority visualization and task tracking display.
HIERARCHICAL: Shows Technical Lead oversight and authority over all agents.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Union
from pathlib import Path
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage

class CodeDevelopmentViewer:
    """Enhanced live conversation viewer for hierarchical 6-agent development system"""
    
    def __init__(self):
        self.colors = {
            'architect': '\033[96m',        # Cyan - Project design
            'writer': '\033[94m',           # Blue - Code writing (ONLY code writer)  
            'executor': '\033[92m',         # Green - Execution
            'technical_lead': '\033[91m',   # Red - Technical Lead AUTHORITY
            'task_manager': '\033[93m',     # Yellow - Task tracking
            'docs': '\033[97m',             # White - Documentation
            'finalizer': '\033[95m',        # Magenta - Session completion
            'user': '\033[90m',             # Gray - User input
            'tool': '\033[35m',             # Purple - Tool operations
            'handoff': '\033[33m',          # Orange - Handoff operations
            'authority': '\033[1;91m',      # Bold Red - Authority decisions
            'system': '\033[37m',           # Light gray - System
            'reset': '\033[0m'
        }
        
        self.agent_icons = {
            'architect': '🏗️',
            'writer': '✍️',
            'executor': '⚡',
            'technical_lead': '🧑‍💼',
            'task_manager': '📊',
            'docs': '📚',
            'finalizer': '🏁',
            'user': '👤',
            'tool': '🛠️',
            'handoff': '🔄',
            'authority': '👑',
            'system': '📋'
        }
        
        self.message_history = []
        self.seen_ids = set()
        self.last_handoff = None
        self.agent_stats = {}
        self.handoff_count = 0
        self.authority_decisions = 0
        self.task_updates = 0
        
        # Ensure logs directory exists
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)

        self.agent_action_history = []  # Track agent actions to detect cycles
        self.agent_visit_count = {}      # Count how many times each agent is visited
        
    def print_header(self):
        """Print enhanced conversation header for hierarchical system"""
        print("\n" + "="*100)
        print("🚀 HIERARCHICAL ENTERPRISE CODE DEVELOPMENT - 7 AI AGENT COLLABORATION")
        print("="*100)
        print("🏗️ ARCHITECT: Cyan    |  ✍️ WRITER: Blue      |  ⚡ EXECUTOR: Green")
        print("🧑‍💼 TECH LEAD: Red    |  📊 TASK MGR: Yellow  |  📚 DOCS: White")
        print("🏁 FINALIZER: Magenta |  🛠️ TOOLS: Purple    |  🔄 HANDOFFS: Orange")
        print("="*100)
        print("HIERARCHY: Technical Lead has AUTHORITY over all agents")
        print("CODE WRITER: Only Writer creates and fixes ALL code")
        print("TASK TRACKING: Task Manager updates only per Technical Lead directive")
        print("SESSION COMPLETION: Finalizer ends when all work is done")
        print("="*100 + "\n")
    
    def extract_text_content(self, content):
        """Extract readable text from various content formats"""
        if content is None:
            return ""
            
        if isinstance(content, str):
            return content.strip()
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                else:
                    text_parts.append(str(item))
            return ' '.join(text_parts).strip()
        
        return str(content).strip()
    
    def format_and_print(self, agent_name: str, content: str, icon: str = None, is_authority: bool = False):
        """Enhanced message formatting with hierarchy and authority visualization"""
        if not content or content in ["[]", "", "None"]:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Special handling for Technical Lead authority decisions
        if is_authority or (agent_name == 'technical_lead' and any(keyword in content.lower() 
            for keyword in ['directive', 'decision', 'authority', 'require', 'demand', 'approve', 'reject'])):
            color = self.colors['authority']
            agent_icon = '👑'
            agent_display = f"TECH LEAD [AUTHORITY]"
            self.authority_decisions += 1
        else:
            color = self.colors.get(agent_name, self.colors['system'])
            agent_icon = icon or self.agent_icons.get(agent_name, "💬")
            agent_display = agent_name.upper()
        
        # Update agent statistics
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {'messages': 0, 'total_chars': 0, 'authority_decisions': 0}
        self.agent_stats[agent_name]['messages'] += 1
        self.agent_stats[agent_name]['total_chars'] += len(content)
        if is_authority:
            self.agent_stats[agent_name]['authority_decisions'] += 1
        
        # Split content into lines for better formatting
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                if i == 0:
                    # First line with full header
                    if is_authority:
                        print(f"{color}[{timestamp}] {agent_icon} {agent_display}: {line}{self.colors['reset']}")
                    else:
                        print(f"{color}[{timestamp}] {agent_icon} {agent_display}: {line}{self.colors['reset']}")
                else:
                    # Continuation lines with proper indentation
                    print(f"{color}{''.ljust(14)}{line}{self.colors['reset']}")
        
        # Record in history with timestamp
        self.message_history.append((timestamp, agent_name, content, is_authority))
    
    def show_handoff(self, from_agent: str, to_agent: str, tool_name: str = None):
        """Display enhanced handoff visualization with hierarchy awareness"""
        handoff_key = f"{from_agent}->{to_agent}"
        if handoff_key != self.last_handoff:
            self.last_handoff = handoff_key
            self.handoff_count += 1
            
            from_icon = self.agent_icons.get(from_agent, "🤖")
            to_icon = self.agent_icons.get(to_agent, "🤖")
            
            # Special visualization for Technical Lead involvement
            if to_agent == 'technical_lead':
                print(f"\n{self.colors['authority']}{'═'*80}")
                print(f"👑 ESCALATION TO TECHNICAL LEAD #{self.handoff_count}: {from_icon} {from_agent.upper()} → {to_icon} TECH LEAD")
                print(f"   📋 Seeking authority guidance and validation")
                print(f"{'═'*80}{self.colors['reset']}\n")
            elif from_agent == 'technical_lead':
                print(f"\n{self.colors['authority']}{'═'*80}")
                print(f"👑 TECHNICAL LEAD DIRECTIVE #{self.handoff_count}: {from_icon} TECH LEAD → {to_icon} {to_agent.upper()}")
                print(f"   📋 Authority decision and direction")
                print(f"{'═'*80}{self.colors['reset']}\n")
            elif to_agent == 'finalizer':
                print(f"\n{self.colors['finalizer']}{'═'*80}")
                print(f"🏁 SESSION COMPLETION #{self.handoff_count}: {from_icon} {from_agent.upper()} → {to_icon} FINALIZER")
                print(f"   📋 All tasks completed - finalizing development session")
                print(f"{'═'*80}{self.colors['reset']}\n")
            else:
                print(f"\n{self.colors['handoff']}{'─'*80}")
                print(f"🔄 HANDOFF #{self.handoff_count}: {from_icon} {from_agent.upper()} → {to_icon} {to_agent.upper()}")
                if tool_name:
                    print(f"   📞 Using: {tool_name}")
                print(f"{'─'*80}{self.colors['reset']}\n")
    
    def show_task_update(self, content: str):
        """Special visualization for task updates"""
        if 'task' in content.lower() and ('update' in content.lower() or 'status' in content.lower()):
            self.task_updates += 1
            print(f"\n{self.colors['task_manager']}📊 TASK UPDATE #{self.task_updates}:")
            print(f"   {content[:100]}{'...' if len(content) > 100 else ''}")
            print(f"{'─'*60}{self.colors['reset']}\n")
    
    def process_message(self, msg, current_agent):
        """Enhanced message processing for hierarchical 6-agent system"""
        try:
            # Get message ID to avoid duplicates
            msg_id = None
            if hasattr(msg, 'id'):
                msg_id = msg.id
            elif isinstance(msg, dict) and 'id' in msg:
                msg_id = msg['id']
                
            # Skip if already seen
            if msg_id and msg_id in self.seen_ids:
                return
            if msg_id:
                self.seen_ids.add(msg_id)
            
            # Handle AIMessage objects
            if isinstance(msg, AIMessage):
                content = msg.content
                name = getattr(msg, 'name', current_agent)
                tool_calls = getattr(msg, 'tool_calls', [])
                
                # Check for authority decisions
                is_authority = (name == 'technical_lead' and any(keyword in content.lower() 
                    for keyword in ['directive', 'decision', 'authority', 'require', 'demand', 'approve', 'reject']))
                
                # Display agent message if there's content
                if content:
                    text = self.extract_text_content(content)
                    if text and not any(skip in text.lower() for skip in ['transfer_to_', 'successfully transferred']):
                        self.format_and_print(name, text, is_authority=is_authority)
                        
                        # Check for task updates
                        if name == 'task_manager':
                            self.show_task_update(text)
                
                # Display tool calls with special handling for authority and handoffs
                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        if tool_name.startswith('transfer_to_'):
                            # This is a handoff tool call
                            target_agent = tool_name.replace('transfer_to_', '')
                            if name == 'technical_lead':
                                self.format_and_print(name, f"👑 DIRECTING {target_agent.upper()} to proceed", is_authority=True)
                            else:
                                self.format_and_print(name, f"🔄 Requesting handoff to {target_agent.upper()}")
                        else:
                            # Regular tool call
                            self.format_and_print(name, f"🔧 Using tool: {tool_name}")
                return
            
            # Handle ToolMessage objects
            if isinstance(msg, ToolMessage):
                content = msg.content
                tool_name = msg.name
                
                # Handle handoff completions with enhanced visualization
                if tool_name.startswith('transfer_to_') and 'Successfully transferred' in content:
                    target = tool_name.replace('transfer_to_', '')
                    self.show_handoff(current_agent, target, tool_name)
                    return
                
                # Show enhanced tool results
                if content and self._is_important_tool_result(tool_name, content):
                    # Truncate very long outputs for readability
                    lines = content.strip().split('\n')
                    if len(lines) > 12:
                        result_text = '\n'.join(lines[:12]) + f"\n... ({len(lines)-12} more lines)"
                    else:
                        result_text = content.strip()
                    
                    # Categorize tool output
                    tool_category = self._categorize_tool(tool_name)
                    self.format_and_print('tool', f"{tool_category} {tool_name}:\n{result_text}")
                return
            
            # Handle dictionary messages
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '')
                name = msg.get('name', '')
                
                # Human messages
                if role in ['human', 'user']:
                    text = self.extract_text_content(content)
                    if text:
                        self.format_and_print('user', text)
                    return
                
                # Tool messages
                if role == 'tool':
                    tool_name = name or 'unknown_tool'
                    
                    # Handle handoff completions
                    if tool_name.startswith('transfer_to_') and 'Successfully transferred' in content:
                        target = tool_name.replace('transfer_to_', '')
                        self.show_handoff(current_agent, target, tool_name)
                        return
                    
                    # Show other tool results
                    if content and self._is_important_tool_result(tool_name, content):
                        lines = content.strip().split('\n')
                        if len(lines) > 12:
                            result_text = '\n'.join(lines[:12]) + f"\n... ({len(lines)-12} more lines)"
                        else:
                            result_text = content.strip()
                        
                        tool_category = self._categorize_tool(tool_name)
                        self.format_and_print('tool', f"{tool_category} {tool_name}:\n{result_text}")
                    return
                
                # Agent messages with name
                agent_names = ['architect', 'writer', 'executor', 'technical_lead', 'task_manager', 'docs', 'finalizer']
                if name in agent_names:
                    if content:
                        text = self.extract_text_content(content)
                        if text and not any(skip in text.lower() for skip in ['transfer_to_', 'successfully transferred']):
                            is_authority = (name == 'technical_lead' and any(keyword in text.lower() 
                                for keyword in ['directive', 'decision', 'authority', 'require', 'demand', 'approve', 'reject']))
                            self.format_and_print(name, text, is_authority=is_authority)
                            
                            # Check for task updates
                            if name == 'task_manager':
                                self.show_task_update(text)
                    return
                    
        except Exception as e:
            # Don't let message processing errors crash the whole system
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"\n{self.colors['tool']}⚠️ Message processing error: {e}{self.colors['reset']}")
    
    def _is_important_tool_result(self, tool_name: str, content: str) -> bool:
        """Determine if tool result should be displayed"""
        # Always show these tools
        important_tools = [
            'write_code_file', 'execute_python_file', 'monitor_execution',
            'analyze_code_quality', 'check_security', 'run_tests',
            'create_project_structure', 'install_missing_packages'
        ]
        
        # Don't show handoff tool results (handled separately)
        if tool_name.startswith('transfer_to_'):
            return False
        
        return (tool_name in important_tools or 
                '✅' in content or '❌' in content or 
                'ERROR' in content.upper() or 
                'SUCCESS' in content.upper() or
                'TASK' in content.upper())
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tools for better display"""
        categories = {
            'write_code_file': '📝',
            'execute_python_file': '⚡',
            'monitor_execution': '📊',
            'analyze_code_quality': '🔍',
            'check_security': '🔒',
            'run_tests': '✅',
            'create_project_structure': '🏗️',
            'install_missing_packages': '📦',
            'backup_code': '💾'
        }
        
        # Special handling for handoff tools
        if tool_name.startswith('transfer_to_'):
            return '🔄'
            
        return categories.get(tool_name, '🛠️')

    def _detect_cycle(self, actions: List[str]) -> bool:
        """Detect if there's a repeating cycle in recent actions"""
        if len(actions) < 4:
            return False
        
        # Check for simple 2-step cycles (A→B→A→B)
        if len(actions) >= 4:
            if actions[-1] == actions[-3] and actions[-2] == actions[-4]:
                return True
        
        # Check for 3-step cycles (A→B→C→A→B→C)
        if len(actions) >= 6:
            pattern = actions[-3:]
            if actions[-6:-3] == pattern:
                return True
        
        return False

    def run(self, graph, initial_message: str):
        """Run enhanced conversation with hierarchical system visualization"""
        self.print_header()
        
        # Show initial message
        self.format_and_print('user', initial_message)
        print()
        
        start_time = time.time()
        chunk_count = 0
        
        # Temporary increase recursion limit while debugging
        config = {"recursion_limit": 100}
        
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=initial_message)]},
                config=config
            ):
                chunk_count += 1
                
                # Detect potential cycles
                if chunk_count > 30:
                    print(f"\n{self.colors['tool']}⚠️ WARNING: High iteration count ({chunk_count})")
                    print(f"Agent visit counts: {self.agent_visit_count}")
                    
                    # Check for repeated patterns
                    if len(self.agent_action_history) > 10:
                        recent_actions = self.agent_action_history[-10:]
                        print(f"Recent actions: {' → '.join(recent_actions)}")
                        
                        # Detect simple cycles
                        if self._detect_cycle(recent_actions):
                            print(f"🔄 CYCLE DETECTED! Agents are repeating actions.")
                            print(f"Consider adding logic to prevent re-executing completed tasks.{self.colors['reset']}\n")
                
                # Process each agent's messages
                for agent_name, data in chunk.items():
                    # Track agent visits
                    self.agent_visit_count[agent_name] = self.agent_visit_count.get(agent_name, 0) + 1
                    
                    # Track actions for cycle detection
                    if len(self.agent_action_history) > 50:
                        self.agent_action_history.pop(0)  # Keep only recent history
                    self.agent_action_history.append(agent_name)
                    
                    try:
                        messages = data.get('messages', [])
                        for msg in messages:
                            try:
                                self.process_message(msg, agent_name)
                            except Exception as msg_error:
                                if os.environ.get("DEBUG", "").lower() == "true":
                                    print(f"\n{self.colors['tool']}⚠️ Message processing error: {msg_error}{self.colors['reset']}")
                                continue
                    except Exception as agent_error:
                        if os.environ.get("DEBUG", "").lower() == "true":
                            print(f"\n{self.colors['tool']}⚠️ Agent {agent_name} processing error: {agent_error}{self.colors['reset']}")
                        continue
                
                time.sleep(0.02)
                
        except Exception as e:
            print(f"\n{self.colors['tool']}❌ System Error: {e}{self.colors['reset']}")
            
            if "recursion_limit" in str(e).lower():
                print(f"\n{self.colors['tool']}🔍 CYCLE ANALYSIS:")
                print(f"Total iterations: {chunk_count}")
                print(f"Agent visit counts: {self.agent_visit_count}")
                print(f"Most visited agent: {max(self.agent_visit_count.items(), key=lambda x: x[1]) if self.agent_visit_count else 'None'}")
                print(f"\nRecent action pattern: {' → '.join(self.agent_action_history[-20:])}")
                print(f"\n💡 Solution: Fix the cycle in agent logic, don't just increase recursion limit!{self.colors['reset']}")
            else:
                print(f"{self.colors['tool']}💡 This may be a handoff tool or LangGraph compatibility issue{self.colors['reset']}")
        
        execution_time = time.time() - start_time
        print(f"\n{self.colors['system']}✅ Session completed in {execution_time:.1f} seconds!{self.colors['reset']}")
        self.print_enhanced_summary()
    
    def print_enhanced_summary(self):
        """Print comprehensive session summary with hierarchical statistics"""
        print(f"\n{self.colors['system']}📊 HIERARCHICAL DEVELOPMENT SESSION SUMMARY:")
        print("=" * 70)
        
        # Agent participation with hierarchy awareness
        print("🤖 Agent Participation & Hierarchy:")
        for agent, stats in sorted(self.agent_stats.items()):
            if agent != 'tool':
                icon = self.agent_icons.get(agent, "🤖")
                color = self.colors.get(agent, self.colors['system'])
                avg_length = stats['total_chars'] / max(1, stats['messages'])
                
                # Show agent role and capabilities
                if agent == 'technical_lead':
                    role_info = " (AUTHORITY - guides all agents)"
                    if stats.get('authority_decisions', 0) > 0:
                        role_info += f" [{stats['authority_decisions']} decisions]"
                elif agent == 'writer':
                    role_info = " (ONLY code writer - creates & fixes)"
                elif agent == 'task_manager':
                    role_info = " (task tracking per Tech Lead directive)"
                elif agent == 'finalizer':
                    role_info = " (session completion)"
                elif agent in ['architect', 'executor']:
                    role_info = " (read-only, reports to Tech Lead)"
                else:
                    role_info = " (reports to Tech Lead)"
                
                print(f"{color}  {icon} {agent.upper()}: {stats['messages']} messages "
                      f"({avg_length:.0f} avg chars){role_info}{self.colors['reset']}")
        
        # Tool usage
        tool_messages = sum(1 for _, agent, _, _ in self.message_history if agent == 'tool')
        if tool_messages > 0:
            print(f"🛠️ Tool operations: {tool_messages}")
        
        # Hierarchical statistics
        if self.handoff_count > 0:
            print(f"🔄 Agent handoffs: {self.handoff_count}")
        
        if self.authority_decisions > 0:
            print(f"👑 Technical Lead authority decisions: {self.authority_decisions}")
            
        if self.task_updates > 0:
            print(f"📊 Task status updates: {self.task_updates}")
        
        print(f"📈 Total exchanges: {len(self.message_history)}")
        print(f"📝 Role separation: Only Writer creates and fixes code")
        print(f"🧑‍💼 Authority structure: Technical Lead oversees all work")
        print(f"📊 Task management: Updates only per Technical Lead directive")
        print(f"🏁 Session completion: Finalizer ends when all work is done")
        print(f"{self.colors['reset']}")
    
    def save_log(self, filename=None):
        """Save enhanced development session log with hierarchical information"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hierarchical_session_{timestamp}.txt"
        
        log_file = self.logs_dir / filename
        
        if not self.message_history:
            print("No messages to save.")
            return str(log_file)
            
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("HIERARCHICAL ENTERPRISE CODE DEVELOPMENT SESSION LOG\n")
                f.write("7-Agent Collaborative Development System with Technical Lead Authority\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("HIERARCHICAL WORKFLOW:\n")
                f.write("🏗️ Architect → 🧑‍💼 Technical Lead → ✍️ Writer → ⚡ Executor → 🧑‍💼 Technical Lead → 📚 Docs → 🏁 Finalizer\n")
                f.write("                     ↕️                                          ↕️\n")
                f.write("                 📊 Task Manager                            📊 Task Manager\n\n")
                f.write("HIERARCHY: Technical Lead has AUTHORITY over all agents\n")
                f.write("ROLE SEPARATION: Only Writer creates and fixes ALL code\n")
                f.write("TASK TRACKING: Task Manager updates only per Technical Lead directive\n")
                f.write("SESSION COMPLETION: Finalizer ends when all work is done\n\n")
                
                for timestamp, agent, message, is_authority in self.message_history:
                    icon = self.agent_icons.get(agent, "🤖")
                    
                    if agent == 'technical_lead':
                        if is_authority:
                            role_marker = " (AUTHORITY DECISION)"
                        else:
                            role_marker = " (TECHNICAL LEAD)"
                    elif agent == 'writer':
                        role_marker = " (ONLY CODE WRITER)"
                    elif agent == 'task_manager':
                        role_marker = " (TASK TRACKING)"
                    elif agent == 'finalizer':
                        role_marker = " (SESSION COMPLETION)"
                    else:
                        role_marker = ""
                    
                    f.write(f"[{timestamp}] {icon} {agent.upper()}{role_marker}:\n")
                    f.write(f"{message}\n")
                    f.write("-" * 60 + "\n\n")
                
                # Add enhanced summary
                f.write("\nHIERARCHICAL SESSION SUMMARY:\n")
                f.write("=" * 40 + "\n")
                f.write(f"Total messages: {len(self.message_history)}\n")
                f.write(f"Agent handoffs: {self.handoff_count}\n")
                f.write(f"Technical Lead authority decisions: {self.authority_decisions}\n")
                f.write(f"Task status updates: {self.task_updates}\n\n")
                
                f.write("Agent Participation:\n")
                for agent, stats in sorted(self.agent_stats.items()):
                    if agent != 'tool':
                        icon = self.agent_icons.get(agent, "🤖")
                        
                        if agent == 'technical_lead':
                            role_info = " (AUTHORITY)"
                        elif agent == 'writer':
                            role_info = " (only code writer)"
                        elif agent == 'task_manager':
                            role_info = " (task tracking)"
                        elif agent == 'finalizer':
                            role_info = " (session completion)"
                        else:
                            role_info = " (reports to Tech Lead)"
                            
                        f.write(f"{icon} {agent.upper()}: {stats['messages']} messages{role_info}\n")
                
                tool_messages = sum(1 for _, agent, _, _ in self.message_history if agent == 'tool')
                if tool_messages > 0:
                    f.write(f"🛠️ Tool operations: {tool_messages}\n")
                
                f.write(f"\nHierarchical System Features:\n")
                f.write(f"👑 Technical Lead Authority: Oversight and decision-making\n")
                f.write(f"✅ Single Code Writer: Writer handles ALL code creation and fixing\n")
                f.write(f"📊 Organized Task Tracking: Updates only per Technical Lead directive\n")
                f.write(f"🔄 Hierarchical Handoffs: All agents report to Technical Lead\n")
                f.write(f"🎯 Clear Authority Structure: Technical Lead guides all work\n")
                f.write(f"🏁 Clean Session Completion: Finalizer ends when work is done\n")
            
            print(f"💾 Hierarchical development session log saved to: {log_file}")
            return str(log_file)
            
        except Exception as e:
            print(f"❌ Error saving log: {e}")
            return None