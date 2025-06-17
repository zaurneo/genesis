# conversation_viewer.py - Enhanced for 7-Agent Development System
"""
Enhanced conversation viewer that shows the collaboration between 7 specialized development agents.
Based on the stock analysis project patterns but enhanced for code development workflow.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Union
from pathlib import Path
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage

class CodeDevelopmentViewer:
    """Enhanced live conversation viewer for 7-agent code development system"""
    
    def __init__(self):
        self.colors = {
            'architect': '\033[96m',     # Cyan - Project design
            'writer': '\033[94m',        # Blue - Code writing  
            'executor': '\033[92m',      # Green - Execution
            'analyzer': '\033[93m',      # Yellow - Error analysis
            'fixer': '\033[95m',         # Magenta - Fixing
            'quality': '\033[91m',       # Red - Quality checks
            'docs': '\033[97m',          # White - Documentation
            'user': '\033[90m',          # Gray - User input
            'tool': '\033[35m',          # Purple - Tool operations
            'system': '\033[37m',        # Light gray - System
            'reset': '\033[0m'
        }
        
        self.agent_icons = {
            'architect': '🏗️',
            'writer': '✍️',
            'executor': '⚡',
            'analyzer': '🔍',
            'fixer': '🔧',
            'quality': '✅',
            'docs': '📚',
            'user': '👤',
            'tool': '🛠️',
            'system': '📋'
        }
        
        self.message_history = []
        self.seen_ids = set()
        self.last_handoff = None
        self.agent_stats = {}
        
        # Ensure logs directory exists
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
    def print_header(self):
        """Print enhanced conversation header"""
        print("\n" + "="*100)
        print("🚀 LIVE ENTERPRISE CODE DEVELOPMENT - 7 AI AGENT COLLABORATION")
        print("="*100)
        print("🏗️ ARCHITECT: Cyan  |  ✍️ WRITER: Blue     |  ⚡ EXECUTOR: Green   |  🔍 ANALYZER: Yellow")
        print("🔧 FIXER: Magenta   |  ✅ QUALITY: Red     |  📚 DOCS: White      |  👤 USER: Gray")
        print("🛠️ TOOLS: Purple    |  📋 SYSTEM: Light Gray")
        print("="*100)
        print("WORKFLOW: Architect → Writer → Executor → Analyzer → Fixer → Quality → Docs")
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
    
    def format_and_print(self, agent_name: str, content: str, icon: str = None):
        """Enhanced message formatting with agent-specific styling"""
        if not content or content in ["[]", "", "None"]:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.colors.get(agent_name, self.colors['system'])
        agent_icon = icon or self.agent_icons.get(agent_name, "💬")
        
        # Update agent statistics
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {'messages': 0, 'total_chars': 0}
        self.agent_stats[agent_name]['messages'] += 1
        self.agent_stats[agent_name]['total_chars'] += len(content)
        
        # Split content into lines for better formatting
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                if i == 0:
                    # First line with full header
                    print(f"{color}[{timestamp}] {agent_icon} {agent_name.upper()}: {line}{self.colors['reset']}")
                else:
                    # Continuation lines with proper indentation
                    print(f"{color}{''.ljust(14)}{line}{self.colors['reset']}")
        
        # Record in history with timestamp
        self.message_history.append((timestamp, agent_name, content))
    
    def show_handoff(self, from_agent: str, to_agent: str):
        """Display enhanced handoff visualization"""
        handoff_key = f"{from_agent}->{to_agent}"
        if handoff_key != self.last_handoff:
            self.last_handoff = handoff_key
            
            from_icon = self.agent_icons.get(from_agent, "🤖")
            to_icon = self.agent_icons.get(to_agent, "🤖")
            
            print(f"\n{self.colors['system']}{'─'*80}")
            print(f"🔄 HANDOFF: {from_icon} {from_agent.upper()} → {to_icon} {to_agent.upper()}")
            print(f"{'─'*80}{self.colors['reset']}\n")
    
    def process_message(self, msg, current_agent):
        """Enhanced message processing for 7-agent system"""
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
            
            # Display agent message if there's content
            if content:
                text = self.extract_text_content(content)
                if text and not any(skip in text.lower() for skip in ['transfer_to_', 'successfully transferred']):
                    self.format_and_print(name, text)
            
            # Display tool calls (except transfers)
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get('name', 'unknown')
                    if not tool_name.startswith('transfer_to_'):
                        self.format_and_print(name, f"🔧 Using tool: {tool_name}")
            return
        
        # Handle ToolMessage objects
        if isinstance(msg, ToolMessage):
            content = msg.content
            tool_name = msg.name
            
            # Handle transfers with enhanced visualization
            if tool_name.startswith('transfer_to_') and 'Successfully transferred' in content:
                target = tool_name.replace('transfer_to_', '')
                self.show_handoff(current_agent, target)
                return
            
            # Show enhanced tool results
            if content and self._is_important_tool_result(tool_name, content):
                # Truncate very long outputs for readability
                lines = content.strip().split('\n')
                if len(lines) > 10:
                    result_text = '\n'.join(lines[:10]) + f"\n... ({len(lines)-10} more lines)"
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
                
                # Handle transfers
                if tool_name.startswith('transfer_to_') and 'Successfully transferred' in content:
                    target = tool_name.replace('transfer_to_', '')
                    self.show_handoff(current_agent, target)
                    return
                
                # Show other tool results
                if content and self._is_important_tool_result(tool_name, content):
                    lines = content.strip().split('\n')
                    if len(lines) > 10:
                        result_text = '\n'.join(lines[:10]) + f"\n... ({len(lines)-10} more lines)"
                    else:
                        result_text = content.strip()
                    
                    tool_category = self._categorize_tool(tool_name)
                    self.format_and_print('tool', f"{tool_category} {tool_name}:\n{result_text}")
                return
            
            # Agent messages with name
            agent_names = ['architect', 'writer', 'executor', 'analyzer', 'fixer', 'quality', 'docs']
            if name in agent_names:
                if content:
                    text = self.extract_text_content(content)
                    if text and not any(skip in text.lower() for skip in ['transfer_to_', 'successfully transferred']):
                        self.format_and_print(name, text)
                return
    
    def _is_important_tool_result(self, tool_name: str, content: str) -> bool:
        """Determine if tool result should be displayed"""
        # Always show these tools
        important_tools = [
            'write_code_file', 'execute_python_file', 'monitor_execution',
            'analyze_code_quality', 'check_security', 'generate_tests',
            'create_project_structure', 'install_missing_packages'
        ]
        
        return (tool_name in important_tools or 
                '✅' in content or '❌' in content or 
                'ERROR' in content.upper() or 
                'SUCCESS' in content.upper())
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tools for better display"""
        categories = {
            'write_code_file': '📝',
            'execute_python_file': '⚡',
            'monitor_execution': '📊',
            'analyze_code_quality': '🔍',
            'check_security': '🔒',
            'generate_tests': '🧪',
            'run_tests': '✅',
            'create_project_structure': '🏗️',
            'install_missing_packages': '📦',
            'backup_code': '💾'
        }
        return categories.get(tool_name, '🛠️')
    
    def run(self, graph, initial_message: str):
        """Run enhanced conversation with progress tracking"""
        self.print_header()
        
        # Show initial message
        self.format_and_print('user', initial_message)
        print()
        
        start_time = time.time()
        chunk_count = 0
        
        try:
            # Stream the conversation with progress tracking
            for chunk in graph.stream({"messages": [HumanMessage(content=initial_message)]}):
                chunk_count += 1
                
                # Process each agent's messages
                for agent_name, data in chunk.items():
                    messages = data.get('messages', [])
                    
                    # Process all messages
                    for msg in messages:
                        self.process_message(msg, agent_name)
                
                # Small delay for readability
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print(f"\n{self.colors['system']}⏹️ Development session interrupted by user{self.colors['reset']}")
        except Exception as e:
            print(f"\n{self.colors['tool']}❌ System Error: {e}{self.colors['reset']}")
            import traceback
            traceback.print_exc()
        
        execution_time = time.time() - start_time
        print(f"\n{self.colors['system']}✅ Development session completed in {execution_time:.1f} seconds!{self.colors['reset']}")
        self.print_enhanced_summary()
    
    def print_enhanced_summary(self):
        """Print comprehensive session summary"""
        print(f"\n{self.colors['system']}📊 DEVELOPMENT SESSION SUMMARY:")
        print("=" * 70)
        
        # Agent participation
        print("🤖 Agent Participation:")
        for agent, stats in sorted(self.agent_stats.items()):
            if agent != 'tool':
                icon = self.agent_icons.get(agent, "🤖")
                color = self.colors.get(agent, self.colors['system'])
                avg_length = stats['total_chars'] / max(1, stats['messages'])
                print(f"{color}  {icon} {agent.upper()}: {stats['messages']} messages "
                      f"({avg_length:.0f} avg chars){self.colors['reset']}")
        
        # Tool usage
        tool_messages = sum(1 for _, agent, _ in self.message_history if agent == 'tool')
        if tool_messages > 0:
            print(f"🛠️ Tool operations: {tool_messages}")
        
        print(f"📈 Total exchanges: {len(self.message_history)}")
        print(f"{self.colors['reset']}")
    
    def save_log(self, filename=None):
        """Save enhanced development session log"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"development_session_{timestamp}.txt"
        
        log_file = self.logs_dir / filename
        
        if not self.message_history:
            print("No messages to save.")
            return str(log_file)
            
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("ENTERPRISE CODE DEVELOPMENT SESSION LOG\n")
                f.write("7-Agent Collaborative Development System\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("AGENT WORKFLOW:\n")
                f.write("🏗️ Architect → ✍️ Writer → ⚡ Executor → 🔍 Analyzer → 🔧 Fixer → ✅ Quality → 📚 Docs\n\n")
                
                for timestamp, agent, message in self.message_history:
                    icon = self.agent_icons.get(agent, "🤖")
                    f.write(f"[{timestamp}] {icon} {agent.upper()}:\n")
                    f.write(f"{message}\n")
                    f.write("-" * 60 + "\n\n")
                
                # Add enhanced summary
                f.write("\nSESSION SUMMARY:\n")
                f.write("=" * 40 + "\n")
                f.write(f"Total messages: {len(self.message_history)}\n\n")
                
                f.write("Agent Participation:\n")
                for agent, stats in sorted(self.agent_stats.items()):
                    if agent != 'tool':
                        icon = self.agent_icons.get(agent, "🤖")
                        f.write(f"{icon} {agent.upper()}: {stats['messages']} messages\n")
                
                tool_messages = sum(1 for _, agent, _ in self.message_history if agent == 'tool')
                if tool_messages > 0:
                    f.write(f"🛠️ Tool operations: {tool_messages}\n")
            
            print(f"💾 Development session log saved to: {log_file}")
            return str(log_file)
            
        except Exception as e:
            print(f"❌ Error saving log: {e}")
            return None