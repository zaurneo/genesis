from autogen_agentchat.teams import RoundRobinGroupChat

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a task that can be assigned to agents."""
    id: str
    name: str
    description: str
    assigned_to: str  # Agent type/role
    status: str = "pending"  # pending, in_progress, completed
    dependencies: List[str] = None  # Task IDs this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class TurnLog:
    """Log entry for each turn."""
    turn_number: int
    agent_name: str
    task_name: str
    task_id: str
    handoff_from: Optional[str] = None
    handoff_to: Optional[str] = None
    handoff_task_description: Optional[str] = None
    handoff_status: bool = True
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FlexibleHandoffGroupChat(RoundRobinGroupChat):
    """
    Enhanced group chat where agents can hand off tasks to each other,
    with comprehensive task management and logging.
    """
    
    def __init__(self, owner, agents, max_agent_turns=5, report_agent=None, tasks=None, 
                 format_agent_output=True, show_agent_messages=True, **kwargs):
        # Validation
        self._validate_inputs(owner, agents)
        
        self.owner = owner
        
        # Remove duplicates from agents while preserving order
        seen = set()
        unique_agents = []
        for agent in agents:
            agent_name = getattr(agent, 'name', str(agent))
            if agent_name not in seen:
                seen.add(agent_name)
                unique_agents.append(agent)
        self.agents = unique_agents
        
        self.max_agent_turns = max_agent_turns
        self.report_agent = report_agent
        self.format_agent_output = format_agent_output
        self.show_agent_messages = show_agent_messages
        
        # Task management
        self.tasks = tasks if tasks else []
        self.task_assignments = {}  # agent -> current task
        self.task_history = []  # List of completed tasks
        
        # Tracking state
        self.turns_since_owner = 0
        self.last_speaker = None
        self.conversation_history = []
        self.owner_intervention_count = 0
        self.turn_logs = []  # Detailed turn logs
        
        # Enhanced features
        self._next_speaker_override = None
        self._force_owner_next = False
        self._current_handoff_status = True  # Default to True
        self._handoff_info = None  # Store handoff details for next turn
        self._output_mode = 'formatted'  # Default output mode
        
        # Build participants - ensure no duplicates
        participants = [owner]
        participant_names = {getattr(owner, 'name', str(owner))}
        
        for agent in self.agents:
            agent_name = getattr(agent, 'name', str(agent))
            if agent_name not in participant_names:
                participants.append(agent)
                participant_names.add(agent_name)
        
        # Add report agent if not already in participants
        if self.report_agent:
            report_name = getattr(self.report_agent, 'name', str(self.report_agent))
            if report_name not in participant_names:
                participants.append(self.report_agent)
                participant_names.add(report_name)
        
        super().__init__(participants, **kwargs)
        self._setup_speaker_selection()

    def _validate_inputs(self, owner, agents):
        """Validate constructor inputs."""
        if not agents:
            raise ValueError("At least one agent is required")
        
        if not hasattr(owner, 'name'):
            raise ValueError("Owner must have a 'name' attribute")
        
        # Check for owner in agents by comparing names
        owner_name = getattr(owner, 'name', str(owner))
        agent_names = [getattr(agent, 'name', str(agent)) for agent in agents]
        
        if owner_name in agent_names:
            raise ValueError("Owner cannot be included in agents list")
        
        # Check for duplicate agent names
        if len(agent_names) != len(set(agent_names)):
            raise ValueError("Agent names must be unique")
        
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents for handoff functionality")

    def _setup_speaker_selection(self):
        """Setup speaker selection method - handles different base class APIs."""
        # Try different possible method names the base class might use
        possible_methods = ['speaker_selection_method', 'select_speaker', 'next_speaker_selector']
        
        for method_name in possible_methods:
            if hasattr(self, method_name):
                setattr(self, method_name, self._select_next_speaker)
                return
        
        # Fallback: set the most common one
        self.speaker_selection_method = self._select_next_speaker

    def set_tasks(self, tasks: List[Task]):
        """Set the list of tasks for the conversation."""
        self.tasks = tasks
        logger.info(f"Owner has set {len(tasks)} tasks for the team")
        self._display_all_tasks()

    def _display_all_tasks(self):
        """Display all tasks shared by the owner."""
        print("\n" + "="*60)
        print("📋 ALL TASKS FROM OWNER:")
        print("="*60)
        for task in self.tasks:
            status_icon = "✅" if task.status == "completed" else "🔄" if task.status == "in_progress" else "⏳"
            print(f"{status_icon} Task {task.id}: {task.name}")
            print(f"   Description: {task.description}")
            print(f"   Assigned to: {task.assigned_to}")
            print(f"   Status: {task.status}")
            if task.dependencies:
                print(f"   Dependencies: {', '.join(task.dependencies)}")
        print("="*60 + "\n")

    def _get_available_tasks_for_agent(self, agent) -> List[Task]:
        """Get tasks that an agent can work on."""
        agent_role = getattr(agent, 'role', getattr(agent, 'name', str(agent)))
        available_tasks = []
        
        for task in self.tasks:
            # Check if task is for this agent type and not completed
            if task.assigned_to.lower() in agent_role.lower() and task.status != "completed":
                # Check if dependencies are met
                deps_met = all(
                    any(t.id == dep_id and t.status == "completed" for t in self.tasks)
                    for dep_id in task.dependencies
                )
                if deps_met:
                    available_tasks.append(task)
        
        return available_tasks

    def _start_turn(self, agent, handoff_from=None):
        """Log and display information at the start of a turn."""
        turn_number = len(self.turn_logs) + 1
        
        # Get current task for agent
        current_task = self.task_assignments.get(agent)
        if not current_task:
            # Assign a new task if agent doesn't have one
            available_tasks = self._get_available_tasks_for_agent(agent)
            if available_tasks:
                current_task = available_tasks[0]
                self.task_assignments[agent] = current_task
                current_task.status = "in_progress"
        
        task_name = current_task.name if current_task else "No task assigned"
        task_id = current_task.id if current_task else "N/A"
        
        # Display start of turn info
        print("\n" + "-"*60)
        print(f"🔄 TURN {turn_number} STARTING")
        print(f"👤 Agent: {getattr(agent, 'name', str(agent))}")
        print(f"📌 Task: {task_name} (ID: {task_id})")
        if handoff_from:
            print(f"🤝 Handed off from: {handoff_from}")
        print("-"*60)
        
        return turn_number, task_name, task_id

    def _end_turn(self, agent, turn_number, task_name, task_id, handoff_to=None, 
                  handoff_task=None, handoff_status=True):
        """Log and display information at the end of a turn."""
        agent_name = getattr(agent, 'name', str(agent))
        
        # Create turn log
        turn_log = TurnLog(
            turn_number=turn_number,
            agent_name=agent_name,
            task_name=task_name,
            task_id=task_id,
            handoff_from=self._handoff_info.get('from') if self._handoff_info else None,
            handoff_to=getattr(handoff_to, 'name', str(handoff_to)) if handoff_to else None,
            handoff_task_description=handoff_task.description if handoff_task else None,
            handoff_status=handoff_status
        )
        
        self.turn_logs.append(turn_log)
        
        # Display end of turn info
        print("\n" + "-"*30)
        print(f"✅ TURN {turn_number} COMPLETED")
        print(f"👤 Agent: {agent_name}")
        print(f"📌 Task: {task_name}")
        if handoff_to:
            print(f"🤝 Handing off to: {getattr(handoff_to, 'name', str(handoff_to))}")
            print(f"📋 Handoff task: {handoff_task.description if handoff_task else 'Continuation'}")
        print(f"🚦 Handoff Status: {'✅ Success' if handoff_status else '❌ Failed'}")
        print("-"*60 + "\n")
        
        # Save to conversation history
        self.conversation_history.append({
            'turn': turn_number,
            'log': turn_log
        })
        
        # Update current handoff status for intervention logic
        self._current_handoff_status = handoff_status
        
        # Log to file/logger
        logger.info(f"Turn {turn_number} - Agent: {agent_name}, Task: {task_name}, "
                   f"Handoff: {handoff_to is not None}, Status: {handoff_status}")

    def _select_next_speaker(self, last_speaker=None, messages=None):
        """Enhanced speaker selection with handoff status check."""
        # Track the last speaker
        if last_speaker:
            self.last_speaker = last_speaker
            self._update_conversation_tracking(last_speaker, messages)
        
        # Format and display the last message if available
        if messages and len(messages) > 0:
            self._format_agent_output(last_speaker, messages[-1])
        
        # Determine next speaker based on current state
        next_speaker = self._determine_next_speaker()
        
        # Log the selection
        self._log_speaker_selection(next_speaker)
        
        # If next speaker is an agent, handle turn start/end
        if next_speaker != self.owner:
            self._simulate_agent_turn(next_speaker)
        else:
            self._handle_owner_turn()
        
        return next_speaker

    def _determine_next_speaker(self):
        """Enhanced logic with handoff status check."""
        
        # Priority 1: Check handoff status from last turn
        if not self._current_handoff_status and self.last_speaker in self.agents:
            logger.info("Handoff failed - Owner intervention required")
            self.owner_intervention_count += 1
            self.turns_since_owner = 0
            return self.owner
        
        # Priority 2: Check for forced owner intervention
        if self._force_owner_next:
            self._force_owner_next = False
            self.turns_since_owner = 0
            return self.owner
        
        # Priority 3: Check for explicit handoff override
        if self._next_speaker_override:
            next_speaker = self._next_speaker_override
            self._next_speaker_override = None
            return next_speaker
        
        # Case 1: Start of conversation - owner begins
        if self.last_speaker is None:
            self.turns_since_owner = 0
            return self.owner
        
        # Case 2: Owner just spoke - any agent can respond
        if self.last_speaker == self.owner:
            self.turns_since_owner = 0
            return self._select_agent_for_task()
        
        # Case 3: Agent spoke - check if owner intervention needed
        if self.last_speaker in self.agents:
            # Force owner intervention if max turns reached
            if self.turns_since_owner >= self.max_agent_turns:
                self.owner_intervention_count += 1
                self.turns_since_owner = 0
                return self.owner
            
            # Otherwise, allow agent-to-agent handoff
            return self._select_next_agent()
        
        # Case 4: Report agent spoke - owner should respond
        if self.report_agent and self.last_speaker == self.report_agent:
            self.turns_since_owner = 0
            return self.owner
        
        # Default fallback
        return self.owner

    def _select_agent_for_task(self) -> Any:
        """Select an agent based on available tasks."""
        # Find agents with available tasks
        for agent in self.agents:
            available_tasks = self._get_available_tasks_for_agent(agent)
            if available_tasks:
                return agent
        
        # Default to first agent if no specific tasks
        return self.agents[0]

    def _select_next_agent(self):
        """Select the next agent when allowing agent-to-agent handoff."""
        # Simple round-robin among agents (can be enhanced with smarter logic)
        if self.last_speaker in self.agents:
            current_idx = self.agents.index(self.last_speaker)
            next_idx = (current_idx + 1) % len(self.agents)
            return self.agents[next_idx]
        else:
            # Fallback to first agent
            return self.agents[0]

    def _simulate_agent_turn(self, agent):
        """Simulate an agent's turn with task management."""
        handoff_from = self._handoff_info.get('from') if self._handoff_info else None
        
        # Start of turn
        turn_number, task_name, task_id = self._start_turn(agent, handoff_from)
        
        # Simulate agent work (in real implementation, agent would do actual work here)
        # For now, we'll simulate some decisions
        
        # Decide if agent will handoff
        will_handoff = self._should_agent_handoff(agent)
        handoff_to = None
        handoff_task = None
        handoff_status = True
        
        if will_handoff:
            # Find suitable agent for handoff
            handoff_to, handoff_task = self._find_handoff_target(agent)
            if handoff_to:
                success, msg = self.request_handoff(agent, handoff_to, 
                                                  reason=f"Completed my part, passing {handoff_task.name}")
                handoff_status = success
            else:
                handoff_status = False  # No suitable handoff target
        
        # End of turn
        self._end_turn(agent, turn_number, task_name, task_id, 
                      handoff_to, handoff_task, handoff_status)
        
        # Store handoff info for next turn
        if handoff_to and handoff_status:
            self._handoff_info = {
                'from': getattr(agent, 'name', str(agent)),
                'to': handoff_to,
                'task': handoff_task
            }
        else:
            self._handoff_info = None

    def _should_agent_handoff(self, agent) -> bool:
        """Determine if agent should handoff (simplified logic)."""
        # In real implementation, this would be based on task completion
        # For demo, use a simple probability
        import random
        return random.random() > 0.3  # 70% chance of handoff

    def _find_handoff_target(self, current_agent) -> Tuple[Optional[Any], Optional[Task]]:
        """Find suitable agent and task for handoff."""
        current_task = self.task_assignments.get(current_agent)
        
        # Look for agents who can handle related tasks
        for agent in self.agents:
            if agent == current_agent:
                continue
            
            available_tasks = self._get_available_tasks_for_agent(agent)
            if available_tasks:
                # Prefer tasks that depend on current task
                for task in available_tasks:
                    if current_task and current_task.id in task.dependencies:
                        return agent, task
                
                # Otherwise, any available task
                return agent, available_tasks[0]
        
        return None, None

    def _format_agent_output(self, speaker, message):
        """Format and display agent output in a more readable way."""
        if not self.show_agent_messages:
            return
            
        if not message or not hasattr(message, 'content'):
            return
        
        agent_name = getattr(speaker, 'name', str(speaker))
        content = message.content
        
        # Handle summary mode
        if self._output_mode == 'summary':
            summary = self._create_message_summary(content)
            print(f"\n💬 {agent_name} (summary): {summary}\n")
            return
        
        # Skip formatting for very short messages
        if len(content) < 100:
            print(f"\n💬 {agent_name}: {content}\n")
            return
        
        if not self.format_agent_output or self._output_mode == 'minimal':
            # Simple output without formatting
            print(f"\n💬 {agent_name}:")
            print(content)
            print()
            return
        
        # Formatted output
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║ 💬 {agent_name:<52} ║")
        print("╠" + "═"*58 + "╣")
        
        # Split content into sections
        lines = content.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # Detect section headers (lines ending with **)
            if line.strip().endswith('**'):
                print("║" + " "*58 + "║")
                print(f"║ 📌 {line.strip():<52} ║")
                print("║" + " "*58 + "║")
            # Detect bullet points or numbered items
            elif line.strip().startswith(('- ', '* ', '1.', '2.', '3.')):
                # Wrap long bullet points
                wrapped = self._wrap_text(line.strip(), 54)
                for i, wrapped_line in enumerate(wrapped):
                    if i == 0:
                        print(f"║   {wrapped_line:<54} ║")
                    else:
                        print(f"║     {wrapped_line:<52} ║")
            # Regular text
            else:
                wrapped = self._wrap_text(line.strip(), 56)
                for wrapped_line in wrapped:
                    print(f"║ {wrapped_line:<56} ║")
        
        print("╚" + "═"*58 + "╝")
        print()
    
    def _create_message_summary(self, content, max_length=200):
        """Create a summary of long agent messages."""
        if len(content) <= max_length:
            return content
        
        # Extract key points
        lines = content.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Look for section headers or important points
            if any(marker in line for marker in ['**', 'Step', 'Tool:', 'Objective:', 'Required:']):
                key_points.append(line.replace('*', '').strip())
        
        if key_points:
            summary = "Key points: " + "; ".join(key_points[:3])
            if len(key_points) > 3:
                summary += f" (+ {len(key_points) - 3} more points)"
        else:
            # Fallback to first few sentences
            summary = content[:max_length] + "..."
        
        return summary
    
    def set_output_mode(self, mode='full'):
        """Set output mode: 'full', 'formatted', 'summary', or 'minimal'."""
        if mode == 'full':
            self.format_agent_output = True
            self.show_agent_messages = True
            self._output_mode = 'full'
        elif mode == 'formatted':
            self.format_agent_output = True
            self.show_agent_messages = True
            self._output_mode = 'formatted'
        elif mode == 'summary':
            self.format_agent_output = False
            self.show_agent_messages = True
            self._output_mode = 'summary'
        elif mode == 'minimal':
            self.format_agent_output = False
            self.show_agent_messages = False
            self._output_mode = 'minimal'
        else:
            raise ValueError("Mode must be 'full', 'formatted', 'summary', or 'minimal'")
        
        """Wrap text to specified width."""
    def _wrap_text(self, text: str, width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current_line: List[str] = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > width:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else ['']
    
    def _handle_owner_turn(self):
        """Handle owner's turn - typically providing guidance or new tasks."""
        print("\n" + "="*60)
        print("👑 OWNER INTERVENTION")
        print("="*60)
        
        if not self._current_handoff_status:
            print("⚠️  Reason: Handoff failed in previous turn")
        elif self.turns_since_owner >= self.max_agent_turns:
            print("⚠️  Reason: Maximum agent turns reached")
        else:
            print("📋 Providing guidance and task updates")
        
        # Display current task status
        print("\n📊 Current Task Status:")
        for task in self.tasks:
            status_icon = "✅" if task.status == "completed" else "🔄" if task.status == "in_progress" else "⏳"
            print(f"  {status_icon} {task.id}: {task.name} ({task.status})")
        
        print("="*60 + "\n")

    def request_handoff(self, from_agent, to_agent, reason=None) -> Tuple[bool, str]:
        """Enhanced handoff with task awareness."""
        if from_agent not in self.agents or to_agent not in self.agents:
            return False, "Both agents must be in the agent list"
        
        if from_agent == to_agent:
            return False, "Cannot handoff to self"
        
        # Check if handoff is allowed
        if self.turns_since_owner >= self.max_agent_turns:
            return False, "Owner intervention required - max agent turns reached"
        
        # Check if target agent has capacity (available tasks)
        available_tasks = self._get_available_tasks_for_agent(to_agent)
        if not available_tasks:
            return False, f"{getattr(to_agent, 'name', str(to_agent))} has no available tasks"
        
        # Set the next speaker override
        self._next_speaker_override = to_agent
        
        # Record the handoff
        self.conversation_history.append({
            'type': 'handoff',
            'from': from_agent,
            'to': to_agent,
            'reason': reason,
            'turn_number': len(self.conversation_history) + 1
        })
        
        return True, f"Handoff approved: {getattr(from_agent, 'name', str(from_agent))} → {getattr(to_agent, 'name', str(to_agent))}"

    def force_owner_intervention(self, reason=None):
        """Force an immediate owner intervention."""
        self._force_owner_next = True
        
        self.conversation_history.append({
            'type': 'forced_intervention',
            'reason': reason,
            'turn_number': len(self.conversation_history) + 1
        })
        return self.owner

    def _update_conversation_tracking(self, speaker, messages=None):
        """Update conversation tracking and history."""
        # Increment turns_since_owner AFTER determining speaker was an agent
        if speaker in self.agents:
            self.turns_since_owner += 1
        elif speaker == self.owner:
            self.turns_since_owner = 0
            
        entry = {
            'speaker': speaker,
            'turn_number': len(self.conversation_history) + 1,
            'turns_since_owner': self.turns_since_owner,
            'owner_intervention': speaker == self.owner and len(self.conversation_history) > 0 and self.conversation_history[-1].get('speaker') in self.agents
        }
        
        # Optionally include message content
        if messages:
            entry['messages'] = messages
            
        self.conversation_history.append(entry)

    def _log_speaker_selection(self, next_speaker):
        """Log speaker selection for debugging."""
        status = f"Turn {len(self.conversation_history) + 1}: {getattr(next_speaker, 'name', str(next_speaker))}"
        status += f" (Turns since owner: {self.turns_since_owner}"
        
        if next_speaker == self.owner and self.turns_since_owner >= self.max_agent_turns:
            status += " - INTERVENTION REQUIRED"
        elif next_speaker == self.owner:
            status += " - Owner guidance"
        else:
            status += " - Agent handoff"
        
        status += ")"
        logger.debug(status)

    # === CONFIGURATION METHODS ===
    
    def set_max_agent_turns(self, max_turns):
        """Update the maximum agent turns before owner intervention."""
        if max_turns < 1:
            raise ValueError("Max agent turns must be at least 1")
        self.max_agent_turns = max_turns

    def get_max_agent_turns(self):
        """Get current max agent turns setting."""
        return self.max_agent_turns

    # === MONITORING AND ANALYTICS ===
    
    def get_conversation_stats(self):
        """Get detailed conversation statistics."""
        total_turns = len(self.conversation_history)
        owner_turns = sum(1 for entry in self.conversation_history 
                         if entry.get('speaker') == self.owner)
        agent_turns = total_turns - owner_turns
        
        return {
            'total_turns': total_turns,
            'owner_turns': owner_turns,
            'agent_turns': agent_turns,
            'turns_since_last_owner': self.turns_since_owner,
            'owner_interventions': self.owner_intervention_count,
            'current_max_agent_turns': self.max_agent_turns,
            'intervention_needed': self.turns_since_owner >= self.max_agent_turns,
            'last_speaker': getattr(self.last_speaker, 'name', str(self.last_speaker)) if self.last_speaker else None
        }

    def get_handoff_history(self):
        """Get history of agent handoffs."""
        return [entry for entry in self.conversation_history 
                if entry.get('type') == 'handoff']

    def get_intervention_history(self):
        """Get history of owner interventions."""
        interventions = []
        for entry in self.conversation_history:
            if entry.get('owner_intervention') or entry.get('type') == 'forced_intervention':
                interventions.append(entry)
        return interventions

    def reset_conversation(self):
        """Reset conversation state."""
        self.turns_since_owner = 0
        self.last_speaker = None
        self.conversation_history = []
        self.owner_intervention_count = 0
        self._next_speaker_override = None
        self._force_owner_next = False

    def get_conversation_flow_summary(self):
        """Get a readable summary of the conversation flow."""
        if not self.conversation_history:
            return "No conversation history"
        
        summary = []
        for entry in self.conversation_history:
            if entry.get('type') == 'handoff':
                from_name = getattr(entry['from'], 'name', str(entry['from']))
                to_name = getattr(entry['to'], 'name', str(entry['to']))
                summary.append(f"[Handoff: {from_name} → {to_name}]")
            elif entry.get('type') == 'forced_intervention':
                summary.append("[Forced Intervention]")
            elif entry.get('speaker'):
                speaker_name = getattr(entry['speaker'], 'name', str(entry['speaker']))
                if entry.get('owner_intervention'):
                    summary.append(f"{speaker_name} (INTERVENTION)")
                else:
                    summary.append(speaker_name)
        
        return " → ".join(summary)

    def get_turn_logs_summary(self) -> str:
        """Get a summary of all turn logs."""
        if not self.turn_logs:
            return "No turns recorded yet"
        
        summary = ["📊 TURN LOGS SUMMARY:", "="*60]
        
        for log in self.turn_logs:
            summary.append(f"\nTurn {log.turn_number}:")
            summary.append(f"  Agent: {log.agent_name}")
            summary.append(f"  Task: {log.task_name}")
            if log.handoff_from:
                summary.append(f"  Received from: {log.handoff_from}")
            if log.handoff_to:
                summary.append(f"  Handed off to: {log.handoff_to}")
            summary.append(f"  Handoff Status: {'✅' if log.handoff_status else '❌'}")
        
        return "\n".join(summary)

    def export_turn_logs(self, filename: str = "turn_logs.txt"):
        """Export turn logs to a file."""
        with open(filename, 'w') as f:
            f.write(self.get_turn_logs_summary())
            f.write("\n\n" + "="*60 + "\n")
            f.write("DETAILED LOGS:\n")
            f.write("="*60 + "\n\n")
            
            for log in self.turn_logs:
                f.write(f"Turn {log.turn_number} - {log.timestamp}\n")
                f.write(f"  Agent: {log.agent_name}\n")
                f.write(f"  Task: {log.task_name} (ID: {log.task_id})\n")
                if log.handoff_from:
                    f.write(f"  Handoff from: {log.handoff_from}\n")
                if log.handoff_to:
                    f.write(f"  Handoff to: {log.handoff_to}\n")
                    f.write(f"  Handoff task: {log.handoff_task_description}\n")
                f.write(f"  Handoff status: {log.handoff_status}\n")
                f.write("-"*40 + "\n")

