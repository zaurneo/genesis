from autogen_agentchat.teams import RoundRobinGroupChat

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# to get rid of autogen logs
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

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

@dataclass
class AgentCapability:
    """Represents an agent's capabilities and role."""
    name: str
    primary_role: str
    capabilities: List[str]
    expertise: List[str]
    can_research: bool = False
    can_write: bool = False
    can_analyze: bool = False
    can_code: bool = False
    can_review: bool = False
    can_manage: bool = False

class FlexibleHandoffGroupChat(RoundRobinGroupChat):
    """
    Enhanced group chat where agents can hand off tasks to each other,
    with comprehensive task management, logging, agent awareness, and capability discovery.
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
        
        # Agent assignment and capability tracking
        self._active_assignments = {}  # agent_name -> assigned_task_info
        self._agent_capabilities = {}  # agent_name -> AgentCapability
        self._assignment_required = True  # Block agent participation without assignment
        
        # Discover agent capabilities
        self._discover_agent_capabilities()
        
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
        
        # Display team composition instead of injecting into system messages
        self.display_team_composition()
        
        # Verify setup
        print(f"\n🔍 DEBUGGING: Team setup verification:")
        print(f"Available agents: {len(self.get_available_agents())}")
        print(f"Agent capabilities discovered: {len(self._agent_capabilities)}")
        print(f"✅ Team setup completed without system message modification")
        
        super().__init__(participants, **kwargs)
        self._setup_speaker_selection()

    def _discover_agent_capabilities(self):
        """Discover capabilities of all agents based on their names and system messages."""
        
        capability_patterns = {
            'research': ['research', 'gather', 'investigate', 'analyze', 'data', 'information'],
            'writing': ['write', 'draft', 'compose', 'document', 'report', 'generate'],
            'analysis': ['analyze', 'evaluate', 'assess', 'review', 'examine'],
            'coding': ['code', 'program', 'develop', 'script', 'python', 'execute'],
            'review': ['review', 'check', 'validate', 'verify', 'quality'],
            'management': ['manage', 'coordinate', 'strategy', 'plan', 'organize']
        }
        
        for agent in self.agents:
            agent_name = getattr(agent, 'name', str(agent))
            
            # Try different possible system message attributes
            system_msg = ""
            if hasattr(agent, 'system_message'):
                system_msg = agent.system_message
            elif hasattr(agent, '_system_messages'):
                system_msg = str(agent._system_messages)
            elif hasattr(agent, '_system_message'):
                system_msg = agent._system_message
            
            system_msg = system_msg.lower() if system_msg else ""
            
            # Extract capabilities based on agent name and system message
            capabilities = []
            expertise = []
            
            # Analyze agent name
            name_lower = agent_name.lower()
            
            # Analyze based on patterns
            can_research = any(pattern in name_lower or pattern in system_msg 
                             for pattern in capability_patterns['research'])
            can_write = any(pattern in name_lower or pattern in system_msg 
                           for pattern in capability_patterns['writing'])
            can_analyze = any(pattern in name_lower or pattern in system_msg 
                            for pattern in capability_patterns['analysis'])
            can_code = any(pattern in name_lower or pattern in system_msg 
                          for pattern in capability_patterns['coding'])
            can_review = any(pattern in name_lower or pattern in system_msg 
                           for pattern in capability_patterns['review'])
            can_manage = any(pattern in name_lower or pattern in system_msg 
                           for pattern in capability_patterns['management'])
            
            # Determine primary role and capabilities
            if 'data' in name_lower or 'engineer' in name_lower:
                primary_role = "Data Specialist"
                capabilities.extend(['data processing', 'analysis', 'visualization'])
            elif 'model' in name_lower or 'executor' in name_lower:
                primary_role = "Model Specialist"
                capabilities.extend(['model training', 'predictions', 'execution'])
            elif 'test' in name_lower:
                primary_role = "Testing Specialist"
                capabilities.extend(['validation', 'testing', 'quality assurance'])
            elif 'quality' in name_lower:
                primary_role = "Quality Specialist"
                capabilities.extend(['quality control', 'review', 'compliance'])
            elif 'report' in name_lower:
                primary_role = "Report Specialist"
                capabilities.extend(['report generation', 'documentation', 'insights'])
            elif 'python' in name_lower or 'code' in name_lower:
                primary_role = "Programming Specialist"
                capabilities.extend(['programming', 'code development', 'debugging'])
            elif 'innovative' in name_lower or 'creative' in name_lower:
                primary_role = "Innovation Specialist"
                capabilities.extend(['creative thinking', 'problem solving', 'ideation'])
            elif 'strategy' in name_lower or 'project' in name_lower:
                primary_role = "Strategy Specialist"
                capabilities.extend(['planning', 'coordination', 'management'])
            elif 'efficiency' in name_lower:
                primary_role = "Efficiency Specialist"
                capabilities.extend(['optimization', 'process improvement', 'analysis'])
            elif 'emotional' in name_lower:
                primary_role = "Emotional Intelligence Specialist"
                capabilities.extend(['team dynamics', 'communication', 'conflict resolution'])
            elif 'first' in name_lower and 'principles' in name_lower:
                primary_role = "First Principles Specialist"
                capabilities.extend(['problem deconstruction', 'fundamental analysis', 'systematic thinking'])
            elif 'agi' in name_lower or 'gestalt' in name_lower:
                primary_role = "General Intelligence Specialist"
                capabilities.extend(['knowledge synthesis', 'multi-domain reasoning', 'coordination'])
            elif 'awareness' in name_lower:
                primary_role = "Process Awareness Specialist"
                capabilities.extend(['team guidance', 'process optimization', 'meta-analysis'])
            elif 'task' in name_lower and 'comprehension' in name_lower:
                primary_role = "Task Analysis Specialist"
                capabilities.extend(['task breakdown', 'requirement analysis', 'goal clarification'])
            elif 'history' in name_lower and 'review' in name_lower:
                primary_role = "Progress Review Specialist"
                capabilities.extend(['progress tracking', 'history analysis', 'milestone review'])
            else:
                primary_role = "General Agent"
                capabilities.extend(['general assistance', 'task execution'])
            
            # Add specific capabilities based on analysis
            if can_research:
                capabilities.append('research and information gathering')
            if can_write:
                capabilities.append('writing and documentation')
            if can_analyze:
                capabilities.append('data analysis and evaluation')
            if can_code:
                capabilities.append('programming and code execution')
            if can_review:
                capabilities.append('review and quality assurance')
            if can_manage:
                capabilities.append('project management and coordination')
            
            # Remove duplicates
            capabilities = list(set(capabilities))
            
            # Create capability object
            agent_capability = AgentCapability(
                name=agent_name,
                primary_role=primary_role,
                capabilities=capabilities,
                expertise=capabilities[:3],  # Top 3 as expertise
                can_research=can_research,
                can_write=can_write,
                can_analyze=can_analyze,
                can_code=can_code,
                can_review=can_review,
                can_manage=can_manage
            )
            
            self._agent_capabilities[agent_name] = agent_capability

    def _generate_agent_awareness_prompt(self, for_agent_name: str = None) -> str:
        """Generate dynamic agent awareness information for any agent."""
        agent_names = [getattr(agent, 'name', str(agent)) for agent in self.agents]
        
        awareness_info = f"""

🚨 CRITICAL - TEAM COMPOSITION & CAPABILITY AWARENESS 🚨
=========================================================
TEAM SIZE: {len(self.agents)} agents + PROJECT OWNER
PROJECT OWNER: {getattr(self.owner, 'name', str(self.owner))}
"""
        
        if self.report_agent:
            awareness_info += f"REPORT AGENT: {getattr(self.report_agent, 'name', str(self.report_agent))}\n"
        
        awareness_info += "\n🤖 AVAILABLE TEAM MEMBERS & THEIR CAPABILITIES:\n"
        awareness_info += "=" * 50 + "\n"
        
        # Add detailed capability information
        for agent_name in agent_names:
            if agent_name in self._agent_capabilities:
                cap = self._agent_capabilities[agent_name]
                awareness_info += f"\n🔹 {agent_name}:\n"
                awareness_info += f"   Primary Role: {cap.primary_role}\n"
                awareness_info += f"   Capabilities: {', '.join(cap.capabilities)}\n"
                
                # Add specific capability flags
                flags = []
                if cap.can_research: flags.append("📊 Research")
                if cap.can_write: flags.append("✍️ Writing") 
                if cap.can_analyze: flags.append("🔍 Analysis")
                if cap.can_code: flags.append("💻 Coding")
                if cap.can_review: flags.append("✅ Review")
                if cap.can_manage: flags.append("📋 Management")
                
                if flags:
                    awareness_info += f"   Specialties: {' | '.join(flags)}\n"
            else:
                awareness_info += f"\n🔹 {agent_name}: (General Agent)\n"
        
        awareness_info += f"""
=========================================================

⚠️  CRITICAL TASK ASSIGNMENT & CONVERSATION RULES ⚠️
- ONLY agents listed above exist in your team
- NEVER create tasks for fictional agents like "Research Agent", "Writing Agent"
- Use EXACT agent names from the list above
- Agents can ONLY participate when:
  ✓ Explicitly assigned a task by Project Owner
  ✓ Explicitly handed off to by another agent
  ✓ Called upon by owner for intervention
- Format for assignments: <exact_agent_name> : <task_description>

AGENT DISCOVERY TOOLS AVAILABLE:
- Use "show team capabilities" to see detailed agent abilities
- Use "find agent for [task type]" to get recommendations
- Use "validate assignment [agent_name]" to check if agent exists

HANDOFF RULES:
- Handoffs only allowed to agents listed above
- Max {self.max_agent_turns} agent turns before owner intervention
- Handoff format: "HANDOFF TO: <exact_agent_name>"

=========================================================
"""
        return awareness_info

    def _inject_agent_awareness_for_all(self):
        """DISABLED: Inject agent awareness into ALL agents to avoid autogen compatibility issues."""
        # NOTE: System message injection disabled due to autogen compatibility issues
        # Team awareness is now provided through:
        # 1. display_team_composition() method  
        # 2. Assignment tracking and validation
        # 3. Agent discovery tools
        print("🛡️  Agent awareness injection disabled for compatibility - using alternative methods")
        pass

    def _inject_agent_awareness(self, agent, agent_name: str):
        """DISABLED: Inject agent awareness to avoid autogen compatibility issues."""
        # NOTE: Direct system message modification causes autogen framework errors
        # Alternative: Team awareness through display methods and tools
        pass

    def assign_task_to_agent(self, agent_name: str, task_description: str, task_id: str = None) -> Tuple[bool, str]:
        """Formally assign a task to an agent - enables them to participate."""
        
        # Validate agent exists
        if not self.validate_agent_assignment(agent_name):
            return False, f"Agent '{agent_name}' does not exist in team"
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{len(self._active_assignments) + 1}"
        
        # Create assignment
        assignment_info = {
            'task_id': task_id,
            'task_description': task_description,
            'assigned_by': 'Project_Owner',
            'status': 'assigned',
            'timestamp': datetime.now()
        }
        
        self._active_assignments[agent_name] = assignment_info
        
        logger.info(f"Task assigned: {agent_name} -> {task_description}")
        return True, f"Task successfully assigned to {agent_name}"

    def is_agent_assigned(self, agent_name: str) -> bool:
        """Check if an agent has an active assignment."""
        return agent_name in self._active_assignments

    def get_agent_assignment(self, agent_name: str) -> Optional[Dict]:
        """Get current assignment for an agent."""
        return self._active_assignments.get(agent_name)

    def complete_agent_assignment(self, agent_name: str) -> bool:
        """Mark an agent's assignment as complete."""
        if agent_name in self._active_assignments:
            self._active_assignments[agent_name]['status'] = 'completed'
            return True
        return False

    def find_agents_for_task(self, task_type: str) -> List[str]:
        """Find agents capable of handling a specific task type."""
        suitable_agents = []
        
        task_type_lower = task_type.lower()
        
        for agent_name, capability in self._agent_capabilities.items():
            # Check if agent can handle this task type
            if ('research' in task_type_lower and capability.can_research) or \
               ('writ' in task_type_lower and capability.can_write) or \
               ('analy' in task_type_lower and capability.can_analyze) or \
               ('cod' in task_type_lower and capability.can_code) or \
               ('review' in task_type_lower and capability.can_review) or \
               ('manag' in task_type_lower and capability.can_manage) or \
               any(task_word in cap.lower() for cap in capability.capabilities for task_word in task_type_lower.split()):
                suitable_agents.append(agent_name)
        
        return suitable_agents

    def get_team_capabilities_summary(self) -> str:
        """Get a formatted summary of all team capabilities."""
        summary = "\n🤖 TEAM CAPABILITIES SUMMARY:\n"
        summary += "=" * 50 + "\n"
        
        for agent_name, capability in self._agent_capabilities.items():
            summary += f"\n🔹 {agent_name}:\n"
            summary += f"   Role: {capability.primary_role}\n"
            summary += f"   Capabilities: {', '.join(capability.capabilities)}\n"
            
            specialties = []
            if capability.can_research: specialties.append("Research")
            if capability.can_write: specialties.append("Writing")
            if capability.can_analyze: specialties.append("Analysis") 
            if capability.can_code: specialties.append("Coding")
            if capability.can_review: specialties.append("Review")
            if capability.can_manage: specialties.append("Management")
            
            if specialties:
                summary += f"   Specialties: {', '.join(specialties)}\n"
        
        return summary

    def validate_agent_assignment(self, agent_name: str) -> bool:
        """Validate if an agent name exists in the team with enhanced feedback."""
        available_names = [getattr(agent, 'name', str(agent)) for agent in self.agents]
        available_names.append(getattr(self.owner, 'name', str(self.owner)))
        if self.report_agent:
            available_names.append(getattr(self.report_agent, 'name', str(self.report_agent)))
        
        is_valid = agent_name in available_names
        
        # Provide immediate feedback for invalid assignments
        if not is_valid and agent_name:
            print(f"\n❌ ASSIGNMENT VALIDATION FAILED!")
            print(f"'{agent_name}' is NOT a valid agent name")
            print(f"🔍 Did you mean one of these?")
            
            # Find close matches
            close_matches = []
            agent_name_lower = agent_name.lower()
            for name in available_names:
                if (agent_name_lower in name.lower() or 
                    name.lower() in agent_name_lower or
                    any(word in name.lower() for word in agent_name_lower.split())):
                    close_matches.append(name)
            
            if close_matches:
                print(f"   Possible matches: {', '.join(close_matches[:3])}")
            else:
                print(f"   First few valid agents: {', '.join(available_names[:5])}")
            
            print(f"📋 Use EXACT names from team composition display!")
            print(f"🚫 Assignment rejected - fix agent name to proceed\n")
        
        return is_valid

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        agent_names = [getattr(agent, 'name', str(agent)) for agent in self.agents]
        return agent_names

    def get_all_participants(self) -> List[str]:
        """Get list of all participant names including owner and report agent."""
        all_names = [getattr(self.owner, 'name', str(self.owner))]
        all_names.extend([getattr(agent, 'name', str(agent)) for agent in self.agents])
        if self.report_agent:
            all_names.append(getattr(self.report_agent, 'name', str(self.report_agent)))
        return all_names

    def get_agent_by_name(self, agent_name: str) -> Optional[Any]:
        """Get agent object by name."""
        # Check owner
        if getattr(self.owner, 'name', str(self.owner)) == agent_name:
            return self.owner
        
        # Check agents
        for agent in self.agents:
            if getattr(agent, 'name', str(agent)) == agent_name:
                return agent
        
        # Check report agent
        if self.report_agent and getattr(self.report_agent, 'name', str(self.report_agent)) == agent_name:
            return self.report_agent
        
        return None

    def validate_task_assignment(self, task_description: str) -> Tuple[bool, str, List[str]]:
        """
        Validate task assignments in a task description.
        Returns: (is_valid, error_message, invalid_agents)
        """
        invalid_agents = []
        available_agents = self.get_all_participants()
        
        # Simple parsing - look for patterns like "AgentName : task"
        lines = task_description.split('\n')
        for line in lines:
            if ':' in line:
                potential_agent = line.split(':')[0].strip()
                # Remove numbering like "1. " if present
                if potential_agent and not potential_agent[0].isdigit():
                    if potential_agent not in available_agents:
                        invalid_agents.append(potential_agent)
        
        if invalid_agents:
            return False, f"Invalid agent assignments: {', '.join(invalid_agents)}", invalid_agents
        
        return True, "All agent assignments are valid", []

    def display_team_composition(self):
        """Display current team composition for debugging."""
        print("\n" + "="*80)
        print("🚨 CRITICAL TEAM AWARENESS - READ CAREFULLY 🚨")
        print("="*80)
        print(f"🔑 PROJECT OWNER: {getattr(self.owner, 'name', str(self.owner))}")
        print(f"   Role: Assigns tasks, coordinates work, provides guidance")
        print()
        print(f"👥 AVAILABLE AGENTS ({len(self.agents)}) - THESE ARE THE ONLY AGENTS THAT EXIST:")
        print("-" * 40)
        
        for i, agent in enumerate(self.agents, 1):
            agent_name = getattr(agent, 'name', str(agent))
            if agent_name in self._agent_capabilities:
                cap = self._agent_capabilities[agent_name]
                print(f"   {i:2d}. {agent_name}")
                print(f"       Role: {cap.primary_role}")
                print(f"       Can do: {', '.join(cap.expertise)}")
                
                # Show specialty flags
                flags = []
                if cap.can_research: flags.append("📊 Research")
                if cap.can_write: flags.append("✍️ Writing") 
                if cap.can_analyze: flags.append("🔍 Analysis")
                if cap.can_code: flags.append("💻 Coding")
                if cap.can_review: flags.append("✅ Review")
                if cap.can_manage: flags.append("📋 Management")
                if flags:
                    print(f"       Specialties: {' | '.join(flags)}")
                print()
            else:
                print(f"   {i:2d}. {agent_name} - (General capabilities)")
                print()
        
        if self.report_agent:
            print(f"📊 REPORT AGENT: {getattr(self.report_agent, 'name', str(self.report_agent))}")
            print()
        
        print("⚠️  CRITICAL RULES:")
        print("   • NEVER assign tasks to fictional agents like 'Research Agent' or 'Writing Agent'")
        print("   • ONLY use exact agent names from the list above")
        print("   • Agents can ONLY participate when explicitly assigned tasks")
        print("   • Use format: <exact_agent_name> : <task_description>")
        print()
        print(f"📋 Total Team Size: {len(self.get_all_participants())} participants")
        print("="*80 + "\n")

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
        """Setup speaker selection method - aggressively override base class."""
        # Override multiple possible base class methods to ensure our logic is used
        override_methods = [
            'speaker_selection_method', 
            'select_speaker', 
            'next_speaker_selector',
            '_select_speaker',
            'get_next_speaker'
        ]
        
        for method_name in override_methods:
            setattr(self, method_name, self._select_next_speaker)
        
        # Also override the private method that might be used internally
        if hasattr(self, '_speaker_selection_method'):
            self._speaker_selection_method = self._select_next_speaker
        
        logger.info("Aggressively overrode base class speaker selection methods")

    def select_speaker(self, *args, **kwargs):
        """Override base class select_speaker method."""
        return self._select_next_speaker(*args, **kwargs)
    
    def _select_speaker(self, *args, **kwargs):
        """Override base class _select_speaker method."""
        return self._select_next_speaker(*args, **kwargs)
    
    def get_next_speaker(self, *args, **kwargs):
        """Override base class get_next_speaker method."""
        return self._select_next_speaker(*args, **kwargs)

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
            
            # Validate assignment
            if not self.validate_agent_assignment(task.assigned_to):
                print(f"   ⚠️  WARNING: Agent '{task.assigned_to}' not found in team!")
                # Suggest alternatives
                suggestions = self.find_agents_for_task(task.description)
                if suggestions:
                    print(f"   💡 Suggested agents: {', '.join(suggestions[:3])}")
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
        """Enhanced speaker selection with assignment validation and task-driven flow."""
        # Track the last speaker
        if last_speaker:
            self.last_speaker = last_speaker
            self._update_conversation_tracking(last_speaker, messages)
        
        # At the start of conversation, provide team awareness manually
        if self.last_speaker is None and len(self.conversation_history) == 0:
            print("\n" + "🤖 TEAM AWARENESS BRIEFING:")
            print(self.get_team_capabilities_summary())
            print("\n" + "📋 ASSIGNMENT RULES:")
            print("- Project Owner must assign tasks before agents can participate")
            print("- Use exact agent names from the list above")
            print("- Agents can only speak when assigned tasks or handed off to")
            print("=" * 60 + "\n")
        
        # Format and display the last message if available
        if messages and len(messages) > 0:
            self._format_agent_output(last_speaker, messages[-1])
        
        # Parse the last message for task assignments and handoffs
        if messages and len(messages) > 0:
            self._parse_message_for_assignments(last_speaker, messages[-1])
        
        # Determine next speaker based on current state
        next_speaker = self._determine_next_speaker()
        
        # Log the selection
        self._log_speaker_selection(next_speaker)
        
        return next_speaker

    def _parse_message_for_assignments(self, speaker, message):
        """Parse message content for task assignments and handoffs with enhanced pattern detection."""
        if not hasattr(message, 'content'):
            return
        
        content = message.content
        speaker_name = getattr(speaker, 'name', str(speaker))
        
        # Check for task assignments (only from Project Owner)
        if speaker == self.owner:
            # Enhanced assignment patterns to catch various formats
            assignment_patterns = [
                r'(\w+)\s*:\s*(.+)',  # "AgentName : task"
                r'\*\*Responsible Agent\*\*:\s*(\w+)',  # "**Responsible Agent**: AgentName"
                r'Assigned to:\s*(\w+)',  # "Assigned to: AgentName"
                r'Agent:\s*(\w+)',  # "Agent: AgentName"
                r'Responsible Agent:\s*(\w+)',  # "Responsible Agent: AgentName"
                r'-\s*\*\*Responsible Agent\*\*:\s*(\w+)',  # "- **Responsible Agent**: AgentName"
            ]
            
            detected_assignments = []
            invalid_assignments = []
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try each pattern
                for pattern in assignment_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            potential_agent = match[0] if len(match) > 0 else ""
                            task_desc = match[1] if len(match) > 1 else "Continue assigned work"
                        else:
                            potential_agent = match
                            task_desc = "Continue assigned work"
                        
                        # Clean up agent name
                        potential_agent = re.sub(r'^\d+\.\s*', '', potential_agent.strip())
                        
                        if potential_agent:
                            # Validate agent exists
                            if self.validate_agent_assignment(potential_agent):
                                success, msg = self.assign_task_to_agent(potential_agent, task_desc)
                                if success:
                                    detected_assignments.append(potential_agent)
                                    print(f"✅ ASSIGNMENT DETECTED: {potential_agent} -> {task_desc[:50]}...")
                            else:
                                invalid_assignments.append(potential_agent)
            
            # Show warnings for invalid assignments
            if invalid_assignments:
                print(f"\n❌ CRITICAL ERROR: Invalid agent assignments detected!")
                print(f"Invalid agents: {', '.join(invalid_assignments)}")
                print(f"✅ Valid agents: {', '.join(self.get_available_agents()[:8])}...")
                print(f"📋 You MUST use exact agent names from the team list!")
                print("🚫 Blocking further progress until valid assignments are made.\n")
                
                # Force owner intervention for invalid assignments
                self._force_owner_next = True
        
        # Check for handoff patterns
        handoff_patterns = [
            r'HANDOFF TO:\s*(\w+)',
            r'handoff to\s*(\w+)',
            r'passing to\s*(\w+)',
            r'transfer to\s*(\w+)'
        ]
        
        for pattern in handoff_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                target_agent_name = match.group(1)
                if self.validate_agent_assignment(target_agent_name):
                    target_agent = self.get_agent_by_name(target_agent_name)
                    if target_agent:
                        self._next_speaker_override = target_agent
                        print(f"🤝 HANDOFF DETECTED: {speaker_name} -> {target_agent_name}")
                        break
                else:
                    print(f"❌ INVALID HANDOFF: {target_agent_name} does not exist!")
                    print(f"✅ Valid agents: {', '.join(self.get_available_agents()[:5])}...")

    def _determine_next_speaker(self):
        """Enhanced logic with aggressive assignment validation and task-driven flow."""
        
        # CRITICAL: Block all unauthorized agent participation
        if (self.last_speaker and self.last_speaker in self.agents):
            speaker_name = getattr(self.last_speaker, 'name', str(self.last_speaker))
            
            # Check if agent has valid assignment
            if not self.is_agent_assigned(speaker_name):
                print(f"\n🚫 BLOCKING UNAUTHORIZED PARTICIPATION!")
                print(f"Agent '{speaker_name}' spoke without assignment")
                print(f"🔄 Forcing owner intervention to assign tasks properly")
                self.owner_intervention_count += 1
                self.turns_since_owner = 0
                self._force_owner_next = True
                return self.owner
        
        # Priority 1: Check for forced owner intervention
        if self._force_owner_next:
            self._force_owner_next = False
            self.turns_since_owner = 0
            print(f"\n👑 OWNER INTERVENTION REQUIRED")
            return self.owner
        
        # Priority 2: Check for explicit handoff override
        if self._next_speaker_override:
            next_speaker = self._next_speaker_override
            self._next_speaker_override = None
            
            # Verify the target agent has assignment or create one
            target_name = getattr(next_speaker, 'name', str(next_speaker))
            if not self.is_agent_assigned(target_name) and next_speaker != self.owner:
                # Auto-assign continuation task
                self.assign_task_to_agent(target_name, "Continue previous agent's work", f"handoff_{len(self._active_assignments)}")
                print(f"🔄 Auto-assigned handoff task to {target_name}")
            
            return next_speaker
        
        # Priority 3: Check handoff status from last turn
        if not self._current_handoff_status and self.last_speaker in self.agents:
            logger.info("Handoff failed - Owner intervention required")
            self.owner_intervention_count += 1
            self.turns_since_owner = 0
            return self.owner
        
        # Case 1: Start of conversation - owner begins
        if self.last_speaker is None:
            self.turns_since_owner = 0
            return self.owner
        
        # Case 2: Owner just spoke - find assigned agent to respond
        if self.last_speaker == self.owner:
            self.turns_since_owner = 0
            
            # Look for recently assigned agents
            for agent_name, assignment in self._active_assignments.items():
                if assignment['status'] == 'assigned':
                    agent = self.get_agent_by_name(agent_name)
                    if agent and agent in self.agents:
                        # Mark as in progress
                        assignment['status'] = 'in_progress'
                        print(f"🎯 Activating assigned agent: {agent_name}")
                        return agent
            
            # CRITICAL: No assigned agents - force owner to assign tasks
            print(f"\n🚫 NO ASSIGNED AGENTS FOUND!")
            print(f"Project Owner must assign tasks before agents can participate")
            print(f"🔄 Staying with owner until proper assignments are made")
            return self.owner
        
        # Case 3: Agent spoke - strict validation
        if self.last_speaker in self.agents:
            speaker_name = getattr(self.last_speaker, 'name', str(self.last_speaker))
            
            # CRITICAL: Check if agent has assignment
            if not self.is_agent_assigned(speaker_name):
                print(f"\n🚫 UNAUTHORIZED AGENT DETECTED!")
                print(f"Agent {speaker_name} participated without assignment")
                print(f"🔄 Forcing immediate owner intervention")
                self.owner_intervention_count += 1
                self.turns_since_owner = 0
                return self.owner
            
            # Force owner intervention if max turns reached
            if self.turns_since_owner >= self.max_agent_turns:
                print(f"\n⏰ MAX AGENT TURNS REACHED ({self.max_agent_turns})")
                print(f"🔄 Owner intervention required")
                self.owner_intervention_count += 1
                self.turns_since_owner = 0
                return self.owner
            
            # Look for other assigned agents to continue
            for agent_name, assignment in self._active_assignments.items():
                if (assignment['status'] in ['assigned', 'in_progress'] and 
                    agent_name != speaker_name):
                    agent = self.get_agent_by_name(agent_name)
                    if agent and agent in self.agents:
                        print(f"🔄 Continuing with next assigned agent: {agent_name}")
                        return agent
            
            # No other assigned agents, return to owner
            print(f"\n📋 No more assigned agents - returning to owner")
            self.turns_since_owner = 0
            return self.owner
        
        # Case 4: Report agent spoke - owner should respond
        if self.report_agent and self.last_speaker == self.report_agent:
            self.turns_since_owner = 0
            return self.owner
        
        # Default fallback - always return to owner if unsure
        print(f"\n🔄 Fallback: Returning to owner for guidance")
        return self.owner

    def _select_agent_for_task(self) -> Any:
        """Select an agent based on available tasks and assignments."""
        # Find agents with active assignments
        for agent_name, assignment in self._active_assignments.items():
            if assignment['status'] == 'assigned':
                agent = self.get_agent_by_name(agent_name)
                if agent and agent in self.agents:
                    return agent
        
        # Find agents with available tasks
        for agent in self.agents:
            available_tasks = self._get_available_tasks_for_agent(agent)
            if available_tasks:
                return agent
        
        # Default to first agent if no specific tasks
        return self.agents[0] if self.agents else self.owner

    def _select_next_agent(self):
        """Select the next agent when allowing agent-to-agent handoff."""
        # Look for assigned agents first
        for agent_name, assignment in self._active_assignments.items():
            if assignment['status'] in ['assigned', 'in_progress']:
                agent = self.get_agent_by_name(agent_name)
                if agent and agent in self.agents and agent != self.last_speaker:
                    return agent
        
        # Fallback to round-robin among agents
        if self.last_speaker in self.agents:
            current_idx = self.agents.index(self.last_speaker)
            next_idx = (current_idx + 1) % len(self.agents)
            return self.agents[next_idx]
        else:
            # Fallback to first agent
            return self.agents[0] if self.agents else self.owner

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
        
        # Formatted output with assignment status
        assignment_status = ""
        if agent_name in self._active_assignments:
            assignment = self._active_assignments[agent_name]
            assignment_status = f" ({assignment['status'].upper()})"
        
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║ 💬 {agent_name}{assignment_status:<{52-len(agent_name)}} ║")
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
        
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
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
        
        # Display active assignments
        print("\n📋 Active Assignments:")
        if self._active_assignments:
            for agent_name, assignment in self._active_assignments.items():
                status_icon = "✅" if assignment['status'] == 'completed' else "🔄" if assignment['status'] == 'in_progress' else "⏳"
                print(f"  {status_icon} {agent_name}: {assignment['task_description'][:50]}... ({assignment['status']})")
        else:
            print("  No active assignments")
        
        print("="*60 + "\n")

    def request_handoff(self, from_agent, to_agent, reason=None) -> Tuple[bool, str]:
        """Enhanced handoff with task awareness and validation."""
        if from_agent not in self.agents or to_agent not in self.agents:
            return False, "Both agents must be in the agent list"
        
        if from_agent == to_agent:
            return False, "Cannot handoff to self"
        
        # Validate that target agent exists
        to_agent_name = getattr(to_agent, 'name', str(to_agent))
        if not self.validate_agent_assignment(to_agent_name):
            return False, f"Target agent '{to_agent_name}' does not exist in team"
        
        # Check if handoff is allowed
        if self.turns_since_owner >= self.max_agent_turns:
            return False, "Owner intervention required - max agent turns reached"
        
        # Check if source agent has an active assignment
        from_agent_name = getattr(from_agent, 'name', str(from_agent))
        if not self.is_agent_assigned(from_agent_name):
            return False, f"{from_agent_name} cannot handoff without an active assignment"
        
        # Create assignment for target agent
        from_assignment = self.get_agent_assignment(from_agent_name)
        handoff_task = f"Continue work from {from_agent_name}: {reason}" if reason else f"Continue work from {from_agent_name}"
        
        success, msg = self.assign_task_to_agent(to_agent_name, handoff_task, f"handoff_{len(self._active_assignments)}")
        if not success:
            return False, f"Failed to assign task to {to_agent_name}: {msg}"
        
        # Mark source assignment as completed
        self.complete_agent_assignment(from_agent_name)
        
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
        
        return True, f"Handoff approved: {from_agent_name} → {to_agent_name}"

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
        speaker_name = getattr(next_speaker, 'name', str(next_speaker))
        status = f"Turn {len(self.conversation_history) + 1}: {speaker_name}"
        status += f" (Turns since owner: {self.turns_since_owner}"
        
        # Add assignment status for agents
        if next_speaker in self.agents:
            if self.is_agent_assigned(speaker_name):
                assignment = self.get_agent_assignment(speaker_name)
                status += f", Status: {assignment['status']}"
            else:
                status += ", Status: UNASSIGNED"
        
        if next_speaker == self.owner and self.turns_since_owner >= self.max_agent_turns:
            status += " - INTERVENTION REQUIRED"
        elif next_speaker == self.owner:
            status += " - Owner guidance"
        else:
            status += " - Agent turn"
        
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

    def set_assignment_required(self, required: bool):
        """Set whether agents require explicit assignment to participate."""
        self._assignment_required = required

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
            'last_speaker': getattr(self.last_speaker, 'name', str(self.last_speaker)) if self.last_speaker else None,
            'active_assignments': len(self._active_assignments),
            'agents_with_capabilities': len(self._agent_capabilities)
        }

    def get_assignment_status(self):
        """Get current assignment status for all agents."""
        status = {}
        for agent_name in self.get_available_agents():
            if self.is_agent_assigned(agent_name):
                assignment = self.get_agent_assignment(agent_name)
                status[agent_name] = {
                    'assigned': True,
                    'task': assignment['task_description'],
                    'status': assignment['status'],
                    'assigned_by': assignment['assigned_by']
                }
            else:
                status[agent_name] = {'assigned': False}
        return status

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
        self._active_assignments = {}

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

    # === AGENT DISCOVERY TOOLS ===
    
    def tool_show_team_capabilities(self) -> str:
        """Tool: Show detailed team capabilities."""
        return self.get_team_capabilities_summary()
    
    def tool_find_agent_for_task(self, task_type: str) -> str:
        """Tool: Find agents suitable for a specific task type."""
        suitable_agents = self.find_agents_for_task(task_type)
        if suitable_agents:
            result = f"Agents suitable for '{task_type}':\n"
            for agent_name in suitable_agents:
                cap = self._agent_capabilities.get(agent_name)
                if cap:
                    result += f"• {agent_name} - {cap.primary_role}\n"
                else:
                    result += f"• {agent_name}\n"
            return result
        else:
            return f"No agents found specifically suitable for '{task_type}'. Consider using general agents or the Project Owner for guidance."
    
    def tool_validate_agent_assignment(self, agent_name: str) -> str:
        """Tool: Validate if an agent exists and get their capabilities."""
        if self.validate_agent_assignment(agent_name):
            cap = self._agent_capabilities.get(agent_name)
            if cap:
                return f"✅ {agent_name} exists. Role: {cap.primary_role}. Capabilities: {', '.join(cap.capabilities)}"
            else:
                return f"✅ {agent_name} exists but capabilities unknown."
        else:
            available = self.get_available_agents()
            return f"❌ {agent_name} does not exist. Available agents: {', '.join(available)}"
    
    def tool_get_assignment_status(self) -> str:
        """Tool: Get current assignment status for all agents."""
        status = self.get_assignment_status()
        result = "📋 Current Assignment Status:\n"
        result += "="*40 + "\n"
        
        for agent_name, info in status.items():
            if info['assigned']:
                result += f"✅ {agent_name}: {info['task'][:50]}... ({info['status']})\n"
            else:
                result += f"⏳ {agent_name}: No active assignment\n"
        
        return result