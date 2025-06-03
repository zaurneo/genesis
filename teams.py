from autogen_agentchat.teams import RoundRobinGroupChat


class OwnerMediationGroupChat(RoundRobinGroupChat):
    """Round-robin chat where the project owner mediates every exchange."""

    def __init__(self, owner, agents, report_agent=None, **kwargs):
        self.owner = owner
        self.agents = list(agents)
        self.report_agent = report_agent
        self.report_phase = False

        # Build the mediation order: owner -> agent -> owner -> next agent -> ...
        order = []
        for agent in self.agents:
            order.extend([owner, agent])
        order.append(owner)

        # Save order and index for the custom speaker selector
        self._order = order
        self._idx = 0

        # Initialize parent with unique participants
        participants = [owner] + self.agents
        if self.report_agent:
            participants.append(self.report_agent)
        # Older versions of ``autogen`` do not accept ``speaker_selection_method``
        # in ``RoundRobinGroupChat.__init__``. We therefore call the parent
        # initializer without it and set the speaker selection method
        # afterwards for compatibility.
        super().__init__(participants, **kwargs)

        # Set the speaker selection hook if supported
        if hasattr(self, "speaker_selection_method"):
            self.speaker_selection_method = self._select_next
        else:
            # Fallback for APIs that use a different attribute name
            setattr(self, "select_speaker", self._select_next)

    def start_report_phase(self):
        """Restrict conversation to the owner and report agent."""
        if self.report_agent:
            self.report_phase = True
            self._order = [self.owner, self.report_agent, self.owner]
            self._idx = 0

    def _select_next(self, last_speaker=None):
        speaker = self._order[self._idx]
        self._idx = (self._idx + 1) % len(self._order)
        return speaker
