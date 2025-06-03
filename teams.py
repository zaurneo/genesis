from autogen_agentchat.teams import RoundRobinGroupChat


class OwnerMediationGroupChat(RoundRobinGroupChat):
    """Round-robin chat where the project owner mediates every exchange."""

    def __init__(self, owner, agents, **kwargs):
        self.owner = owner
        self.agents = list(agents)

        # Build the mediation order: owner -> agent -> owner -> next agent -> ...
        order = []
        for agent in self.agents:
            order.extend([owner, agent])
        order.append(owner)

        # Save order and index for the custom speaker selector
        self._order = order
        self._idx = 0

        # Initialize parent with unique participants and custom speaker selector
        participants = [owner] + self.agents
        super().__init__(participants, speaker_selection_method=self._select_next, **kwargs)

    def _select_next(self, last_speaker=None):
        speaker = self._order[self._idx]
        self._idx = (self._idx + 1) % len(self._order)
        return speaker
