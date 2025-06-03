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

        # Initialize parent with unique participants
        participants = [owner] + self.agents
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

    def _select_next(self, last_speaker=None):
        speaker = self._order[self._idx]
        self._idx = (self._idx + 1) % len(self._order)
        return speaker
