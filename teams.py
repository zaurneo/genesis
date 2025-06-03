from autogen_agentchat.teams import RoundRobinGroupChat

class OwnerMediationGroupChat(RoundRobinGroupChat):
    """Round-robin chat where the project owner mediates every exchange."""

    def __init__(self, owner, agents, **kwargs):
        self.owner = owner
        order = []
        for agent in agents:
            order.extend([owner, agent])
        order.append(owner)
        super().__init__(order, **kwargs)
