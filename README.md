# Company Research Multi-Agent System

This is the modular structure.

## Owner Mediated Workflow

The `OwnerMediationGroupChat` ensures every agent interacts only with the
`Project_Owner`. Agents take turns in a round-robin sequence mediated by the
owner. Once all analytical tasks are complete the owner switches the chat to a
report phase. During this phase only the `Project_Owner` and the new
`Report_Insight_Generator` agent can speak. Attempting to start the report phase
before all tasks are marked completed will result in an error. The report agent
will refuse to generate the HTML summary unless the report phase has been
activated. Once active, the agent compiles `investor_report.html` using all
generated data and evaluations so investors receive a clear, structured summary.

## Agent Roles

- `InnovativeThinkerAgent` merges creative solution generation with out-of-the-box
  perspectives to inspire unique approaches.
- `ProjectStrategyManagerAgent` combines classic project management with long-term
  strategic planning to keep tasks on schedule and aligned with overall goals.
