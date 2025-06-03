# Company Research Multi-Agent System

This is the modular structure.

## Owner Mediated Workflow

The `OwnerMediationGroupChat` ensures every agent interacts only with the
`Project_Owner`. Agents take turns in a round-robin sequence mediated by the
owner. Once all analytical tasks are complete the owner switches the chat to a
report phase. During this phase only the `Project_Owner` and the new
`Report_Insight_Generator` agent can speak. The report agent compiles
`investor_report.html` using all generated data and evaluations so investors
receive a clear, structured summary.
