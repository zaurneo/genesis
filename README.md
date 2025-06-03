# Company Research Multi-Agent System

This is the modular structure.

## Owner Mediated Workflow

The `OwnerMediationGroupChat` ensures every agent interacts only with the
`Project_Owner`. Agents take turns in a round-robin sequence mediated by the
owner. Once all agents finish their tasks, the project owner can use the
`generate_html_report` tool to create `investor_report.html` summarizing the
results.
