"""
Demo 1: Production Support Ticket Classifier
Building Production AI Agents with Microsoft Foundry

This agent classifies enterprise support tickets using GPT-4o,
demonstrating identity-based auth, streaming output, and
automatic OpenTelemetry tracing through Foundry.
"""
import asyncio
import pandas as pd
from agent_framework.azure import AzureAIAgentClient
from azure.identity import DefaultAzureCredential


async def main():
    # ---------------------------------------------------------
    # Production pattern: identity-based auth, no API keys
    # DefaultAzureCredential tries managed identity first,
    # then falls back to Azure CLI, VS Code, etc.
    # ---------------------------------------------------------
    client = AzureAIAgentClient(
        credential=DefaultAzureCredential(),
        endpoint="https://your-hub.services.ai.azure.com"
    )

    # ---------------------------------------------------------
    # Create agent with clear, constrained instructions
    # In production, these instructions would come from a
    # configuration store, not hardcoded -- but this is demo-clear.
    # ---------------------------------------------------------
    agent = client.as_agent(
        name="Support Ticket Classifier",
        instructions="""You are an enterprise support ticket classifier.

For each ticket, analyze the subject and description, then return:
- Category: one of [Infrastructure, Security, Access Management,
  Application, Data, Compliance]
- Priority: P1 (critical/outage), P2 (degraded service),
  P3 (important but not urgent), P4 (request/informational)
- Routing Team: the team best suited to handle this
- Summary: one-line summary of the core issue

Format your response as a structured table.""",
        model="gpt-4o"
    )

    # Load support tickets from CSV
    tickets = pd.read_csv("tickets.csv")

    # Build prompt with all tickets
    ticket_text = tickets.to_string(index=False)
    prompt = f"""Classify these support tickets:\n\n{ticket_text}"""

    # ---------------------------------------------------------
    # Stream the response -- results appear in real time
    # This is more engaging for demos and matches production
    # patterns where you want progressive output.
    # ---------------------------------------------------------
    print("Classifying support tickets...\n")
    async for event in agent.run_stream(prompt):
        if event.type == "output":
            print(event.data, end="", flush=True)

    print("\n\nClassification complete. Check Foundry Studio for traces.")


if __name__ == "__main__":
    asyncio.run(main())