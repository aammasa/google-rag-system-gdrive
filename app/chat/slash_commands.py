"""
Google Chat slash command definitions and dispatcher.

Register these commands in GCP Console → Google Chat API → Configuration → Slash commands.
The commandId here must match what you register in the console.
"""

from dataclasses import dataclass
from enum import IntEnum


class CommandId(IntEnum):
    HELP = 1
    CLEAR = 2
    SOURCES = 3
    INGEST = 4


@dataclass
class SlashCommand:
    id: CommandId
    name: str
    description: str


# Registry — mirrors what you configure in GCP Console
COMMANDS: dict[int, SlashCommand] = {
    CommandId.HELP: SlashCommand(
        id=CommandId.HELP,
        name="/help",
        description="Show available commands",
    ),
    CommandId.CLEAR: SlashCommand(
        id=CommandId.CLEAR,
        name="/clear",
        description="Clear your conversation history in this space",
    ),
    CommandId.SOURCES: SlashCommand(
        id=CommandId.SOURCES,
        name="/sources",
        description="Show sources from the last answer",
    ),
    CommandId.INGEST: SlashCommand(
        id=CommandId.INGEST,
        name="/ingest",
        description="Trigger re-ingestion of Google Drive documents",
    ),
}


def parse_slash_command(event: dict) -> int | None:
    """
    Extract the slash command ID from a Chat event.
    Returns None if the message is not a slash command.
    """
    slash = event.get("message", {}).get("slashCommand")
    if slash:
        return int(slash.get("commandId", -1))
    return None
