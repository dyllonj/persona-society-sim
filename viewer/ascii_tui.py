"""ASCII TUI Viewer using rich."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class AsciiViewer:
    def __init__(self) -> None:
        self.console = Console()
        self.layout = Layout()
        self.layout.split(
            Layout(name="map", ratio=2),
            Layout(name="log", ratio=1),
            Layout(name="status", size=3),
        )
        self.current_status = "Initializing..."
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.locations: Dict[str, List[str]] = {}  # location_id -> [agent_ids]
        self.log_messages: deque[str] = deque(maxlen=20)
        self.tick = 0
        self.live: Optional[Live] = None

    def start(self) -> None:
        """Start the live display."""
        self.live = Live(self.layout, refresh_per_second=4, screen=True)
        self.live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()

    def broadcast(self, event: Dict[str, Any]) -> None:
        """Receive an event from the runner and update state."""
        evt_type = event.get("type")
        
        if evt_type == "init":
            self._handle_init(event)
        elif evt_type == "tick":
            self._handle_tick(event)
        elif evt_type == "chat":
            self._handle_chat(event)
        elif evt_type == "action":
            self._handle_action(event)
        elif evt_type == "processing":
            self._handle_processing(event)
            
        if self.live:
            self._update_render()

    def _handle_init(self, event: Dict[str, Any]) -> None:
        agents_list = event.get("agents", [])
        for agent in agents_list:
            aid = agent["agent_id"]
            self.agents[aid] = agent
            loc = agent.get("location_id", "unknown")
            if loc not in self.locations:
                self.locations[loc] = []
            if aid not in self.locations[loc]:
                self.locations[loc].append(aid)

    def _handle_tick(self, event: Dict[str, Any]) -> None:
        self.tick = event.get("tick", 0)
        positions = event.get("positions", {})
        
        # Clear old locations
        self.locations = {}
        
        for aid, loc in positions.items():
            if loc not in self.locations:
                self.locations[loc] = []
            self.locations[loc].append(aid)
            
            if aid not in self.agents:
                self.agents[aid] = {"agent_id": aid, "display_name": aid}

    def _handle_chat(self, event: Dict[str, Any]) -> None:
        speaker = event.get("from_agent", "?")
        content = event.get("content", "")
        room = event.get("room_id", "unknown")
        timestamp = f"[T{self.tick}]"
        
        # Color code speaker based on hash or ID
        color = "cyan"
        msg = f"{timestamp} [bold {color}]{speaker}[/bold {color}] ({room}): {content}"
        self.log_messages.append(msg)

    def _handle_action(self, event: Dict[str, Any]) -> None:
        # Optional: Log significant actions if needed, or just keep chat
        pass

    def _handle_processing(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent_id", "?")
        self.current_status = f"Agent {agent_id} is thinking..."

    def _update_render(self) -> None:
        self.layout["map"].update(self._render_map())
        self.layout["log"].update(self._render_log())
        self.layout["status"].update(self._render_status())

    def _render_map(self) -> Panel:
        # Create a grid of rooms
        grid = Table.grid(expand=True, padding=1)
        
        # Simple 2x2 or 3x3 grid logic depending on room count
        # For now, let's just list them in rows
        
        rooms = sorted(self.locations.keys())
        if not rooms:
            return Panel("Waiting for world state...", title=f"World Map (Tick {self.tick})")

        # Group rooms into rows of 3
        rows = []
        current_row = []
        for room in rooms:
            current_row.append(room)
            if len(current_row) == 3:
                rows.append(current_row)
                current_row = []
        if current_row:
            rows.append(current_row)

        for row_rooms in rows:
            cells = []
            for room_id in row_rooms:
                occupants = self.locations.get(room_id, [])
                
                agent_texts = []
                for aid in occupants:
                    # Shorten ID for display, e.g., "agent-001" -> "001"
                    short_id = aid.split("-")[-1] if "-" in aid else aid[:3]
                    agent_texts.append(f"[bold white on blue]{short_id}[/]")
                
                content = "\n".join(agent_texts) if agent_texts else "[dim]Empty[/dim]"
                
                room_panel = Panel(
                    content,
                    title=f"[bold green]{room_id.replace('_', ' ').title()}[/]",
                    border_style="green",
                    height=8  # Fixed height for uniformity
                )
                cells.append(room_panel)
            
            # Pad row if needed
            while len(cells) < 3:
                cells.append(Text(""))
                
            grid.add_row(*cells)

        return Panel(grid, title=f"World Map (Tick {self.tick})", border_style="blue")

    def _render_log(self) -> Panel:
        text = "\n".join(self.log_messages)
        return Panel(text, title="Dialogue Log", border_style="yellow")

    def _render_status(self) -> Panel:
        return Panel(self.current_status, title="System Status", border_style="white")
