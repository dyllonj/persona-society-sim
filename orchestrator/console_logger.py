"""Live console logger for real-time simulation viewing."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Dict, Optional

from schemas.logs import ActionLog, MsgLog


class ConsoleLogger:
    """Real-time console output for simulation events."""

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "red": "\033[91m",
        "gray": "\033[90m",
    }

    ACTION_COLORS = {
        "move": "blue",
        "talk": "green",
        "trade": "yellow",
        "work": "cyan",
    }

    def __init__(self, enabled: bool = True, use_colors: bool = True):
        self.enabled = enabled
        self.use_colors = use_colors and sys.stdout.isatty()
        self.current_tick = -1

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _truncate(self, text: str, max_len: int = 100) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def log_tick_start(self, tick: int, num_events: int) -> None:
        """Log the start of a new tick."""
        if not self.enabled:
            return

        self.current_tick = tick
        separator = "=" * 80
        print(f"\n{self._color(separator, 'cyan')}")
        print(
            f"{self._color('TICK', 'bold')} {self._color(str(tick), 'cyan')} "
            f"| {self._color(f'{num_events} events scheduled', 'dim')}"
        )
        print(f"{self._color(separator, 'cyan')}")

    def log_action(self, action_log: ActionLog) -> None:
        """Log an agent action."""
        if not self.enabled:
            return

        action_color = self.ACTION_COLORS.get(action_log.action_type, "gray")
        agent_id = action_log.agent_id
        action_type = action_log.action_type.upper()
        outcome = action_log.outcome

        # Format outcome
        outcome_symbol = "✓" if outcome == "success" else "✗"
        outcome_color = "green" if outcome == "success" else "red"

        # Build action details
        details = []
        if "destination" in action_log.params:
            details.append(f"→ {action_log.params['destination']}")
        if "item" in action_log.params:
            details.append(f"item: {action_log.params['item']}")
        if "qty" in action_log.params:
            details.append(f"qty: {action_log.params['qty']}")

        details_str = " ".join(details) if details else ""

        print(
            f"  {self._color(outcome_symbol, outcome_color)} "
            f"{self._color(agent_id, 'bold')} "
            f"{self._color(action_type, action_color)} "
            f"{self._color(details_str, 'dim')}"
        )

    def log_message(self, msg_log: MsgLog) -> None:
        """Log an agent message/dialogue."""
        if not self.enabled:
            return

        agent_id = msg_log.from_agent
        room = msg_log.room_id or "unknown"
        content = self._truncate(msg_log.content, max_len=120)

        # Format steering snapshot
        steering = msg_log.steering_snapshot
        if steering:
            traits = ", ".join(
                f"{k}:{v:+.1f}" for k, v in sorted(steering.items()) if abs(v) > 0.1
            )
            steering_str = f"[{traits}]" if traits else ""
        else:
            steering_str = ""

        print(f"    {self._color('💬', 'gray')} {self._color(agent_id, 'bold')} @ {self._color(room, 'magenta')}")

        # Handle multi-line content
        lines = content.split('\n')
        for i, line in enumerate(lines[:3]):  # Show max 3 lines
            if i == 0:
                print(f"       {self._color(line, 'white')}")
            else:
                print(f"       {self._color(line, 'dim')}")

        if len(lines) > 3:
            print(f"       {self._color('...', 'dim')}")

        # Show token usage and steering
        print(
            f"       {self._color(f'[{msg_log.tokens_in}→{msg_log.tokens_out} tokens]', 'gray')} "
            f"{self._color(steering_str, 'gray')}"
        )

    def log_tick_end(self, tick: int, duration_ms: Optional[float] = None) -> None:
        """Log the end of a tick."""
        if not self.enabled:
            return

        if duration_ms:
            duration_str = f"{duration_ms/1000:.2f}s"
            print(f"  {self._color('⏱', 'gray')}  Tick completed in {self._color(duration_str, 'dim')}")

    def log_summary(self, run_id: str, total_ticks: int, total_agents: int, total_time_s: float) -> None:
        """Log simulation summary."""
        if not self.enabled:
            return

        print(f"\n{'=' * 80}")
        print(f"{self._color('SIMULATION COMPLETE', 'bold')}")
        print(f"  Run ID: {self._color(run_id, 'cyan')}")
        print(f"  Ticks: {self._color(str(total_ticks), 'green')}")
        print(f"  Agents: {self._color(str(total_agents), 'green')}")
        print(f"  Total time: {self._color(f'{total_time_s:.2f}s', 'yellow')}")
        print(f"  Avg time/tick: {self._color(f'{total_time_s/total_ticks:.2f}s', 'yellow')}")
        print(f"{'=' * 80}\n")

    def log_error(self, message: str) -> None:
        """Log an error message."""
        if not self.enabled:
            return
        print(f"{self._color('ERROR:', 'red')} {message}", file=sys.stderr)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        if not self.enabled:
            return
        print(f"{self._color('WARNING:', 'yellow')} {message}")

    def log_info(self, message: str) -> None:
        """Log an info message."""
        if not self.enabled:
            return
        print(f"{self._color('INFO:', 'blue')} {message}")
