"""
title: Time Tools
author: OpenWebUI
description: Tools for getting current date and time information
required_open_webui_version: 0.4.0
version: 0.1.0
license: MIT
"""

from datetime import datetime
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        time_format: str = Field(
            default="%H:%M:%S",
            description="Time format string (e.g. %H:%M:%S for 24-hour, %I:%M:%S %p for 12-hour)"
        )

    class UserValves(BaseModel):
        use_24h: bool = Field(
            default=True,
            description="Use 24-hour time format instead of 12-hour"
        )

    def __init__(self):
        """Initialize the Tool."""
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    async def get_current_date(self, __event_emitter__=None) -> str:
        """
        Get the current date.
        :return: The current date as a string.
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Fetching current date...", "done": False}
                    }
                )

            current_date = datetime.now().strftime("%A, %B %d, %Y")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Date retrieved", "done": True}
                    }
                )

            return f"Today's date is {current_date}"
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True}
                    }
                )
            return f"Error getting date: {str(e)}"

    async def get_current_time(self, __event_emitter__=None) -> str:
        """
        Get the current time.
        :return: The current time as a string.
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Fetching current time...", "done": False}
                    }
                )

            time_format = self.valves.time_format
            if not self.user_valves.use_24h:
                time_format = time_format.replace("%H", "%I") + " %p"

            current_time = datetime.now().strftime(time_format)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Time retrieved", "done": True}
                    }
                )

            return f"Current Time: {current_time}"
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True}
                    }
                )
            return f"Error getting time: {str(e)}"
