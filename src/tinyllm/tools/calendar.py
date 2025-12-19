"""Calendar tools for TinyLLM.

This module provides tools for managing calendar events,
with support for iCal format and time zones.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class EventStatus(str, Enum):
    """Event status values."""

    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class RecurrenceFrequency(str, Enum):
    """Recurrence frequency values."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class Weekday(str, Enum):
    """Days of the week."""

    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


@dataclass
class RecurrenceRule:
    """Rule for recurring events."""

    frequency: RecurrenceFrequency
    interval: int = 1
    count: Optional[int] = None
    until: Optional[datetime] = None
    by_day: List[Weekday] = field(default_factory=list)
    by_month_day: List[int] = field(default_factory=list)
    by_month: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frequency": self.frequency.value,
            "interval": self.interval,
            "count": self.count,
            "until": self.until.isoformat() if self.until else None,
            "by_day": [d.value for d in self.by_day],
            "by_month_day": self.by_month_day,
            "by_month": self.by_month,
        }


@dataclass
class Attendee:
    """Event attendee."""

    email: str
    name: Optional[str] = None
    required: bool = True
    rsvp: Optional[str] = None  # accepted, declined, tentative, needs-action

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email": self.email,
            "name": self.name,
            "required": self.required,
            "rsvp": self.rsvp,
        }


@dataclass
class Reminder:
    """Event reminder."""

    minutes_before: int
    method: str = "popup"  # popup, email, sms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "minutes_before": self.minutes_before,
            "method": self.method,
        }


@dataclass
class CalendarEvent:
    """A calendar event."""

    id: str
    title: str
    start: datetime
    end: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    status: EventStatus = EventStatus.CONFIRMED
    all_day: bool = False
    recurrence: Optional[RecurrenceRule] = None
    attendees: List[Attendee] = field(default_factory=list)
    reminders: List[Reminder] = field(default_factory=list)
    calendar_id: str = "default"
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """Get event duration."""
        return self.end - self.start

    @property
    def is_recurring(self) -> bool:
        """Check if event is recurring."""
        return self.recurrence is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "description": self.description,
            "location": self.location,
            "status": self.status.value,
            "all_day": self.all_day,
            "recurrence": self.recurrence.to_dict() if self.recurrence else None,
            "attendees": [a.to_dict() for a in self.attendees],
            "reminders": [r.to_dict() for r in self.reminders],
            "calendar_id": self.calendar_id,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "duration_minutes": int(self.duration.total_seconds() / 60),
            "is_recurring": self.is_recurring,
        }


@dataclass
class Calendar:
    """A calendar containing events."""

    id: str
    name: str
    description: Optional[str] = None
    timezone: str = "UTC"
    color: Optional[str] = None
    readonly: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "timezone": self.timezone,
            "color": self.color,
            "readonly": self.readonly,
        }


@dataclass
class TimeSlot:
    """A time slot for availability."""

    start: datetime
    end: datetime
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "available": self.available,
            "duration_minutes": int((self.end - self.start).total_seconds() / 60),
        }


class CalendarStore:
    """In-memory calendar storage."""

    def __init__(self):
        """Initialize store."""
        self._calendars: Dict[str, Calendar] = {}
        self._events: Dict[str, CalendarEvent] = {}

        # Create default calendar
        self._calendars["default"] = Calendar(
            id="default",
            name="Default Calendar",
        )

    def add_calendar(self, calendar: Calendar) -> None:
        """Add a calendar.

        Args:
            calendar: Calendar to add.
        """
        self._calendars[calendar.id] = calendar

    def get_calendar(self, calendar_id: str) -> Optional[Calendar]:
        """Get a calendar.

        Args:
            calendar_id: Calendar ID.

        Returns:
            Calendar or None.
        """
        return self._calendars.get(calendar_id)

    def list_calendars(self) -> List[Calendar]:
        """List all calendars."""
        return list(self._calendars.values())

    def delete_calendar(self, calendar_id: str) -> bool:
        """Delete a calendar and its events.

        Args:
            calendar_id: Calendar ID.

        Returns:
            True if deleted.
        """
        if calendar_id in self._calendars:
            # Delete associated events
            event_ids = [
                eid for eid, e in self._events.items()
                if e.calendar_id == calendar_id
            ]
            for eid in event_ids:
                del self._events[eid]

            del self._calendars[calendar_id]
            return True
        return False

    def add_event(self, event: CalendarEvent) -> None:
        """Add an event.

        Args:
            event: Event to add.
        """
        self._events[event.id] = event

    def get_event(self, event_id: str) -> Optional[CalendarEvent]:
        """Get an event.

        Args:
            event_id: Event ID.

        Returns:
            Event or None.
        """
        return self._events.get(event_id)

    def update_event(self, event: CalendarEvent) -> bool:
        """Update an event.

        Args:
            event: Event to update.

        Returns:
            True if updated.
        """
        if event.id in self._events:
            event.updated = datetime.now()
            self._events[event.id] = event
            return True
        return False

    def delete_event(self, event_id: str) -> bool:
        """Delete an event.

        Args:
            event_id: Event ID.

        Returns:
            True if deleted.
        """
        if event_id in self._events:
            del self._events[event_id]
            return True
        return False

    def list_events(
        self,
        calendar_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[CalendarEvent]:
        """List events with optional filtering.

        Args:
            calendar_id: Filter by calendar ID.
            start: Filter by start date.
            end: Filter by end date.

        Returns:
            List of events.
        """
        events = list(self._events.values())

        if calendar_id:
            events = [e for e in events if e.calendar_id == calendar_id]

        if start:
            events = [e for e in events if e.end >= start]

        if end:
            events = [e for e in events if e.start <= end]

        return sorted(events, key=lambda e: e.start)

    def find_free_slots(
        self,
        start: datetime,
        end: datetime,
        duration_minutes: int,
        calendar_ids: Optional[List[str]] = None,
    ) -> List[TimeSlot]:
        """Find available time slots.

        Args:
            start: Start of search range.
            end: End of search range.
            duration_minutes: Required slot duration.
            calendar_ids: Calendars to check (None = all).

        Returns:
            List of available time slots.
        """
        # Get all events in range
        events = self.list_events(start=start, end=end)

        if calendar_ids:
            events = [e for e in events if e.calendar_id in calendar_ids]

        # Sort events by start time
        events = sorted(events, key=lambda e: e.start)

        # Find gaps
        slots = []
        current = start
        duration = timedelta(minutes=duration_minutes)

        for event in events:
            if event.start > current:
                # Gap before this event
                gap_end = event.start
                while current + duration <= gap_end:
                    slot_end = current + duration
                    slots.append(TimeSlot(
                        start=current,
                        end=slot_end,
                        available=True,
                    ))
                    current = slot_end

            # Move past this event
            if event.end > current:
                current = event.end

        # Check for gap after last event
        while current + duration <= end:
            slot_end = current + duration
            slots.append(TimeSlot(
                start=current,
                end=slot_end,
                available=True,
            ))
            current = slot_end

        return slots


# Pydantic models for tool inputs/outputs


class CreateEventInput(BaseModel):
    """Input for creating an event."""

    title: str = Field(..., description="Event title")
    start: str = Field(..., description="Start time (ISO format)")
    end: str = Field(..., description="End time (ISO format)")
    description: Optional[str] = Field(default=None, description="Event description")
    location: Optional[str] = Field(default=None, description="Event location")
    calendar_id: str = Field(default="default", description="Calendar ID")
    all_day: bool = Field(default=False, description="Whether it's an all-day event")
    attendees: Optional[List[str]] = Field(
        default=None,
        description="List of attendee emails",
    )


class CreateEventOutput(BaseModel):
    """Output from creating an event."""

    success: bool = Field(description="Whether event was created")
    event_id: Optional[str] = Field(default=None, description="Created event ID")
    event: Optional[Dict[str, Any]] = Field(default=None, description="Event details")
    error: Optional[str] = Field(default=None, description="Error message")


class GetEventInput(BaseModel):
    """Input for getting an event."""

    event_id: str = Field(..., description="Event ID")


class GetEventOutput(BaseModel):
    """Output from getting an event."""

    success: bool = Field(description="Whether event was found")
    event: Optional[Dict[str, Any]] = Field(default=None, description="Event details")
    error: Optional[str] = Field(default=None, description="Error message")


class ListEventsInput(BaseModel):
    """Input for listing events."""

    calendar_id: Optional[str] = Field(default=None, description="Filter by calendar")
    start: Optional[str] = Field(default=None, description="Start of range (ISO format)")
    end: Optional[str] = Field(default=None, description="End of range (ISO format)")


class ListEventsOutput(BaseModel):
    """Output from listing events."""

    success: bool = Field(description="Whether operation succeeded")
    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of events",
    )
    count: int = Field(default=0, description="Number of events")
    error: Optional[str] = Field(default=None, description="Error message")


class UpdateEventInput(BaseModel):
    """Input for updating an event."""

    event_id: str = Field(..., description="Event ID")
    title: Optional[str] = Field(default=None, description="New title")
    start: Optional[str] = Field(default=None, description="New start time")
    end: Optional[str] = Field(default=None, description="New end time")
    description: Optional[str] = Field(default=None, description="New description")
    location: Optional[str] = Field(default=None, description="New location")
    status: Optional[str] = Field(default=None, description="New status")


class UpdateEventOutput(BaseModel):
    """Output from updating an event."""

    success: bool = Field(description="Whether event was updated")
    event: Optional[Dict[str, Any]] = Field(default=None, description="Updated event")
    error: Optional[str] = Field(default=None, description="Error message")


class DeleteEventInput(BaseModel):
    """Input for deleting an event."""

    event_id: str = Field(..., description="Event ID")


class DeleteEventOutput(BaseModel):
    """Output from deleting an event."""

    success: bool = Field(description="Whether event was deleted")
    error: Optional[str] = Field(default=None, description="Error message")


class FindFreeSlotsInput(BaseModel):
    """Input for finding free time slots."""

    start: str = Field(..., description="Start of search range (ISO format)")
    end: str = Field(..., description="End of search range (ISO format)")
    duration_minutes: int = Field(default=60, description="Required slot duration")
    calendar_ids: Optional[List[str]] = Field(
        default=None,
        description="Calendars to check",
    )


class FindFreeSlotsOutput(BaseModel):
    """Output from finding free slots."""

    success: bool = Field(description="Whether search succeeded")
    slots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Available time slots",
    )
    count: int = Field(default=0, description="Number of slots")
    error: Optional[str] = Field(default=None, description="Error message")


# Tool implementations


class CreateEventTool(BaseTool[CreateEventInput, CreateEventOutput]):
    """Tool for creating calendar events."""

    metadata = ToolMetadata(
        id="create_calendar_event",
        name="Create Calendar Event",
        description="Create a new calendar event",
        category="utility",
    )
    input_type = CreateEventInput
    output_type = CreateEventOutput

    def __init__(self, store: CalendarStore):
        """Initialize tool."""
        self.store = store

    async def execute(self, input: CreateEventInput) -> CreateEventOutput:
        """Create an event."""
        try:
            start = datetime.fromisoformat(input.start)
            end = datetime.fromisoformat(input.end)
        except ValueError as e:
            return CreateEventOutput(
                success=False,
                error=f"Invalid date format: {e}",
            )

        if end <= start:
            return CreateEventOutput(
                success=False,
                error="End time must be after start time",
            )

        calendar = self.store.get_calendar(input.calendar_id)
        if not calendar:
            return CreateEventOutput(
                success=False,
                error=f"Calendar '{input.calendar_id}' not found",
            )

        attendees = []
        if input.attendees:
            attendees = [Attendee(email=email) for email in input.attendees]

        event = CalendarEvent(
            id=str(uuid.uuid4()),
            title=input.title,
            start=start,
            end=end,
            description=input.description,
            location=input.location,
            calendar_id=input.calendar_id,
            all_day=input.all_day,
            attendees=attendees,
        )

        self.store.add_event(event)

        return CreateEventOutput(
            success=True,
            event_id=event.id,
            event=event.to_dict(),
        )


class GetEventTool(BaseTool[GetEventInput, GetEventOutput]):
    """Tool for getting a calendar event."""

    metadata = ToolMetadata(
        id="get_calendar_event",
        name="Get Calendar Event",
        description="Get a calendar event by ID",
        category="utility",
    )
    input_type = GetEventInput
    output_type = GetEventOutput

    def __init__(self, store: CalendarStore):
        """Initialize tool."""
        self.store = store

    async def execute(self, input: GetEventInput) -> GetEventOutput:
        """Get an event."""
        event = self.store.get_event(input.event_id)

        if not event:
            return GetEventOutput(
                success=False,
                error=f"Event '{input.event_id}' not found",
            )

        return GetEventOutput(
            success=True,
            event=event.to_dict(),
        )


class ListEventsTool(BaseTool[ListEventsInput, ListEventsOutput]):
    """Tool for listing calendar events."""

    metadata = ToolMetadata(
        id="list_calendar_events",
        name="List Calendar Events",
        description="List calendar events with optional filtering",
        category="utility",
    )
    input_type = ListEventsInput
    output_type = ListEventsOutput

    def __init__(self, store: CalendarStore):
        """Initialize tool."""
        self.store = store

    async def execute(self, input: ListEventsInput) -> ListEventsOutput:
        """List events."""
        try:
            start = datetime.fromisoformat(input.start) if input.start else None
            end = datetime.fromisoformat(input.end) if input.end else None
        except ValueError as e:
            return ListEventsOutput(
                success=False,
                error=f"Invalid date format: {e}",
            )

        events = self.store.list_events(
            calendar_id=input.calendar_id,
            start=start,
            end=end,
        )

        return ListEventsOutput(
            success=True,
            events=[e.to_dict() for e in events],
            count=len(events),
        )


class UpdateEventTool(BaseTool[UpdateEventInput, UpdateEventOutput]):
    """Tool for updating a calendar event."""

    metadata = ToolMetadata(
        id="update_calendar_event",
        name="Update Calendar Event",
        description="Update an existing calendar event",
        category="utility",
    )
    input_type = UpdateEventInput
    output_type = UpdateEventOutput

    def __init__(self, store: CalendarStore):
        """Initialize tool."""
        self.store = store

    async def execute(self, input: UpdateEventInput) -> UpdateEventOutput:
        """Update an event."""
        event = self.store.get_event(input.event_id)

        if not event:
            return UpdateEventOutput(
                success=False,
                error=f"Event '{input.event_id}' not found",
            )

        try:
            if input.title is not None:
                event.title = input.title

            if input.start is not None:
                event.start = datetime.fromisoformat(input.start)

            if input.end is not None:
                event.end = datetime.fromisoformat(input.end)

            if input.description is not None:
                event.description = input.description

            if input.location is not None:
                event.location = input.location

            if input.status is not None:
                event.status = EventStatus(input.status.lower())

        except ValueError as e:
            return UpdateEventOutput(
                success=False,
                error=f"Invalid input: {e}",
            )

        if event.end <= event.start:
            return UpdateEventOutput(
                success=False,
                error="End time must be after start time",
            )

        self.store.update_event(event)

        return UpdateEventOutput(
            success=True,
            event=event.to_dict(),
        )


class DeleteEventTool(BaseTool[DeleteEventInput, DeleteEventOutput]):
    """Tool for deleting a calendar event."""

    metadata = ToolMetadata(
        id="delete_calendar_event",
        name="Delete Calendar Event",
        description="Delete a calendar event",
        category="utility",
    )
    input_type = DeleteEventInput
    output_type = DeleteEventOutput

    def __init__(self, store: CalendarStore):
        """Initialize tool."""
        self.store = store

    async def execute(self, input: DeleteEventInput) -> DeleteEventOutput:
        """Delete an event."""
        deleted = self.store.delete_event(input.event_id)

        if not deleted:
            return DeleteEventOutput(
                success=False,
                error=f"Event '{input.event_id}' not found",
            )

        return DeleteEventOutput(success=True)


class FindFreeSlotsTool(BaseTool[FindFreeSlotsInput, FindFreeSlotsOutput]):
    """Tool for finding free time slots."""

    metadata = ToolMetadata(
        id="find_free_slots",
        name="Find Free Slots",
        description="Find available time slots in calendars",
        category="utility",
    )
    input_type = FindFreeSlotsInput
    output_type = FindFreeSlotsOutput

    def __init__(self, store: CalendarStore):
        """Initialize tool."""
        self.store = store

    async def execute(self, input: FindFreeSlotsInput) -> FindFreeSlotsOutput:
        """Find free slots."""
        try:
            start = datetime.fromisoformat(input.start)
            end = datetime.fromisoformat(input.end)
        except ValueError as e:
            return FindFreeSlotsOutput(
                success=False,
                error=f"Invalid date format: {e}",
            )

        if end <= start:
            return FindFreeSlotsOutput(
                success=False,
                error="End time must be after start time",
            )

        slots = self.store.find_free_slots(
            start=start,
            end=end,
            duration_minutes=input.duration_minutes,
            calendar_ids=input.calendar_ids,
        )

        return FindFreeSlotsOutput(
            success=True,
            slots=[s.to_dict() for s in slots],
            count=len(slots),
        )


# Convenience functions


def create_calendar_store() -> CalendarStore:
    """Create a calendar store.

    Returns:
        Calendar store.
    """
    return CalendarStore()


def create_calendar_tools(store: CalendarStore) -> Dict[str, BaseTool]:
    """Create all calendar tools.

    Args:
        store: Calendar store.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "create_calendar_event": CreateEventTool(store),
        "get_calendar_event": GetEventTool(store),
        "list_calendar_events": ListEventsTool(store),
        "update_calendar_event": UpdateEventTool(store),
        "delete_calendar_event": DeleteEventTool(store),
        "find_free_slots": FindFreeSlotsTool(store),
    }
