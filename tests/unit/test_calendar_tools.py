"""Tests for calendar tools."""

import pytest
from datetime import datetime, timedelta

from tinyllm.tools.calendar import (
    Attendee,
    Calendar,
    CalendarEvent,
    CalendarStore,
    CreateEventInput,
    CreateEventOutput,
    CreateEventTool,
    DeleteEventInput,
    DeleteEventOutput,
    DeleteEventTool,
    EventStatus,
    FindFreeSlotsInput,
    FindFreeSlotsOutput,
    FindFreeSlotsTool,
    GetEventInput,
    GetEventOutput,
    GetEventTool,
    ListEventsInput,
    ListEventsOutput,
    ListEventsTool,
    RecurrenceFrequency,
    RecurrenceRule,
    Reminder,
    TimeSlot,
    UpdateEventInput,
    UpdateEventOutput,
    UpdateEventTool,
    Weekday,
    create_calendar_store,
    create_calendar_tools,
)


class TestRecurrenceRule:
    """Tests for RecurrenceRule."""

    def test_creation(self):
        """Test rule creation."""
        rule = RecurrenceRule(
            frequency=RecurrenceFrequency.WEEKLY,
            interval=2,
            by_day=[Weekday.MONDAY, Weekday.WEDNESDAY],
        )

        assert rule.frequency == RecurrenceFrequency.WEEKLY
        assert rule.interval == 2
        assert len(rule.by_day) == 2

    def test_to_dict(self):
        """Test converting to dictionary."""
        rule = RecurrenceRule(
            frequency=RecurrenceFrequency.MONTHLY,
            by_month_day=[1, 15],
        )

        d = rule.to_dict()

        assert d["frequency"] == "monthly"
        assert d["by_month_day"] == [1, 15]


class TestAttendee:
    """Tests for Attendee."""

    def test_creation(self):
        """Test attendee creation."""
        attendee = Attendee(
            email="user@example.com",
            name="Test User",
            required=True,
        )

        assert attendee.email == "user@example.com"
        assert attendee.name == "Test User"

    def test_to_dict(self):
        """Test converting to dictionary."""
        attendee = Attendee(
            email="user@example.com",
            rsvp="accepted",
        )

        d = attendee.to_dict()

        assert d["email"] == "user@example.com"
        assert d["rsvp"] == "accepted"


class TestReminder:
    """Tests for Reminder."""

    def test_creation(self):
        """Test reminder creation."""
        reminder = Reminder(minutes_before=30, method="popup")

        assert reminder.minutes_before == 30
        assert reminder.method == "popup"

    def test_to_dict(self):
        """Test converting to dictionary."""
        reminder = Reminder(minutes_before=15, method="email")

        d = reminder.to_dict()

        assert d["minutes_before"] == 15
        assert d["method"] == "email"


class TestCalendarEvent:
    """Tests for CalendarEvent."""

    def test_creation(self):
        """Test event creation."""
        now = datetime.now()
        event = CalendarEvent(
            id="test-123",
            title="Test Event",
            start=now,
            end=now + timedelta(hours=1),
        )

        assert event.id == "test-123"
        assert event.title == "Test Event"
        assert event.status == EventStatus.CONFIRMED

    def test_duration(self):
        """Test duration calculation."""
        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Test",
            start=now,
            end=now + timedelta(hours=2),
        )

        assert event.duration == timedelta(hours=2)

    def test_is_recurring(self):
        """Test recurring check."""
        now = datetime.now()
        event1 = CalendarEvent(
            id="test1",
            title="Single",
            start=now,
            end=now + timedelta(hours=1),
        )

        event2 = CalendarEvent(
            id="test2",
            title="Recurring",
            start=now,
            end=now + timedelta(hours=1),
            recurrence=RecurrenceRule(frequency=RecurrenceFrequency.DAILY),
        )

        assert event1.is_recurring is False
        assert event2.is_recurring is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Test Event",
            start=now,
            end=now + timedelta(minutes=30),
            location="Room 101",
        )

        d = event.to_dict()

        assert d["id"] == "test"
        assert d["title"] == "Test Event"
        assert d["location"] == "Room 101"
        assert d["duration_minutes"] == 30


class TestCalendar:
    """Tests for Calendar."""

    def test_creation(self):
        """Test calendar creation."""
        calendar = Calendar(
            id="work",
            name="Work Calendar",
            timezone="America/New_York",
        )

        assert calendar.id == "work"
        assert calendar.name == "Work Calendar"
        assert calendar.timezone == "America/New_York"

    def test_to_dict(self):
        """Test converting to dictionary."""
        calendar = Calendar(
            id="personal",
            name="Personal",
            color="#FF0000",
        )

        d = calendar.to_dict()

        assert d["id"] == "personal"
        assert d["color"] == "#FF0000"


class TestTimeSlot:
    """Tests for TimeSlot."""

    def test_creation(self):
        """Test slot creation."""
        now = datetime.now()
        slot = TimeSlot(
            start=now,
            end=now + timedelta(hours=1),
            available=True,
        )

        assert slot.available is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        slot = TimeSlot(
            start=now,
            end=now + timedelta(minutes=30),
        )

        d = slot.to_dict()

        assert d["available"] is True
        assert d["duration_minutes"] == 30


class TestCalendarStore:
    """Tests for CalendarStore."""

    def test_default_calendar(self):
        """Test default calendar exists."""
        store = CalendarStore()

        calendar = store.get_calendar("default")

        assert calendar is not None
        assert calendar.name == "Default Calendar"

    def test_add_calendar(self):
        """Test adding a calendar."""
        store = CalendarStore()
        calendar = Calendar(id="work", name="Work")

        store.add_calendar(calendar)

        assert store.get_calendar("work") == calendar

    def test_list_calendars(self):
        """Test listing calendars."""
        store = CalendarStore()
        store.add_calendar(Calendar(id="work", name="Work"))

        calendars = store.list_calendars()

        assert len(calendars) >= 2  # default + work

    def test_delete_calendar(self):
        """Test deleting a calendar."""
        store = CalendarStore()
        store.add_calendar(Calendar(id="temp", name="Temp"))

        deleted = store.delete_calendar("temp")

        assert deleted is True
        assert store.get_calendar("temp") is None

    def test_add_event(self):
        """Test adding an event."""
        store = CalendarStore()
        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Test",
            start=now,
            end=now + timedelta(hours=1),
        )

        store.add_event(event)

        assert store.get_event("test") == event

    def test_update_event(self):
        """Test updating an event."""
        store = CalendarStore()
        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Original",
            start=now,
            end=now + timedelta(hours=1),
        )

        store.add_event(event)
        event.title = "Updated"
        store.update_event(event)

        updated = store.get_event("test")
        assert updated.title == "Updated"

    def test_delete_event(self):
        """Test deleting an event."""
        store = CalendarStore()
        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Test",
            start=now,
            end=now + timedelta(hours=1),
        )

        store.add_event(event)
        deleted = store.delete_event("test")

        assert deleted is True
        assert store.get_event("test") is None

    def test_list_events(self):
        """Test listing events."""
        store = CalendarStore()
        now = datetime.now()

        store.add_event(CalendarEvent(
            id="1",
            title="Event 1",
            start=now,
            end=now + timedelta(hours=1),
        ))
        store.add_event(CalendarEvent(
            id="2",
            title="Event 2",
            start=now + timedelta(hours=2),
            end=now + timedelta(hours=3),
        ))

        events = store.list_events()

        assert len(events) == 2

    def test_list_events_filter_calendar(self):
        """Test filtering events by calendar."""
        store = CalendarStore()
        store.add_calendar(Calendar(id="work", name="Work"))
        now = datetime.now()

        store.add_event(CalendarEvent(
            id="1",
            title="Default Event",
            start=now,
            end=now + timedelta(hours=1),
            calendar_id="default",
        ))
        store.add_event(CalendarEvent(
            id="2",
            title="Work Event",
            start=now,
            end=now + timedelta(hours=1),
            calendar_id="work",
        ))

        events = store.list_events(calendar_id="work")

        assert len(events) == 1
        assert events[0].title == "Work Event"

    def test_list_events_filter_date(self):
        """Test filtering events by date."""
        store = CalendarStore()
        now = datetime.now()

        store.add_event(CalendarEvent(
            id="1",
            title="Past",
            start=now - timedelta(days=2),
            end=now - timedelta(days=2) + timedelta(hours=1),
        ))
        store.add_event(CalendarEvent(
            id="2",
            title="Future",
            start=now + timedelta(days=1),
            end=now + timedelta(days=1) + timedelta(hours=1),
        ))

        events = store.list_events(start=now)

        assert len(events) == 1
        assert events[0].title == "Future"

    def test_find_free_slots(self):
        """Test finding free slots."""
        store = CalendarStore()
        now = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

        # Add an event from 10-11
        store.add_event(CalendarEvent(
            id="1",
            title="Meeting",
            start=now + timedelta(hours=1),
            end=now + timedelta(hours=2),
        ))

        slots = store.find_free_slots(
            start=now,
            end=now + timedelta(hours=4),
            duration_minutes=60,
        )

        # Should have slots: 9-10, 11-12, 12-13
        assert len(slots) >= 3


class TestCreateEventTool:
    """Tests for CreateEventTool."""

    @pytest.fixture
    def store(self):
        """Create store."""
        return CalendarStore()

    @pytest.mark.asyncio
    async def test_create_event(self, store):
        """Test creating an event."""
        tool = CreateEventTool(store)
        now = datetime.now()

        result = await tool.execute(
            CreateEventInput(
                title="Test Event",
                start=now.isoformat(),
                end=(now + timedelta(hours=1)).isoformat(),
            )
        )

        assert result.success is True
        assert result.event_id is not None
        assert result.event["title"] == "Test Event"

    @pytest.mark.asyncio
    async def test_create_event_invalid_dates(self, store):
        """Test creating event with invalid dates."""
        tool = CreateEventTool(store)

        result = await tool.execute(
            CreateEventInput(
                title="Test",
                start="invalid",
                end="also-invalid",
            )
        )

        assert result.success is False
        assert "Invalid date format" in result.error

    @pytest.mark.asyncio
    async def test_create_event_end_before_start(self, store):
        """Test creating event with end before start."""
        tool = CreateEventTool(store)
        now = datetime.now()

        result = await tool.execute(
            CreateEventInput(
                title="Test",
                start=now.isoformat(),
                end=(now - timedelta(hours=1)).isoformat(),
            )
        )

        assert result.success is False
        assert "after start" in result.error

    @pytest.mark.asyncio
    async def test_create_event_calendar_not_found(self, store):
        """Test creating event in non-existent calendar."""
        tool = CreateEventTool(store)
        now = datetime.now()

        result = await tool.execute(
            CreateEventInput(
                title="Test",
                start=now.isoformat(),
                end=(now + timedelta(hours=1)).isoformat(),
                calendar_id="nonexistent",
            )
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_create_event_with_attendees(self, store):
        """Test creating event with attendees."""
        tool = CreateEventTool(store)
        now = datetime.now()

        result = await tool.execute(
            CreateEventInput(
                title="Meeting",
                start=now.isoformat(),
                end=(now + timedelta(hours=1)).isoformat(),
                attendees=["user1@example.com", "user2@example.com"],
            )
        )

        assert result.success is True
        assert len(result.event["attendees"]) == 2


class TestGetEventTool:
    """Tests for GetEventTool."""

    @pytest.fixture
    def store_with_event(self):
        """Create store with an event."""
        store = CalendarStore()
        now = datetime.now()
        store.add_event(CalendarEvent(
            id="test-event",
            title="Test Event",
            start=now,
            end=now + timedelta(hours=1),
        ))
        return store

    @pytest.mark.asyncio
    async def test_get_event(self, store_with_event):
        """Test getting an event."""
        tool = GetEventTool(store_with_event)

        result = await tool.execute(
            GetEventInput(event_id="test-event")
        )

        assert result.success is True
        assert result.event["title"] == "Test Event"

    @pytest.mark.asyncio
    async def test_get_event_not_found(self, store_with_event):
        """Test getting non-existent event."""
        tool = GetEventTool(store_with_event)

        result = await tool.execute(
            GetEventInput(event_id="nonexistent")
        )

        assert result.success is False
        assert "not found" in result.error


class TestListEventsTool:
    """Tests for ListEventsTool."""

    @pytest.fixture
    def store_with_events(self):
        """Create store with events."""
        store = CalendarStore()
        now = datetime.now()

        store.add_event(CalendarEvent(
            id="1",
            title="Event 1",
            start=now,
            end=now + timedelta(hours=1),
        ))
        store.add_event(CalendarEvent(
            id="2",
            title="Event 2",
            start=now + timedelta(hours=2),
            end=now + timedelta(hours=3),
        ))

        return store

    @pytest.mark.asyncio
    async def test_list_all_events(self, store_with_events):
        """Test listing all events."""
        tool = ListEventsTool(store_with_events)

        result = await tool.execute(ListEventsInput())

        assert result.success is True
        assert result.count == 2

    @pytest.mark.asyncio
    async def test_list_events_with_filter(self, store_with_events):
        """Test listing events with date filter."""
        tool = ListEventsTool(store_with_events)
        now = datetime.now()

        result = await tool.execute(
            ListEventsInput(
                start=(now + timedelta(hours=1)).isoformat(),
            )
        )

        assert result.success is True
        assert result.count >= 1


class TestUpdateEventTool:
    """Tests for UpdateEventTool."""

    @pytest.fixture
    def store_with_event(self):
        """Create store with an event."""
        store = CalendarStore()
        now = datetime.now()
        store.add_event(CalendarEvent(
            id="test-event",
            title="Original Title",
            start=now,
            end=now + timedelta(hours=1),
        ))
        return store

    @pytest.mark.asyncio
    async def test_update_event(self, store_with_event):
        """Test updating an event."""
        tool = UpdateEventTool(store_with_event)

        result = await tool.execute(
            UpdateEventInput(
                event_id="test-event",
                title="Updated Title",
                location="Room 101",
            )
        )

        assert result.success is True
        assert result.event["title"] == "Updated Title"
        assert result.event["location"] == "Room 101"

    @pytest.mark.asyncio
    async def test_update_event_not_found(self, store_with_event):
        """Test updating non-existent event."""
        tool = UpdateEventTool(store_with_event)

        result = await tool.execute(
            UpdateEventInput(
                event_id="nonexistent",
                title="New Title",
            )
        )

        assert result.success is False
        assert "not found" in result.error


class TestDeleteEventTool:
    """Tests for DeleteEventTool."""

    @pytest.fixture
    def store_with_event(self):
        """Create store with an event."""
        store = CalendarStore()
        now = datetime.now()
        store.add_event(CalendarEvent(
            id="test-event",
            title="Test",
            start=now,
            end=now + timedelta(hours=1),
        ))
        return store

    @pytest.mark.asyncio
    async def test_delete_event(self, store_with_event):
        """Test deleting an event."""
        tool = DeleteEventTool(store_with_event)

        result = await tool.execute(
            DeleteEventInput(event_id="test-event")
        )

        assert result.success is True
        assert store_with_event.get_event("test-event") is None

    @pytest.mark.asyncio
    async def test_delete_event_not_found(self, store_with_event):
        """Test deleting non-existent event."""
        tool = DeleteEventTool(store_with_event)

        result = await tool.execute(
            DeleteEventInput(event_id="nonexistent")
        )

        assert result.success is False
        assert "not found" in result.error


class TestFindFreeSlotsTool:
    """Tests for FindFreeSlotsTool."""

    @pytest.fixture
    def store_with_events(self):
        """Create store with events."""
        store = CalendarStore()
        now = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

        # Add events at 10-11 and 14-15
        store.add_event(CalendarEvent(
            id="1",
            title="Meeting 1",
            start=now + timedelta(hours=1),
            end=now + timedelta(hours=2),
        ))
        store.add_event(CalendarEvent(
            id="2",
            title="Meeting 2",
            start=now + timedelta(hours=5),
            end=now + timedelta(hours=6),
        ))

        return store, now

    @pytest.mark.asyncio
    async def test_find_free_slots(self, store_with_events):
        """Test finding free slots."""
        store, now = store_with_events
        tool = FindFreeSlotsTool(store)

        result = await tool.execute(
            FindFreeSlotsInput(
                start=now.isoformat(),
                end=(now + timedelta(hours=8)).isoformat(),
                duration_minutes=60,
            )
        )

        assert result.success is True
        assert result.count >= 3

    @pytest.mark.asyncio
    async def test_find_free_slots_invalid_dates(self, store_with_events):
        """Test with invalid dates."""
        store, _ = store_with_events
        tool = FindFreeSlotsTool(store)

        result = await tool.execute(
            FindFreeSlotsInput(
                start="invalid",
                end="also-invalid",
                duration_minutes=60,
            )
        )

        assert result.success is False
        assert "Invalid date format" in result.error


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_calendar_store(self):
        """Test creating calendar store."""
        store = create_calendar_store()

        assert isinstance(store, CalendarStore)
        assert store.get_calendar("default") is not None

    def test_create_calendar_tools(self):
        """Test creating all calendar tools."""
        store = create_calendar_store()
        tools = create_calendar_tools(store)

        assert "create_calendar_event" in tools
        assert "get_calendar_event" in tools
        assert "list_calendar_events" in tools
        assert "update_calendar_event" in tools
        assert "delete_calendar_event" in tools
        assert "find_free_slots" in tools
