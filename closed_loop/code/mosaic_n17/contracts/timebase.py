"""Control-tick contract for the SONIC78 route.

The deployment loop consumes **one new 78D token per 50 Hz tick** and replans
every ``execute_steps`` ticks (the official inference runner uses 2.5 Hz, i.e.
20 of 40 steps).  ``frame_index`` on the wire is log-only: it may never select,
repeat, order or interpolate control frames.  A new plan therefore may only
start strictly after the last consumed tick; rewriting a consumed tick is a
contract violation, not something to clip.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sonic78 import SONIC78_V1, Sonic78Spec


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    return value


@dataclass(frozen=True)
class TickSchedule:
    """Fixed 50 Hz execution with a deterministic replan cadence."""

    name: str = SONIC78_V1.name
    control_hz: int = SONIC78_V1.control_hz
    plan_horizon: int = SONIC78_V1.horizon
    execute_steps: int = SONIC78_V1.execute_steps

    def __post_init__(self) -> None:
        if self.control_hz != 50:
            raise ValueError(f"SONIC executes at 50 Hz, got {self.control_hz}")
        if self.execute_steps <= 0 or self.execute_steps > self.plan_horizon:
            raise ValueError("execute_steps must be in (0, plan_horizon]")
        if self.plan_horizon % self.execute_steps:
            raise ValueError("plan_horizon must be an integer multiple of execute_steps")

    @property
    def replan_hz(self) -> float:
        return self.control_hz / self.execute_steps

    @property
    def execute_seconds(self) -> float:
        return self.execute_steps / self.control_hz

    @classmethod
    def from_spec(cls, spec: Sonic78Spec = SONIC78_V1) -> TickSchedule:
        return cls(
            name=spec.name,
            control_hz=spec.control_hz,
            plan_horizon=spec.horizon,
            execute_steps=spec.execute_steps,
        )

    def ticks_into_plan(self, tick: int, observation_tick: int) -> int:
        """Offset of ``tick`` inside the plan started at ``observation_tick``."""

        offset = _require_int(tick, "tick") - _require_int(observation_tick, "observation_tick")
        if not 0 <= offset < self.plan_horizon:
            raise ValueError(
                f"tick {tick} is outside plan [{observation_tick}, "
                f"{observation_tick + self.plan_horizon})"
            )
        return offset


@dataclass(frozen=True)
class PlanCommitment:
    """Identity of one plan and the ticks it already consumed.

    ``consumed_ticks`` is append-only in deployment: once a tick was executed
    with a token, no later plan may replace that token.  A new commitment must
    start after the last consumed tick (see :meth:`assert_can_replace`).
    """

    plan_id: str
    observation_tick: int
    ready_tick: int
    first_executable_tick: int
    prefix_len: int
    consumed_ticks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("plan_id must be a non-empty string")
        for field in ("observation_tick", "ready_tick", "first_executable_tick", "prefix_len"):
            _require_int(getattr(self, field), field)
        if self.ready_tick < self.observation_tick:
            raise ValueError("ready_tick must not precede observation_tick")
        if self.first_executable_tick < self.ready_tick:
            raise ValueError("first_executable_tick must not precede ready_tick")
        if self.prefix_len < 0:
            raise ValueError("prefix_len must be non-negative")
        ticks = tuple(self.consumed_ticks)
        if any(isinstance(tick, bool) or not isinstance(tick, int) for tick in ticks):
            raise TypeError("consumed_ticks must be ints")
        if any(later <= earlier for earlier, later in zip(ticks, ticks[1:])):
            raise ValueError("consumed_ticks must be strictly increasing")
        if any(tick < self.first_executable_tick for tick in ticks):
            raise ValueError("consumed ticks must not precede first_executable_tick")
        object.__setattr__(self, "consumed_ticks", ticks)

    @property
    def last_consumed_tick(self) -> int:
        return self.consumed_ticks[-1] if self.consumed_ticks else self.first_executable_tick - 1

    def assert_can_replace(
        self, previous: PlanCommitment, *, schedule: TickSchedule | None = None
    ) -> None:
        """Fail closed when a new plan would rewrite an executed tick."""

        schedule = TickSchedule.from_spec() if schedule is None else schedule
        if self.observation_tick <= previous.last_consumed_tick:
            raise ValueError(
                f"plan {self.plan_id} observed tick {self.observation_tick} at or before the "
                f"last consumed tick {previous.last_consumed_tick} of {previous.plan_id}"
            )
        if self.first_executable_tick <= previous.last_consumed_tick:
            raise ValueError(
                f"plan {self.plan_id} starts at tick {self.first_executable_tick}, which is not "
                f"after the last consumed tick {previous.last_consumed_tick} of {previous.plan_id}"
            )
        if self.prefix_len > schedule.plan_horizon:
            raise ValueError("prefix_len exceeds the plan horizon")

    def consume(self, tick: int) -> PlanCommitment:
        """Return a copy with ``tick`` appended to the consumed set."""

        tick = _require_int(tick, "tick")
        if self.consumed_ticks and tick <= self.consumed_ticks[-1]:
            raise ValueError(f"tick {tick} does not follow {self.consumed_ticks[-1]}")
        return PlanCommitment(
            plan_id=self.plan_id,
            observation_tick=self.observation_tick,
            ready_tick=self.ready_tick,
            first_executable_tick=self.first_executable_tick,
            prefix_len=self.prefix_len,
            consumed_ticks=self.consumed_ticks + (tick,),
        )


__all__ = ["PlanCommitment", "TickSchedule"]
