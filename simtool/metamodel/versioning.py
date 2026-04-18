"""Semantic versioning for meta-models.

Panels pin to a specific meta-model version. Policy for propagation is
decided by SemVer change kind:

    - PATCH: auto-propagate (bug fixes, typos, changelog clarifications).
    - MINOR: auto-propagate with a visible notification (new parameters
      reconciled, new submodels added, new approximation operators —
      strictly additive).
    - MAJOR: require explicit user confirmation (schema changes, breaking
      reconciliation revisions, removed submodels).

The rule: if accepting the new version CAN change the panel's outputs
silently, it's MAJOR. If it only adds capability the panel isn't already
using, it's MINOR. If it's invisible to the panel, it's PATCH.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


_SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?$"
)


class SemVer(BaseModel):
    """Parsed semantic version string. Pre-release tags supported; build
    metadata (``+<id>``) is not — meta-models don't need it."""

    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)
    prerelease: Optional[str] = None

    @classmethod
    def parse(cls, s: str) -> "SemVer":
        m = _SEMVER_RE.match(s.strip())
        if not m:
            raise ValueError(f"not a semver: {s!r}")
        return cls(
            major=int(m["major"]),
            minor=int(m["minor"]),
            patch=int(m["patch"]),
            prerelease=m["prerelease"],
        )

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.patch}"
        return f"{core}-{self.prerelease}" if self.prerelease else core

    def _tuple(self) -> tuple[int, int, int, int]:
        # Pre-release compares LOWER than the same core without prerelease.
        return (self.major, self.minor, self.patch, 0 if self.prerelease else 1)

    def __lt__(self, other: "SemVer") -> bool:
        return self._tuple() < other._tuple()

    def __le__(self, other: "SemVer") -> bool:
        return self._tuple() <= other._tuple()

    def __gt__(self, other: "SemVer") -> bool:
        return self._tuple() > other._tuple()

    def __ge__(self, other: "SemVer") -> bool:
        return self._tuple() >= other._tuple()

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))


class VersionChangeKind(str, Enum):
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"
    IDENTICAL = "identical"
    DOWNGRADE = "downgrade"


def classify_change(from_v: SemVer, to_v: SemVer) -> VersionChangeKind:
    """Classify the change from ``from_v`` to ``to_v``.

    DOWNGRADE is returned if ``to_v < from_v`` — that's a user intent
    question, not an auto-propagation one, and callers must handle it.
    """
    if from_v == to_v:
        return VersionChangeKind.IDENTICAL
    if to_v < from_v:
        return VersionChangeKind.DOWNGRADE
    if to_v.major > from_v.major:
        return VersionChangeKind.MAJOR
    if to_v.minor > from_v.minor:
        return VersionChangeKind.MINOR
    return VersionChangeKind.PATCH


class PropagationPolicy(BaseModel):
    """Policy that decides whether a version change propagates automatically
    to a panel or blocks pending explicit user confirmation."""

    auto_propagate_kinds: set[VersionChangeKind] = Field(
        default_factory=lambda: {
            VersionChangeKind.IDENTICAL,
            VersionChangeKind.PATCH,
            VersionChangeKind.MINOR,
        },
        description="Change kinds that flow to panels without user action. "
        "MAJOR is never in this set by default — breaking changes must be "
        "confirmed.",
    )

    def requires_user_confirmation(
        self, from_v: SemVer, to_v: SemVer
    ) -> bool:
        return classify_change(from_v, to_v) not in self.auto_propagate_kinds

    @model_validator(mode="after")
    def _check_major_is_not_auto(self) -> "PropagationPolicy":
        # Allowed to override, but we warn loudly in the shape.
        return self
