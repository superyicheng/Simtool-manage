"""Unit tests — SemVer + VersionChangeKind + PropagationPolicy."""

from __future__ import annotations

import pytest

from simtool.metamodel.versioning import (
    PropagationPolicy,
    SemVer,
    VersionChangeKind,
    classify_change,
)


# --- SemVer parsing ----------------------------------------------------------


def test_parse_basic() -> None:
    v = SemVer.parse("1.2.3")
    assert (v.major, v.minor, v.patch, v.prerelease) == (1, 2, 3, None)


def test_parse_prerelease() -> None:
    v = SemVer.parse("0.1.0-rc.1")
    assert v.prerelease == "rc.1"


def test_parse_rejects_garbage() -> None:
    for bad in ["", "x", "1", "1.2", "1.2.3.4", "v1.2.3"]:
        with pytest.raises(ValueError):
            SemVer.parse(bad)


def test_str_round_trip() -> None:
    for s in ["1.2.3", "0.0.0", "10.0.5-beta.2"]:
        assert str(SemVer.parse(s)) == s


# --- Ordering ---------------------------------------------------------------


def test_ordering_major_minor_patch() -> None:
    assert SemVer.parse("1.0.0") > SemVer.parse("0.9.9")
    assert SemVer.parse("1.2.0") > SemVer.parse("1.1.9")
    assert SemVer.parse("1.2.1") > SemVer.parse("1.2.0")


def test_prerelease_ranks_below_core() -> None:
    assert SemVer.parse("1.2.3-rc.1") < SemVer.parse("1.2.3")


def test_equality() -> None:
    assert SemVer.parse("1.2.3") == SemVer.parse("1.2.3")


# --- classify_change --------------------------------------------------------


@pytest.mark.parametrize(
    "from_s,to_s,expected",
    [
        ("1.0.0", "1.0.0", VersionChangeKind.IDENTICAL),
        ("1.0.0", "1.0.1", VersionChangeKind.PATCH),
        ("1.0.0", "1.1.0", VersionChangeKind.MINOR),
        ("1.0.0", "2.0.0", VersionChangeKind.MAJOR),
        ("1.0.0", "1.2.3", VersionChangeKind.MINOR),
        ("1.1.0", "1.0.5", VersionChangeKind.DOWNGRADE),
        ("2.0.0", "1.9.9", VersionChangeKind.DOWNGRADE),
    ],
)
def test_classify_change(from_s: str, to_s: str, expected: VersionChangeKind) -> None:
    assert classify_change(SemVer.parse(from_s), SemVer.parse(to_s)) == expected


# --- Propagation policy ------------------------------------------------------


def test_default_policy_auto_patches_minors_not_majors() -> None:
    p = PropagationPolicy()
    f = SemVer.parse("1.0.0")
    assert not p.requires_user_confirmation(f, SemVer.parse("1.0.1"))
    assert not p.requires_user_confirmation(f, SemVer.parse("1.1.0"))
    assert p.requires_user_confirmation(f, SemVer.parse("2.0.0"))


def test_default_policy_requires_confirmation_for_downgrade() -> None:
    p = PropagationPolicy()
    assert p.requires_user_confirmation(SemVer.parse("1.1.0"), SemVer.parse("1.0.9"))


def test_policy_identical_never_needs_confirmation() -> None:
    p = PropagationPolicy()
    f = SemVer.parse("1.2.3")
    assert not p.requires_user_confirmation(f, f)


def test_policy_customizable() -> None:
    strict = PropagationPolicy(auto_propagate_kinds={VersionChangeKind.IDENTICAL})
    f = SemVer.parse("1.0.0")
    assert strict.requires_user_confirmation(f, SemVer.parse("1.0.1"))
    assert strict.requires_user_confirmation(f, SemVer.parse("1.1.0"))


def test_semver_hashable() -> None:
    s = {SemVer.parse("1.2.3"), SemVer.parse("1.2.3")}
    assert len(s) == 1
