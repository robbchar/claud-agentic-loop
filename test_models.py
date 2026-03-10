"""Tests for models.py — SwarmState and AgentResult dataclasses."""

import pytest
from models import SwarmState, AgentResult


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(output="hello")
        assert r.output == "hello"
        assert r.passed is True
        assert r.feedback is None

    def test_explicit_values(self):
        r = AgentResult(output="x", passed=False, feedback="fix this")
        assert r.passed is False
        assert r.feedback == "fix this"


class TestSwarmState:
    def test_all_fields_have_defaults(self):
        # Should construct with no arguments
        state = SwarmState()
        assert state.feature_request == ""
        assert state.architecture == ""
        assert state.tasks_doc == ""
        assert state.requirements is None
        assert state.code is None
        assert state.qa_report is None
        assert state.review is None
        assert state.feedback is None
        assert state.approved is False
        assert state.dev_messages == []
        assert state.history == []

    def test_feature_request_optional(self):
        state = SwarmState()
        assert state.feature_request == ""

    def test_feature_request_explicit(self):
        state = SwarmState(feature_request="build something")
        assert state.feature_request == "build something"

    def test_spec_fields(self):
        state = SwarmState(architecture="arch content", tasks_doc="tasks content")
        assert state.architecture == "arch content"
        assert state.tasks_doc == "tasks content"

    def test_mutable_defaults_are_independent(self):
        # dev_messages and history must not share the same list across instances
        s1 = SwarmState()
        s2 = SwarmState()
        s1.dev_messages.append("msg")
        s1.history.append("event")
        assert s2.dev_messages == []
        assert s2.history == []

    def test_mutation(self):
        state = SwarmState(feature_request="req")
        state.requirements = "some requirements"
        state.approved = True
        assert state.requirements == "some requirements"
        assert state.approved is True
