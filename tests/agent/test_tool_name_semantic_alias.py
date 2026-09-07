"""Regression tests for the semantic alias layer in repair_tool_call.

This file binds to the alias table introduced to map names like `shell`,
`bash`, `sh`, `zsh`, `shell_command`, `run_shell`, `run_command`,
`exec_command`, `execute_command`, and `run_terminal_cmd` to `terminal`,
and `grep` to `search_files`, even though they are lexically unrelated
(e.g. 'shell' vs 'terminal' score only 0.3077 on difflib
SequenceMatcher).
"""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

REPO_ROOT = "/Users/kepler/.hermes/hermes-agent"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.agent_runtime_helpers import repair_tool_call  # noqa: E402


class TestSemanticShellAliases(unittest.TestCase):
    """Shell-vocabulary synonyms for `terminal`."""

    def _agent(self, tool_names):
        return SimpleNamespace(valid_tool_names=set(tool_names))

    def test_shell_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "shell"),
            "terminal",
        )

    def test_bash_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "bash"),
            "terminal",
        )

    def test_sh_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "sh"),
            "terminal",
        )

    def test_zsh_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "zsh"),
            "terminal",
        )

    def test_shell_command_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "shell_command"),
            "terminal",
        )

    def test_run_shell_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "run_shell"),
            "terminal",
        )

    def test_run_command_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "run_command"),
            "terminal",
        )

    def test_exec_command_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "exec_command"),
            "terminal",
        )

    def test_execute_command_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "execute_command"),
            "terminal",
        )

    def test_run_terminal_cmd_maps_to_terminal(self):
        self.assertEqual(
            repair_tool_call(self._agent({"terminal", "read_file"}), "run_terminal_cmd"),
            "terminal",
        )

    def test_case_variants_map_to_terminal(self):
        for variant in ("Shell", "SHELL", "Shell_Command"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    repair_tool_call(self._agent({"terminal", "read_file"}), variant),
                    "terminal",
                )

    def test_hyphen_forms_map_to_terminal(self):
        for variant in ("run-command", "run command", "Run-Command"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    repair_tool_call(self._agent({"terminal", "read_file"}), variant),
                    "terminal",
                )


class TestSemanticGrepAlias(unittest.TestCase):
    """`grep` -> `search_files` mapping."""

    def _agent(self, tool_names):
        return SimpleNamespace(valid_tool_names=set(tool_names))

    def test_grep_maps_to_search_files(self):
        self.assertEqual(
            repair_tool_call(self._agent({"search_files", "read_file"}), "grep"),
            "search_files",
        )

    def test_grep_returns_none_when_search_files_disabled(self):
        self.assertIsNone(
            repair_tool_call(self._agent({"read_file", "web_search"}), "grep")
        )

    def test_grep_precedence_when_grep_is_registered(self):
        self.assertEqual(
            repair_tool_call(self._agent({"grep", "search_files"}), "grep"),
            "grep",
        )


class TestSemanticAliasGuard(unittest.TestCase):
    """The alias table must not invent tools that are not enabled."""

    def _agent(self, tool_names):
        return SimpleNamespace(valid_tool_names=set(tool_names))

    def test_shell_returns_none_when_terminal_disabled(self):
        agent = self._agent({"read_file", "web_search"})
        self.assertIsNone(repair_tool_call(agent, "shell"))


class TestSemanticAliasPrecedence(unittest.TestCase):
    """Exact real-tool names must outrank semantic aliases."""

    def _agent(self, tool_names):
        return SimpleNamespace(valid_tool_names=set(tool_names))

    def test_existing_shell_tool_wins(self):
        agent = self._agent({"shell", "terminal"})
        self.assertEqual(
            repair_tool_call(agent, "shell"),
            "shell",
        )


class TestSemanticAliasRegression(unittest.TestCase):
    """Ensure pre-existing repair behaviour still works."""

    def _agent(self, tool_names):
        return SimpleNamespace(valid_tool_names=set(tool_names))

    def test_todo_tool_suffix_still_strips(self):
        agent = self._agent({"todo"})
        self.assertEqual(
            repair_tool_call(agent, "TodoTool_tool"),
            "todo",
        )

    def test_terminal_exact_match_still_passes(self):
        agent = self._agent({"terminal"})
        self.assertEqual(
            repair_tool_call(agent, "terminal"),
            "terminal",
        )


if __name__ == "__main__":
    unittest.main()
