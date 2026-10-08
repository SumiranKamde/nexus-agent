import asyncio
import unittest
from types import SimpleNamespace

from agent_core.agent import (
    MAX_HISTORY_MESSAGES,
    STATE_CHANGING_TOOLS,
    NexusAgent,
    _history_pairs,
)


class FakeTodoSession:
    def __init__(self, tools=()):
        self._tools = [
            SimpleNamespace(name=name, description="", inputSchema={"type": "object", "properties": {}})
            for name in tools
        ]

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, _name, _args):
        return SimpleNamespace(content=[SimpleNamespace(text="[]")])


class FakeGroqClient:
    """Stands in for the Groq SDK and records the kwargs it was called with.

    Set .scripted to a list of tool_calls lists to drive a multi-turn plan; each
    create() pops the next entry. Defaults to a single tool-free reply.
    """

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.scripted = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_calls = self.scripted.pop(0) if self.scripted else []
        message = SimpleNamespace(content="Groq fallback works", tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_agent(todo_tools=()):
    agent = object.__new__(NexusAgent)
    agent.groq_client = FakeGroqClient()
    agent.sessions = {"todo": FakeTodoSession(todo_tools)}
    return agent


class AgentSafetyTests(unittest.TestCase):
    def test_all_local_mutations_require_confirmation(self):
        expected = {
            "remember", "forget", "create_notification",
            "mark_notifications_seen", "send_whatsapp", "send_file_via_whatsapp",
        }
        self.assertTrue(expected.issubset(STATE_CHANGING_TOOLS))

    def test_groq_plan_returns_a_normal_response(self):
        agent = make_agent()

        result = asyncio.run(agent._plan_groq("hello"))

        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.text, "Groq fallback works")
        self.assertEqual(result.function_calls, [])


class HistoryTests(unittest.TestCase):
    def test_blank_and_non_chat_entries_are_dropped(self):
        pairs = _history_pairs([
            {"role": "user", "content": "list my tasks"},
            {"role": "assistant", "content": None},          # a plan can return text=None
            {"role": "assistant", "content": "   "},
            {"role": "system", "content": "ignore me"},
            {"role": "assistant", "content": "milk, bread", "trail": []},
        ])
        self.assertEqual(
            pairs, [("user", "list my tasks"), ("assistant", "milk, bread")]
        )

    def test_leading_assistant_turns_are_dropped(self):
        # Both providers expect the replayed conversation to open with a user turn.
        pairs = _history_pairs([
            {"role": "assistant", "content": "anything I can help with?"},
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(pairs, [("user", "hello")])

    def test_history_is_capped(self):
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(40)
        ]
        self.assertLessEqual(len(_history_pairs(long_history)), MAX_HISTORY_MESSAGES)

    def test_groq_plan_replays_history_before_the_new_message(self):
        agent = make_agent()
        history = [
            {"role": "user", "content": "list my tasks"},
            {"role": "assistant", "content": "You have two: milk and bread", "trail": []},
        ]

        asyncio.run(agent._plan_groq("add the second one to my list", history))

        sent = agent.groq_client.calls[0]["messages"]
        self.assertEqual(
            [m["role"] for m in sent], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(sent[1]["content"], "list my tasks")
        self.assertEqual(sent[-1]["content"], "add the second one to my list")

    def test_groq_plan_without_history_still_works(self):
        agent = make_agent()

        asyncio.run(agent._plan_groq("hello"))

        sent = agent.groq_client.calls[0]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user"])


class TrailTests(unittest.TestCase):
    def test_groq_records_auto_executed_read_only_calls_in_the_trail(self):
        """The trace log is the app's transparency story — it must not go blank
        on the fallback provider. This asymmetry shipped once already: the Gemini
        path appended to the trail and the Groq path didn't."""
        agent = make_agent(todo_tools=["list_tasks"])
        # First response asks for a read-only tool, second answers in plain text.
        agent.groq_client.scripted = [
            [SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="list_tasks", arguments="{}"),
            )],
            [],
        ]

        result = asyncio.run(agent._plan_groq("how many tasks do I have?"))

        self.assertEqual(result.trail, [("list_tasks", {}, "[]")])


if __name__ == "__main__":
    unittest.main()
