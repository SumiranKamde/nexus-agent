import asyncio
import unittest
from types import SimpleNamespace

from agent_core.agent import NexusAgent, STATE_CHANGING_TOOLS


class FakeTodoSession:
    async def list_tools(self):
        return SimpleNamespace(tools=[])

    async def call_tool(self, _name, _args):
        return SimpleNamespace(content=[SimpleNamespace(text="[]")])


class FakeGroqClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        message = SimpleNamespace(content="Groq fallback works", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentSafetyTests(unittest.TestCase):
    def test_all_local_mutations_require_confirmation(self):
        expected = {
            "remember", "forget", "create_notification",
            "mark_notifications_seen", "send_whatsapp", "send_file_via_whatsapp",
        }
        self.assertTrue(expected.issubset(STATE_CHANGING_TOOLS))

    def test_groq_plan_returns_a_normal_response(self):
        agent = object.__new__(NexusAgent)
        agent.groq_client = FakeGroqClient()
        agent.sessions = {"todo": FakeTodoSession()}

        result = asyncio.run(agent._plan_groq("hello"))

        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.text, "Groq fallback works")
        self.assertEqual(result.function_calls, [])


if __name__ == "__main__":
    unittest.main()
