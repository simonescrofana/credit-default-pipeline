"""Test suite for the CLI entry point.

Covers a single conversational turn (the compiled agent is invoked and its
final_answer is appended to history), an exit command ending the loop, an
empty input being skipped without invoking the agent, and a
KeyboardInterrupt ending the loop cleanly instead of propagating.

"""

from unittest.mock import MagicMock, patch

from agent.run_agent import run_agent


@patch("agent.run_agent.build_agent")
@patch("builtins.input", side_effect=["Ciao!", "quit"])
def test_run_agent_single_turn_then_exit(mock_input, mock_build_agent, capsys) -> None:
    """Verify a turn invokes the agent and the exit command ends the loop."""
    mock_compiled = MagicMock()
    seen_calls = []

    def _fake_invoke(payload):
        seen_calls.append(
            {"user_input": payload["user_input"], "messages": list(payload["messages"])}
        )
        return {"final_answer": "Ciao, come posso aiutarti?"}

    mock_compiled.invoke.side_effect = _fake_invoke
    mock_build_agent.return_value = mock_compiled

    run_agent()

    assert seen_calls == [{"user_input": "Ciao!", "messages": []}]
    output = capsys.readouterr().out
    assert "Ciao, come posso aiutarti?" in output
    assert "Goodbye! / Arrivederci!" in output


@patch("agent.run_agent.build_agent")
@patch("builtins.input", side_effect=["", "quit"])
def test_run_agent_skips_empty_input(mock_input, mock_build_agent) -> None:
    """Verify an empty input is skipped without invoking the agent."""
    mock_compiled = MagicMock()
    mock_build_agent.return_value = mock_compiled

    run_agent()

    mock_compiled.invoke.assert_not_called()


@patch("agent.run_agent.build_agent")
@patch("builtins.input", side_effect=["esci"])
def test_run_agent_accepts_italian_exit_command(mock_input, mock_build_agent) -> None:
    """Verify the Italian exit command also ends the loop."""
    mock_compiled = MagicMock()
    mock_build_agent.return_value = mock_compiled

    run_agent()

    mock_compiled.invoke.assert_not_called()


@patch("agent.run_agent.build_agent")
@patch("builtins.input", side_effect=KeyboardInterrupt())
def test_run_agent_handles_keyboard_interrupt(
    mock_input, mock_build_agent, capsys
) -> None:
    """Verify Ctrl+C ends the loop cleanly instead of propagating."""
    mock_compiled = MagicMock()
    mock_build_agent.return_value = mock_compiled

    run_agent()  # must not raise

    output = capsys.readouterr().out
    assert "Goodbye! / Arrivederci!" in output


@patch("agent.run_agent.build_agent")
@patch("builtins.input", side_effect=["Rischia default Rossi SRL?", "Grazie!", "quit"])
def test_run_agent_accumulates_conversation_history(
    mock_input, mock_build_agent
) -> None:
    """Verify each turn's user input and answer are appended to history."""
    mock_compiled = MagicMock()
    seen_messages_per_call = []

    def _fake_invoke(payload):
        seen_messages_per_call.append(list(payload["messages"]))
        answers = ["Rossi SRL has a low default risk.", "You're welcome!"]
        return {"final_answer": answers[len(seen_messages_per_call) - 1]}

    mock_compiled.invoke.side_effect = _fake_invoke
    mock_build_agent.return_value = mock_compiled

    run_agent()

    assert seen_messages_per_call[0] == []
    assert seen_messages_per_call[1] == [
        {"role": "user", "content": "Rischia default Rossi SRL?"},
        {"role": "assistant", "content": "Rossi SRL has a low default risk."},
    ]
