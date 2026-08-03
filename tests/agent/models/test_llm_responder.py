"""Test suite for the responder LLM factory.

Cover the happy path of instantiating the Groq-hosted chat model with the
default temperature and with an explicit one, without ever making a real
call to Groq.

"""

from unittest.mock import MagicMock, patch

from agent.models.llm_responder import RESPONDER_MODEL_NAME, get_responder_llm


@patch("agent.models.llm_responder.settings")
@patch("agent.models.llm_responder.ChatGroq")
def test_get_responder_llm_default_temperature(
    mock_chat_groq_class, mock_settings
) -> None:
    """Verify get_responder_llm defaults to temperature=0.7."""
    mock_settings.GROQ_API_KEY.get_secret_value.return_value = "fake-api-key"
    fake_llm = MagicMock()
    mock_chat_groq_class.return_value = fake_llm

    result = get_responder_llm()

    assert result is fake_llm
    mock_chat_groq_class.assert_called_once_with(
        model=RESPONDER_MODEL_NAME,
        temperature=0.7,
        api_key="fake-api-key",
    )


@patch("agent.models.llm_responder.settings")
@patch("agent.models.llm_responder.ChatGroq")
def test_get_responder_llm_explicit_temperature(
    mock_chat_groq_class, mock_settings
) -> None:
    """Verify an explicit temperature overrides the default."""
    mock_settings.GROQ_API_KEY.get_secret_value.return_value = "fake-api-key"
    fake_llm = MagicMock()
    mock_chat_groq_class.return_value = fake_llm

    result = get_responder_llm(temperature=0.0)

    assert result is fake_llm
    mock_chat_groq_class.assert_called_once_with(
        model=RESPONDER_MODEL_NAME,
        temperature=0.0,
        api_key="fake-api-key",
    )
