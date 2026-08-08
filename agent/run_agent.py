"""CLI entry point for the credit default insolvency prediction agent.

Runs an interactive conversation loop: reads the user's input, invokes the
compiled agent graph with the running message history, prints the final
answer, and appends the turn to history before asking for the next input.
The loop ends when the user types an exit command, or on Ctrl+c.

"""

import logging
import os

os.environ["ORT_LOGGING_LEVEL"] = "3"  # 3 = ERROR, silenzia WARNING e INFO

from agent.graph import build_agent

EXIT_COMMANDS = {"quit", "exit", "esci"}

WELCOME_MESSAGE = """\
Welcome to the Credit Default Prediction Agent, built by Simone Scrofana.
Project repository: https://github.com/simonescrofana/credit-default-pipeline
Ask a question to get started.

Benvenuto nell'agente del progetto Credit Default Prediction, creato da Simone Scrofana.
Repository del progetto: https://github.com/simonescrofana/credit-default-pipeline
Scrivi una domanda per iniziare.
"""

EXIT_HINT = (
    "(Type 'quit', 'exit', or 'esci' to end the chat / "
    "Scrivi 'quit', 'exit' o 'esci' per uscire dalla chat)"
)

# hides a log in the chat with the AI agent
logging.getLogger("agent.utils.llm_utils").setLevel(logging.ERROR)


def run_agent() -> None:
    """Run the interactive CLI loop for the agent."""
    compiled_agent = build_agent()
    conversation_history: list[dict[str, str]] = []

    print(WELCOME_MESSAGE)

    try:
        while True:
            print(EXIT_HINT)
            user_input = input("> ").strip()

            if user_input.lower() in EXIT_COMMANDS:
                print("Goodbye! / Arrivederci!")
                break

            # logic for empty prompts
            if not user_input:
                continue

            result = compiled_agent.invoke(
                {"user_input": user_input, "messages": conversation_history}
            )
            final_answer = result.get("final_answer", "")
            print(f"\n{final_answer}\n")

            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": final_answer})
    except KeyboardInterrupt:
        print("\nGoodbye! / Arrivederci!")


if __name__ == "__main__":
    import os

    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    run_agent()
