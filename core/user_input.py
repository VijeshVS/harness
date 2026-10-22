EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def get_user_query(query: str | None, allow_exit: bool = False) -> str | None:
    if query:
        return query

    while True:
        try:
            text = input("User Query: ")
        except EOFError:
            print("\n[system] End of input. Exiting...")
            return None

        text = text.strip()
        if not text:
            continue
        if allow_exit and text.lower() in EXIT_COMMANDS:
            return None
        return text
