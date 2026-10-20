def get_user_query(query: str | None) -> str:
    if query:
        return query
    return input("User Query: ")
