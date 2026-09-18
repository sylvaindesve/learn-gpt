def print_title(title: str, underline_with: str = "=") -> None:
    print(title)
    print(underline_with * len(title))


def print_new_line() -> None:
    print()


def print_indented(text: str, tabs: int = 0) -> None:
    print("  " * tabs + text)
