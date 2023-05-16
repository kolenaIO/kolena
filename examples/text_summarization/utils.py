import re


def get_readable(text: str) -> str:
    # no spaces before periods, only after
    return re.sub(r"\s+(\.)", r"\1", text)
