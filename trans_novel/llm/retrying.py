"""LLM retry errors."""


class EmptyResponseError(Exception):
    """A provider response did not contain usable message content."""
