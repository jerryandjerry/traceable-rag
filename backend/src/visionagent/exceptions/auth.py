class AuthError(Exception):
    """An expected authentication failure whose message is safe for clients."""

    def __init__(self, public_message: str) -> None:
        self.public_message = public_message
        super().__init__(public_message)
