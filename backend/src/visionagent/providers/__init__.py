"""External I/O adapters: the model, embedding, search and rerank clients.

These are not utilities. Each holds a connection, an API key, a loaded model or
a session, and each is the thing a deployment swaps to change vendor. They sit
below service/ so a slot can call one, and beside database/ rather than inside
it because a provider is a remote capability, not a store this system owns.
"""
