"""Give each environment its own stream of Python's global `random` module.

Catanatron draws seating, the board, the development deck's shuffle, dice,
robber steals and random discards from the *global* `random` module. Two
games stepped alternately in one process therefore share one stream, and
neither reproduces from its seed — measured directly: a game interleaved with
another diverges from the same game played alone. That breaks vectorised
training in a single process.

`RandomStream` keeps a private copy of the global generator's state and swaps
it in only for the duration of an engine call, restoring the caller's state
afterwards. Each environment then sees an uninterrupted stream of its own.
"""

import random
from collections.abc import Iterator
from contextlib import contextmanager


class RandomStream:
    """A private stream of the global `random` module.

    Attributes:
        state: The stream's saved generator state while inactive.
    """

    def __init__(self) -> None:
        """Start the stream from fresh OS entropy, independent of any seed."""
        self.state = random.Random().getstate()

    @contextmanager
    def active(self) -> Iterator[None]:
        """Run a block with this stream installed as the global generator.

        Yields:
            Nothing; engine calls inside the block draw from this stream, and
            any reseeding inside it (such as `Game.__init__`'s) reseeds only
            this stream.
        """
        outer = random.getstate()
        random.setstate(self.state)
        try:
            yield
        finally:
            self.state = random.getstate()
            random.setstate(outer)
