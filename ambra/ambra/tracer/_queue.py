"""Queue full policy enum for subscriber queues."""

from enum import Enum, auto

DEFAULT_MAX_QUEUE_SIZE = 32


class QueueFullPolicy(Enum):
    """Policy for handling full queues in subscribers.

    WAIT: Block until space is available in the queue.
    DROP: Drop the event silently if the queue is full.
    """

    WAIT = auto()
    DROP = auto()
