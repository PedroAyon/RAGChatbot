"""
MessagingService: abstract base for any chat client integration
(Telegram, WhatsApp, Signal, ...).

Everything that is not specific to a chat platform lives here:

* **Inbound queue.** Platforms call ``on_message_received`` from their
  receive loop. It only enqueues, so receiving never blocks on the LLM. A
  single worker thread answers messages one at a time, in arrival order.
  One worker is deliberate: the MessagingAgent keeps per-turn state
  (last query result, pending attachments) and is not thread-safe.
* **Commands** (``/connect``) are parsed here and run on the same worker, so
  they can never swap the data source in the middle of someone else's turn.
  The connection is done by ``MessagingAgent.connect``.
* **Replies**: history bookkeeping, timeout/error apologies and the
  "empty reply" fallback.

A platform implements only two things:

* ``_receive_loop``: block, turning inbound platform events into Messages
  passed to ``on_message_received``, until ``self.stopped`` is True.
* ``send_message``: deliver one Message (text, images, files).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional

from ..llm.base import LLMTimeoutError
from ..models import Message, MessageHistory, MessageSource

if TYPE_CHECKING:  # avoid a hard import cycle; only needed for type hints
    from ..agent.messaging_agent import MessagingAgent

logger = logging.getLogger(__name__)

CONNECT_COMMAND = "/connect"
CONNECT_USAGE = f"Please specify a data source. Usage: `{CONNECT_COMMAND} <data_source_id>`"

EMPTY_REPLY_FALLBACK = (
    "I wasn't able to finish working that out. "
    "Could you rephrase or narrow down your question?"
)
TIMEOUT_REPLY = (
    "Sorry, that took too long to work out. "
    "Please try again, or ask a simpler question."
)
ERROR_REPLY = "Sorry, something went wrong while handling your message. Please try again."
BUSY_REPLY = "Sorry, I'm handling too many requests right now. Please try again in a moment."

_QUEUE_POLL_SECONDS = 0.5     # how often an idle worker checks for shutdown
_SHUTDOWN_JOIN_SECONDS = 5.0  # how long run_forever waits for the worker on exit


class MessagingService(ABC):
    def __init__(
        self,
        messaging_agent: "MessagingAgent",
        max_queue_size: int = 50,
    ):
        """
        Args:
            messaging_agent: The conversational agent that answers messages.
            max_queue_size: Max messages waiting for the worker. Beyond that,
                new messages get ``BUSY_REPLY``. 0 means unbounded.
        """
        self.messaging_agent = messaging_agent
        self._histories: Dict[str, MessageHistory] = {}  # only touched by the worker
        self._inbox: "queue.Queue[Message]" = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------

    @property
    def stopped(self) -> bool:
        """True once ``stop()`` was called. Receive loops should exit on it."""
        return self._stop_event.is_set()

    def run_forever(self) -> None:
        """Start the worker and block in the platform's receive loop.

        Returns after ``stop()`` or when the receive loop exits or raises
        (e.g. Ctrl+C). Messages still queued at that point are discarded.
        """
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name=f"{type(self).__name__}-worker", daemon=True
        )
        self._worker.start()
        try:
            self._receive_loop()
        finally:
            self.stop()
            self._worker.join(timeout=_SHUTDOWN_JOIN_SECONDS)
            if self._worker.is_alive():
                logger.warning("Worker still busy at shutdown; abandoning the in-flight turn.")
            dropped = self._inbox.qsize()
            if dropped:
                logger.warning("Shutting down with %d unanswered message(s) in the queue.", dropped)

    def stop(self) -> None:
        """Ask the receive loop and the worker to finish. Safe to call from any thread."""
        self._stop_event.set()

    # -- inbound -------------------------------------------------------

    def on_message_received(self, message: Message) -> None:
        """Queue an inbound message for the worker. Called by platform receive loops.

        Never blocks on the LLM. If the queue is full, the user is told to retry.
        """
        try:
            self._inbox.put_nowait(message)
        except queue.Full:
            logger.warning("Inbox full; rejecting message from user '%s'.", message.user_id)
            self.send_message(self._agent_message(message.user_id, BUSY_REPLY))

    def _worker_loop(self) -> None:
        """Answer queued messages one at a time until stopped."""
        while not self._stop_event.is_set():
            try:
                message = self._inbox.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                self._process(message)
            except Exception:
                logger.exception("Failed to process a message from user '%s'.", message.user_id)

    def _process(self, message: Message) -> None:
        """Run one queued message: a command, or a conversational turn."""
        command_reply = self._handle_command(message.message or "")
        if command_reply is not None:
            reply = self._agent_message(message.user_id, command_reply)  # not kept in history
        else:
            reply = self._generate_reply(message)
        self.send_message(self._ensure_content(reply))

    # -- replies -------------------------------------------------------

    def _generate_reply(self, message: Message) -> Message:
        """Record `message` in the user's history and return the agent's reply.

        Never raises: on a timeout or any other failure the user gets a short
        apology instead of silence. Replies, apologies included, are recorded
        in the history.
        """
        history = self._get_history(message.user_id)
        history.add(message)
        try:
            reply = self.messaging_agent.send(history)
        except LLMTimeoutError as exc:
            logger.warning("Timed out answering user '%s': %s", message.user_id, exc)
            reply = self._agent_message(message.user_id, TIMEOUT_REPLY)
        except Exception:
            logger.exception("Failed to answer user '%s'", message.user_id)
            reply = self._agent_message(message.user_id, ERROR_REPLY)
        history.add(reply)
        return reply

    @staticmethod
    def _ensure_content(reply: Message) -> Message:
        """Replace a reply with no text and no attachments by a fallback text."""
        if (reply.message or "").strip() or reply.image_paths or reply.file_paths:
            return reply
        return MessagingService._agent_message(reply.user_id, EMPTY_REPLY_FALLBACK)

    @staticmethod
    def _agent_message(user_id: str, text: str) -> Message:
        return Message(user_id=user_id, source=MessageSource.AGENT, message=text)

    # -- history -------------------------------------------------------

    def _get_history(self, user_id: str) -> MessageHistory:
        if user_id not in self._histories:
            self._histories[user_id] = MessageHistory(user_id=user_id)
        return self._histories[user_id]

    # -- commands ------------------------------------------------------

    def _handle_command(self, text: str) -> Optional[str]:
        """Handle platform-independent slash commands.

        Returns the reply text if `text` was a command, or None if it
        should be forwarded to the conversational agent.
        """
        parts = text.split(maxsplit=1)
        if not parts or parts[0] != CONNECT_COMMAND:
            return None
        if len(parts) < 2:
            return CONNECT_USAGE
        return self.messaging_agent.connect(parts[1].strip())


    # -- platform contract ---------------------------------------------

    @abstractmethod
    def _receive_loop(self) -> None:
        """Block, feeding inbound platform events to ``on_message_received``,
        until ``self.stopped`` is True."""

    @abstractmethod
    def send_message(self, message: Message) -> None:
        """Deliver a Message (text, images, files) to the chat platform."""