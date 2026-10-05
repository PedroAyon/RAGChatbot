"""
TelegramMessagingService: MessagingService over the Telegram Bot API,
using long polling (``getUpdates``). Only Telegram specifics live here:
polling, parsing updates into Messages and delivering replies. Queueing,
commands, history and error handling are in MessagingService.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests

from ..models import Message, MessageSource
from .base import MessagingService

if TYPE_CHECKING:
    from ..agent.messaging_agent import MessagingAgent

logger = logging.getLogger(__name__)


class TelegramMessagingService(MessagingService):
    API_ROOT = "https://api.telegram.org"

    def __init__(
        self,
        messaging_agent: MessagingAgent,
        token: str,
        poll_interval: float = 1.0,
        poll_timeout: int = 30,
        max_queue_size: int = 50,
    ):
        if not token:
            raise ValueError("TelegramMessagingService requires a bot 'token'")
        super().__init__(messaging_agent, max_queue_size)
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._api_url = f"{self.API_ROOT}/bot{token}"
        self._offset: Optional[int] = None

    # -- inbound -------------------------------------------------------

    def _receive_loop(self) -> None:
        logger.info("TelegramMessagingService: starting long-poll loop")
        while not self.stopped:
            try:
                self._poll_once()
            except requests.RequestException as exc:
                logger.warning("Telegram polling error: %s", exc)
            except Exception:
                logger.exception("Unexpected error while handling Telegram updates")
            self._stop_event.wait(self.poll_interval)

    def _poll_once(self) -> None:
        params: Dict[str, Any] = {"timeout": self.poll_timeout}
        if self._offset is not None:
            params["offset"] = self._offset

        resp = requests.get(
            f"{self._api_url}/getUpdates",
            params=params,
            timeout=self.poll_timeout + 10,
        )
        resp.raise_for_status()

        for update in resp.json().get("result", []):
            self._offset = update["update_id"] + 1
            self._handle_update(update)

    def _handle_update(self, update: Dict[str, Any]) -> None:
        tg_message = update.get("message")
        if not tg_message or "text" not in tg_message:
            return  # ignore non-text updates (stickers, edits, etc.)

        self.on_message_received(
            Message(
                user_id=str(tg_message["chat"]["id"]),
                source=MessageSource.USER,
                message=tg_message["text"].strip(),
            )
        )

    # -- outbound ------------------------------------------------------

    def send_message(self, message: Message) -> None:
        chat_id = message.user_id

        for path in message.image_paths:
            self._send_file("sendPhoto", "photo", chat_id, path)

        text = (message.message or "").strip()
        if text:
            self._send_text(chat_id, text)

        for path in message.file_paths:
            self._send_file("sendDocument", "document", chat_id, path)

    def _send_text(self, chat_id: str, text: str) -> None:
        resp = requests.post(
            f"{self._api_url}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        resp.raise_for_status()

    def _send_file(self, endpoint: str, field: str, chat_id: str, path: str) -> None:
        with open(path, "rb") as fh:
            resp = requests.post(
                f"{self._api_url}/{endpoint}",
                data={"chat_id": chat_id},
                files={field: fh},
                timeout=30,
            )
        resp.raise_for_status()