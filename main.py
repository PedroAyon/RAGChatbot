import logging
import os
from pathlib import Path
import signal

from dotenv import load_dotenv

from rag_chatbot.config import ComponentFactory


def configure_logging() -> None:
    debug = os.environ.get("DEBUG_MODE", "").strip().lower() == "true"

    logging.basicConfig(
        level=logging.INFO,  # Libraries stay at INFO+
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if debug:
        logging.getLogger("rag_chatbot").setLevel(logging.DEBUG)


def main() -> None:
    config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.yaml"))
    service = ComponentFactory.build_messaging_service(config_path)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # `docker stop` == Ctrl+C
    service.run_forever()


if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    main()