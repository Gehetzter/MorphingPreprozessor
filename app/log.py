import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name="bodymorph"):
    """Logger, der nach stdout schreibt (sichtbar in `docker logs`)."""
    root = logging.getLogger("bodymorph")

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
        # Streamlit konfiguriert den Root-Logger selbst, Doppelausgaben vermeiden
        root.propagate = False

    return logging.getLogger(name)
