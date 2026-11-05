import os

import onnxruntime as ort
from rembg.sessions import sessions

MODEL_NAME = os.environ.get("SEGMENTATION_MODEL", "birefnet-general")


def load_session(name=MODEL_NAME):
    """Lädt das rembg-Modell mit speichersparenden ONNX-Einstellungen.

    - Kein CPU-Memory-Arena: ONNX behält sonst den Spitzenverbrauch dauerhaft
    - Kein Memory-Pattern: weniger Zusatzspeicher bei großen Eingaben
    - Wenige Threads: weniger parallele Zwischenpuffer, Host bleibt bedienbar
    """
    if name not in sessions:
        raise ValueError(f"Unbekanntes Modell: {name}")

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False

    threads = int(os.environ.get("ORT_THREADS", "2"))
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    return sessions[name](name, options)
