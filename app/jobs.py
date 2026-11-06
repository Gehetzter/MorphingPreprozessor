"""Serverseitige Aufträge und dauerhaft gespeicherte Ergebnisse."""

import json
import re
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from log import get_logger

log = get_logger("bodymorph.jobs")

ACTIVE = {"queued", "running"}


def _safe_stem(name):
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(name).stem
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).rstrip(". ")[:120] or "bild"


class JobManager:
    def __init__(self, output_dir, load_session, process_batch, cancelled_exception):
        self.root = Path(output_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._load_session = load_session
        self._process_batch = process_batch
        self._cancelled_exception = cancelled_exception
        self._session = None
        self._lock = threading.RLock()
        self._cancel = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bodymorph")
        for job in self.list_jobs():
            if job["status"] in ACTIVE:
                job["status"] = "interrupted"
                job["message"] = "Server neu gestartet; Auftrag wurde unterbrochen."
                self._write(job)

    def _dir(self, job_id):
        # Auftrags-IDs werden ausschließlich vom Server erzeugt.
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Ungültige Auftrags-ID")
        return self.root / job_id

    def _write(self, job):
        path = self._dir(job["id"]) / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def list_jobs(self):
        with self._lock:
            jobs = []
            for path in self.root.glob("*/job.json"):
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                    if self._dir(job["id"]) != path.parent:
                        raise ValueError("Auftrags-ID stimmt nicht mit dem Ordner überein")
                    jobs.append(job)
                except (OSError, ValueError, KeyError, TypeError):
                    log.exception("Auftrag %s konnte nicht gelesen werden", path)
                    raise
            return sorted(jobs, key=lambda job: job["created"], reverse=True)

    def submit(self, uploads, options):
        job_id = uuid.uuid4().hex
        directory = self._dir(job_id)
        directory.mkdir()
        try:
            inputs = directory / "input"
            inputs.mkdir()
            files = []
            for index, upload in enumerate(uploads):
                path = inputs / f"{index:06d}"
                with path.open("wb") as target:
                    target.write(upload.getbuffer())
                files.append((upload.name, path))
            job = {
                "id": job_id,
                "created": datetime.now(timezone.utc).isoformat(),
                "status": "queued",
                "message": "Wartet auf Verarbeitung",
                "progress": 0.0,
                "total": len(files),
                "results": [],
                "errors": [],
            }
            self._write(job)
            cancel = threading.Event()
            with self._lock:
                self._cancel[job_id] = cancel
                try:
                    self._executor.submit(self._run, job, files, options, cancel)
                except Exception:
                    del self._cancel[job_id]
                    raise
        except Exception:
            shutil.rmtree(directory)
            raise
        return job_id

    def _run(self, job, files, options, cancel):
        directory = self._dir(job["id"])
        results = job["results"]
        errors = job["errors"]

        def update(status, message):
            with self._lock:
                job["status"] = status
                job["message"] = message
                self._write(job)

        def progress(value, text):
            with self._lock:
                job["progress"] = value
                job["message"] = text
                self._write(job)

        def on_result(index, image, metadata):
            filename = f"{index + 1:03d}_{_safe_stem(files[index][0])}.png"
            image.save(directory / filename, format="PNG")
            with self._lock:
                results.append({"filename": filename, "metadata": metadata})
                self._write(job)

        status = "completed"
        message = "Verarbeitung abgeschlossen"
        try:
            update("running", "Modell wird geladen")
            if cancel.is_set():
                raise self._cancelled_exception()
            if self._session is None:
                self._session = self._load_session()
            _, batch_errors = self._process_batch(
                [(name, path.read_bytes) for name, path in files],
                options, self._session, progress=progress, on_result=on_result,
                should_cancel=cancel.is_set,
            )
            errors.extend(batch_errors)
        except self._cancelled_exception as exc:
            errors.extend(exc.errors)
            status, message = "cancelled", "Verarbeitung abgebrochen"
        except Exception as exc:
            log.exception("Auftrag %s fehlgeschlagen", job["id"])
            status, message = "failed", f"Verarbeitung fehlgeschlagen: {exc}"

        try:
            if results:
                temporary_zip = directory / "bodymorph_output.zip.tmp"
                with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_STORED) as archive:
                    for result in results:
                        archive.write(directory / result["filename"], result["filename"])
                temporary_zip.replace(directory / "bodymorph_output.zip")
            shutil.rmtree(directory / "input")
        except Exception as exc:
            log.exception("Auftrag %s konnte nicht abgeschlossen werden", job["id"])
            status, message = "failed", f"Speichern oder Aufräumen fehlgeschlagen: {exc}"
        try:
            update(status, message)
        except OSError:
            log.exception("Status von Auftrag %s konnte nicht gespeichert werden", job["id"])
        finally:
            with self._lock:
                self._cancel.pop(job["id"], None)

    def cancel(self, job_id):
        with self._lock:
            event = self._cancel.get(job_id)
            if event is None:
                raise ValueError("Auftrag läuft nicht mehr")
            event.set()

    def delete(self, job_id):
        with self._lock:
            if job_id in self._cancel:
                raise ValueError("Laufende Aufträge können nicht gelöscht werden")
            shutil.rmtree(self._dir(job_id))
