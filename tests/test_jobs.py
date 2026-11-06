import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from jobs import JobManager


class Cancelled(Exception):
    def __init__(self, errors=None):
        self.errors = errors or []


class Upload(io.BytesIO):
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name


class Image:
    def save(self, path, format):
        path.write_bytes(b"png")


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager._executor.shutdown(wait=True)
        self.temp.cleanup()

    def manager(self, process, load=lambda: object()):
        manager = JobManager(self.root, load, process, Cancelled)
        self.managers.append(manager)
        return manager

    def test_disconnect_and_multiple_jobs_preserve_results_and_manual_delete(self):
        started = threading.Event()
        release = threading.Event()

        def process(files, options, session, progress, on_result, should_cancel):
            started.set()
            self.assertTrue(release.wait(5))
            self.assertEqual(files[0][1](), b"input")
            on_result(0, Image(), {"scale": 1})
            progress(1.0, "fertig")
            return 1, []

        manager = self.manager(process)
        legacy = self.root / "old.png"
        legacy.write_bytes(b"old")
        upload = Upload("../portrait.jpg", b"input")
        first = manager.submit([upload], None)
        self.assertTrue(started.wait(5))
        upload.close()  # Browser/session no longer owns the input
        second = manager.submit([Upload("portrait.jpg", b"input")], None)
        self.assertEqual({job["status"] for job in manager.list_jobs()}, {"queued", "running"})
        with self.assertRaises(ValueError):
            manager.delete(first)
        release.set()
        manager._executor.shutdown(wait=True)

        jobs = {job["id"]: job for job in manager.list_jobs()}
        self.assertEqual({job["status"] for job in jobs.values()}, {"completed"})
        for job_id in (first, second):
            directory = manager.root / job_id
            self.assertFalse((directory / "input").exists())
            self.assertEqual(jobs[job_id]["results"][0]["filename"], "001_portrait.png")
            with zipfile.ZipFile(directory / "bodymorph_output.zip") as archive:
                self.assertEqual(archive.read("001_portrait.png"), b"png")
        restored = self.manager(process)
        self.assertEqual(len(restored.list_jobs()), 2)
        manager.delete(first)
        self.assertEqual([job["id"] for job in manager.list_jobs()], [second])
        self.assertTrue(legacy.exists())

    def test_cancel_queued_job_and_retain_errors(self):
        started = threading.Event()
        release = threading.Event()

        def process(files, options, session, progress, on_result, should_cancel):
            started.set()
            self.assertTrue(release.wait(5))
            if should_cancel():
                raise Cancelled(["Bild fehlgeschlagen"])
            return 0, []

        manager = self.manager(process)
        first = manager.submit([Upload("one.jpg", b"1")], None)
        self.assertTrue(started.wait(5))
        second = manager.submit([Upload("two.jpg", b"2")], None)
        manager.cancel(first)
        manager.cancel(second)
        release.set()
        manager._executor.shutdown(wait=True)
        jobs = {job["id"]: job for job in manager.list_jobs()}
        self.assertEqual(jobs[first]["status"], "cancelled")
        self.assertEqual(jobs[first]["errors"], ["Bild fehlgeschlagen"])
        self.assertEqual(jobs[second]["status"], "cancelled")

    def test_partial_results_remain_downloadable_after_cancel(self):
        def process(files, options, session, progress, on_result, should_cancel):
            on_result(0, Image(), {"scale": 1})
            raise Cancelled(["zweites Bild fehlgeschlagen"])

        manager = self.manager(process)
        job_id = manager.submit([Upload("one.jpg", b"1")], None)
        manager._executor.shutdown(wait=True)
        job = manager.list_jobs()[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["errors"], ["zweites Bild fehlgeschlagen"])
        with zipfile.ZipFile(manager.root / job_id / "bodymorph_output.zip") as archive:
            self.assertEqual(archive.read("001_one.png"), b"png")

    def test_failed_model_and_restart_mark_interrupted(self):
        def fail():
            raise RuntimeError("Modellfehler")

        manager = self.manager(None, fail)
        job_id = manager.submit([Upload("x.jpg", b"x")], None)
        manager._executor.shutdown(wait=True)
        job = manager.list_jobs()[0]
        self.assertEqual(job["status"], "failed")
        self.assertIn("Modellfehler", job["message"])

        job["status"] = "running"
        (manager.root / job_id / "job.json").write_text(json.dumps(job), encoding="utf-8")
        restored = self.manager(None)
        self.assertEqual(restored.list_jobs()[0]["status"], "interrupted")
        restored.delete(job_id)
        self.assertEqual(restored.list_jobs(), [])

    def test_invalid_id_cannot_delete_outside_jobs(self):
        manager = self.manager(None)
        with self.assertRaises(ValueError):
            manager.delete("../other")


if __name__ == "__main__":
    unittest.main()
