
import os
import streamlit as st

from datetime import datetime
from pathlib import Path

from jobs import ACTIVE, JobManager
from log import get_logger
from model import load_session
from process import BatchCancelled, Options, process_batch
from steps.background import load_background_image

log = get_logger("bodymorph.app")


# ------------------------------------------------
# Konfiguration
# ------------------------------------------------

OUTPUT_DIR = Path("/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="BodyMorph Preprocessor",
    page_icon="🧍",
    layout="wide"
)


# ------------------------------------------------
# Auftragsverwaltung
# ------------------------------------------------

@st.cache_resource
def job_manager():
    return JobManager(OUTPUT_DIR, load_session, process_batch, BatchCancelled)


manager = job_manager()


# ------------------------------------------------
# Oberfläche
# ------------------------------------------------

st.title("BodyMorph Preprocessor")

st.caption(
    "Lokale Ganzkörper-Bildaufbereitung "
    "für JavaMorph"
)

st.caption(
    f"Version: Build {os.environ.get('BUILD_NUMBER', 'dev')} "
    f"({os.environ.get('GIT_SHA', 'unknown')})"
)

with st.sidebar:

    st.header("Bildparameter")

    target_height = st.number_input(
        "Einheitliche Körperbildhöhe",
        min_value=256,
        max_value=4096,
        value=1200,
        step=64
    )

    margin = st.slider(
        "Randabstand (%)",
        min_value=0,
        max_value=20,
        value=5
    )

    background_label = st.radio(
        "Hintergrund",
        ["Farbe", "Bild", "Transparent", "Keine Entfernung"]
    )

    edge_erode = st.slider(
        "Kanten erodieren (px)",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

    edge_softness = st.slider(
        "Kantenweichheit",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

    background_color = "#D3D3D3"
    background_image = None

    if background_label == "Farbe":
        background_color = st.color_picker(
            "Hintergrundfarbe",
            "#D3D3D3"
        )

    elif background_label == "Bild":
        bg_upload = st.file_uploader(
            "Hintergrundbild",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if bg_upload:
            background_image = load_background_image(
                bg_upload.getvalue()
            )

    st.header("Farbnormalisierung")

    normalize_color = st.checkbox(
        "Körperfarbe angleichen",
        value=True
    )


# ------------------------------------------------
# Upload
# ------------------------------------------------

with st.form("upload_form", border=False):
    uploads = st.file_uploader(
        "Ganzkörperbilder auswählen",
        type=[
            "jpg", "jpeg", "png", "webp",
            "cr2", "cr3", "nef", "nrw", "arw", "dng",
            "orf", "rw2", "raf", "pef", "srw",
        ],
        accept_multiple_files=True
    )
    st.form_submit_button("Bilder übernehmen")

reference_index = None

if uploads and normalize_color:

    reference_name = st.selectbox(
        "Referenzbild für die Körperfarbe",
        ["Durchschnitt aller Bilder"] + [u.name for u in uploads]
    )

    if reference_name != "Durchschnitt aller Bilder":
        reference_index = [u.name for u in uploads].index(
            reference_name
        )


# ------------------------------------------------
# Verarbeitung
# ------------------------------------------------

if uploads:

    st.write(f"{len(uploads)} Bilder ausgewählt")

    process_clicked = st.button(
        "Bilder verarbeiten",
        type="primary",
        use_container_width=True
    )

    if process_clicked:

        if background_label == "Bild" and background_image is None:
            st.error("Bitte ein Hintergrundbild auswählen")
            st.stop()

        options = Options(
            target_height=int(target_height),
            margin_percent=margin,
            background_mode={
                "Farbe": "color",
                "Bild": "image",
                "Transparent": "transparent",
                "Keine Entfernung": "none",
            }[background_label],
            edge_softness=edge_softness,
            edge_erode=edge_erode,
            background_color=background_color,
            background_image=background_image,
            normalize_color=normalize_color,
            reference_index=reference_index,
        )

        try:
            job_id = manager.submit(uploads, options)
            st.success(f"Auftrag {job_id[:8]} wurde gestartet.")
        except (OSError, RuntimeError) as exc:
            log.exception("Auftrag konnte nicht gestartet werden")
            st.error(f"Auftrag konnte nicht gestartet werden: {exc}")


# ------------------------------------------------
# Auftragsverlauf
# ------------------------------------------------

@st.fragment(run_every=5)
def show_history():
    st.divider()
    st.header("Auftragsverlauf")
    st.caption(
        "Aufträge und Ergebnisse sind für alle Benutzer dieser Anwendung sichtbar "
        "und bleiben bis zum manuellen Löschen erhalten."
    )
    try:
        jobs = manager.list_jobs()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Auftragsverlauf konnte nicht geladen werden: {exc}")
        return
    if not jobs:
        st.info("Noch keine Aufträge vorhanden.")
        return

    labels = {
        "queued": "Wartet", "running": "In Bearbeitung",
        "completed": "Abgeschlossen", "cancelled": "Abgebrochen",
        "failed": "Fehlgeschlagen", "interrupted": "Unterbrochen",
    }
    job_by_id = {job["id"]: job for job in jobs}

    def job_label(job_id):
        job = job_by_id[job_id]
        created = datetime.fromisoformat(job["created"]).astimezone()
        return f"{created:%d.%m.%Y %H:%M:%S} · {job_id[:8]} · {labels.get(job['status'], job['status'])}"

    selected = st.selectbox("Auftrag auswählen", list(job_by_id), format_func=job_label)
    for job in [job_by_id[selected]]:
        job_id = job["id"]
        created = datetime.fromisoformat(job["created"]).astimezone()
        label = f"{created:%d.%m.%Y %H:%M:%S} · {job_id[:8]} · {labels.get(job['status'], job['status'])}"
        with st.expander(label, expanded=job["status"] in ACTIVE):
            st.write(job["message"])
            st.write(f"{len(job['results'])} von {job['total']} Bildern gespeichert")
            if job["status"] in ACTIVE:
                st.progress(job["progress"])
                if st.button("Auftrag abbrechen", key=f"cancel-{job_id}"):
                    try:
                        manager.cancel(job_id)
                        st.info("Abbruch angefordert; das aktuelle Bild wird noch beendet.")
                    except ValueError as exc:
                        st.warning(str(exc))
                continue

            for error in job["errors"]:
                st.error(error)
            directory = manager.root / job_id
            try:
                archive = directory / "bodymorph_output.zip"
                if archive.exists():
                    st.download_button(
                        "Alle Bilder als ZIP herunterladen", data=archive.read_bytes(),
                        file_name=f"bodymorph_{job_id[:8]}.zip", mime="application/zip",
                        key=f"zip-{job_id}", type="primary",
                    )
                for result in job["results"]:
                    filename = result["filename"]
                    metadata = result["metadata"]
                    with st.container(border=True):
                        st.write(f"**{filename}**")
                        st.image(str(directory / filename), width=300)
                        st.write(
                            f"Ausgabe: {metadata['output_size']} · "
                            f"Körpergröße: {metadata['scaled_size']} · "
                            f"Bounding Box: {metadata['bounding_box']} · "
                            f"Skalierung: {metadata['scale']:.3f}"
                        )
                        st.download_button(
                            "PNG herunterladen", data=(directory / filename).read_bytes(),
                            file_name=filename, mime="image/png",
                            key=f"png-{job_id}-{filename}",
                        )
            except OSError as exc:
                log.exception("Ergebnisse von Auftrag %s konnten nicht gelesen werden", job_id)
                st.error(f"Ergebnisse konnten nicht gelesen werden: {exc}")

            if st.checkbox("Löschen bestätigen", key=f"confirm-{job_id}"):
                if st.button("Auftrag und Ergebnisse löschen", key=f"delete-{job_id}"):
                    try:
                        manager.delete(job_id)
                    except (OSError, ValueError) as exc:
                        log.exception("Auftrag %s konnte nicht gelöscht werden", job_id)
                        st.error(f"Auftrag konnte nicht gelöscht werden: {exc}")
                    else:
                        st.rerun(scope="fragment")


show_history()
