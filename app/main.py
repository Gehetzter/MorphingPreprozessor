
import os
import time
import zipfile
import streamlit as st

from pathlib import Path

from log import get_logger
from model import MODEL_NAME, load_session
from process import Options, process_batch
from steps.background import load_background_image

log = get_logger("bodymorph.app")


# ------------------------------------------------
# Konfiguration
# ------------------------------------------------

OUTPUT_DIR = Path("/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUTPUT_DIR / "bodymorph_output.zip"

# Dateien im Ausgabeordner, die älter sind, werden beim Aufruf der Seite gelöscht
OUTPUT_MAX_AGE_HOURS = float(os.environ.get("OUTPUT_MAX_AGE_HOURS", "24"))


def clear_output(max_age_seconds=0):
    """Löscht Dateien im Ausgabeordner (max_age_seconds=0: alle)."""
    removed = 0
    now = time.time()

    for path in OUTPUT_DIR.iterdir():
        try:
            if path.is_file() and now - path.stat().st_mtime >= max_age_seconds:
                path.unlink()
                removed += 1
        except OSError:
            log.exception("Konnte %s nicht löschen", path)

    if removed:
        log.info("%d alte Ausgabedateien gelöscht", removed)


clear_output(OUTPUT_MAX_AGE_HOURS * 3600)

st.set_page_config(
    page_title="BodyMorph Preprocessor",
    page_icon="🧍",
    layout="wide"
)


# ------------------------------------------------
# Modell laden
# ------------------------------------------------

@st.cache_resource
def load_model():
    log.info("Lade Segmentierungsmodell %s", MODEL_NAME)
    session = load_session()
    log.info("Modell geladen")
    return session


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

# Im Formular löst nicht jede hochgeladene Datei einen Rerun aus;
# bei vielen Bildern verursachte das HTTP-400-Fehler beim Upload.
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


def on_cancel():
    log.warning("Verarbeitung vom Benutzer abgebrochen")
    st.session_state["cancelled"] = True
    st.session_state["results"] = []
    st.session_state["errors"] = []


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

    if st.session_state.pop("cancelled", False):
        st.warning("Verarbeitung abgebrochen")

    if process_clicked:

        if background_label == "Bild" and background_image is None:
            st.error("Bitte ein Hintergrundbild auswählen")
            st.stop()

        clear_output()
        st.session_state["results"] = []
        st.session_state["errors"] = []

        # Ein Klick löst einen Rerun aus (Callback läuft zuerst) und stoppt den
        # laufenden Durchlauf beim nächsten Streamlit-Aufruf, also nach dem
        # aktuellen Bild.
        st.button(
            "Abbrechen",
            key="cancel",
            on_click=on_cancel,
            use_container_width=True
        )

        session = load_model()

        results = []
        errors = []

        progress = st.progress(0)
        status = st.empty()

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

        def on_progress(value, text):
            progress.progress(value)
            status.write(text)

        def on_result(index, output, metadata):
            filename = (
                f"{index + 1:03d}_"
                f"{Path(uploads[index].name).stem}.png"
            )

            output.save(
                OUTPUT_DIR / filename,
                format="PNG"
            )

            results.append({
                "filename": filename,
                "metadata": metadata
            })

        _, errors = process_batch(
            [(u.name, u.getvalue) for u in uploads],
            options,
            session,
            progress=on_progress,
            on_result=on_result,
        )

        # ZIP auf der Platte (PNGs sind schon komprimiert -> nur ablegen)
        with zipfile.ZipFile(
            ZIP_PATH,
            "w",
            zipfile.ZIP_STORED
        ) as archive:

            for result in results:
                archive.write(
                    OUTPUT_DIR / result["filename"],
                    result["filename"]
                )

        st.session_state["results"] = results
        st.session_state["errors"] = errors

        status.success("Verarbeitung abgeschlossen")


# ------------------------------------------------
# Ergebnisse
# ------------------------------------------------

results = st.session_state.get("results", [])
errors = st.session_state.get("errors", [])

if results:

    st.divider()
    st.header("Verarbeitete Bilder")

    st.success(
        f"{len(results)} Bilder erfolgreich verarbeitet"
    )

    for result in results:

        with st.expander(result["filename"]):

            col1, col2 = st.columns([2, 1])

            with col1:

                st.image(
                    str(OUTPUT_DIR / result["filename"]),
                    use_container_width=True
                )

            with col2:

                metadata = result["metadata"]

                st.write("**Bildinformationen**")

                st.write(
                    f"Ausgabe: {metadata['output_size']}"
                )

                st.write(
                    f"Körpergröße: {metadata['scaled_size']}"
                )

                st.write(
                    f"Bounding Box: {metadata['bounding_box']}"
                )

                st.write(
                    f"Skalierung: {metadata['scale']:.3f}"
                )

                st.download_button(
                    "PNG herunterladen",
                    data=(OUTPUT_DIR / result["filename"]).read_bytes(),
                    file_name=result["filename"],
                    mime="image/png",
                    key=result["filename"]
                )

    # ZIP-Export (liegt fertig auf der Platte)
    if ZIP_PATH.exists():

        st.download_button(
            "Alle Bilder als ZIP herunterladen",
            data=ZIP_PATH.read_bytes(),
            file_name="bodymorph_output.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )


if errors:

    st.warning("Fehler bei einzelnen Bildern")

    for error in errors:
        st.error(error)
