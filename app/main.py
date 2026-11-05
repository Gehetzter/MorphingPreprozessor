
import io
import zipfile
import streamlit as st

from pathlib import Path
from rembg import new_session

from process import Options, process_batch
from steps.background import load_background_image


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
# Modell laden
# ------------------------------------------------

@st.cache_resource
def load_model():
    return new_session("isnet-general-use")


# ------------------------------------------------
# Oberfläche
# ------------------------------------------------

st.title("BodyMorph Preprocessor")

st.caption(
    "Lokale Ganzkörper-Bildaufbereitung "
    "für JavaMorph"
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

uploads = st.file_uploader(
    "Ganzkörperbilder auswählen",
    type=[
        "jpg", "jpeg", "png", "webp",
        "cr2", "cr3", "nef", "nrw", "arw", "dng",
        "orf", "rw2", "raf", "pef", "srw",
    ],
    accept_multiple_files=True
)

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

    if st.button(
        "Bilder verarbeiten",
        type="primary",
        use_container_width=True
    ):

        session = load_model()

        results = []
        errors = []

        progress = st.progress(0)
        status = st.empty()

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

        def on_progress(value, text):
            progress.progress(value)
            status.write(text)

        processed, errors = process_batch(
            [(u.name, u.getvalue()) for u in uploads],
            options,
            session,
            progress=on_progress,
        )

        for index, output, metadata in processed:

            filename = (
                f"{index + 1:03d}_"
                f"{Path(uploads[index].name).stem}.png"
            )

            output.save(
                OUTPUT_DIR / filename,
                format="PNG"
            )

            buffer = io.BytesIO()

            output.save(
                buffer,
                format="PNG"
            )

            results.append({
                "filename": filename,
                "image": output,
                "data": buffer.getvalue(),
                "metadata": metadata
            })

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
                    result["image"],
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
                    data=result["data"],
                    file_name=result["filename"],
                    mime="image/png",
                    key=result["filename"]
                )

    # ZIP-Export
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        for result in results:

            archive.writestr(
                result["filename"],
                result["data"]
            )

    st.download_button(
        "Alle Bilder als ZIP herunterladen",
        data=zip_buffer.getvalue(),
        file_name="bodymorph_output.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )


if errors:

    st.warning("Fehler bei einzelnen Bildern")

    for error in errors:
        st.error(error)
