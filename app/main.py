
import io
import zipfile
import streamlit as st

from pathlib import Path
from rembg import new_session

from processor import process_image


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
    return new_session("u2net")


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

    background = st.color_picker(
        "Hintergrund",
        "#D3D3D3"
    )


# ------------------------------------------------
# Upload
# ------------------------------------------------

uploads = st.file_uploader(
    "Ganzkörperbilder auswählen",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
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

        for index, upload in enumerate(uploads):

            status.write(f"Verarbeite {upload.name}")

            try:

                image_bytes = upload.getvalue()

                output, metadata = process_image(
                    image_bytes=image_bytes,
                    target_height=int(target_height),
                    margin_percent=margin,
                    background=background,
                    session=session
                )

                filename = (
                    f"{index + 1:03d}_"
                    f"{Path(upload.name).stem}.png"
                )

                output_path = OUTPUT_DIR / filename

                output.save(
                    output_path,
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

            except Exception as exc:

                errors.append(
                    f"{upload.name}: {exc}"
                )

            progress.progress(
                (index + 1) / len(uploads)
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
