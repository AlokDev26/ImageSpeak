import time

import streamlit as st
import torch
from PIL import Image

import config as C
from model import EncoderCNN, CaptionDecoder
from vocabulary import Vocabulary


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CaptioCompare",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(0, 229, 255, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(72, 118, 255, 0.08),
                transparent 30%
            ),
            #081018;
        color: #f4f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- HEADER ---------- */

    .brand {
        text-align: center;
        margin-bottom: 0.4rem;
    }

    .brand-main {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #31f5e5;
        text-shadow:
            0 0 8px rgba(49,245,229,0.8),
            0 0 22px rgba(49,245,229,0.35);
    }

    .brand-sub {
        color: #d9e2ea;
        font-size: 1.05rem;
        margin-top: -5px;
    }

    /* ---------- CARDS ---------- */

    .panel {
        background:
            linear-gradient(
                145deg,
                rgba(24, 35, 47, 0.95),
                rgba(10, 18, 27, 0.95)
            );
        border: 1px solid rgba(150, 180, 200, 0.18);
        border-radius: 18px;
        padding: 22px;
        box-shadow:
            0 10px 35px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.025);
    }

    .model-card {
        border-radius: 18px;
        padding: 22px;
        min-height: 315px;
        background:
            linear-gradient(
                145deg,
                rgba(16, 30, 39, 0.97),
                rgba(7, 15, 23, 0.98)
            );
    }

    .custom-card {
        border: 1px solid rgba(65, 255, 154, 0.65);
        box-shadow:
            0 0 18px rgba(65,255,154,0.12),
            inset 0 0 25px rgba(65,255,154,0.025);
    }

    .blip-card {
        border: 1px solid rgba(64, 154, 255, 0.7);
        box-shadow:
            0 0 18px rgba(64,154,255,0.13),
            inset 0 0 25px rgba(64,154,255,0.025);
    }

    /* ---------- MODEL HEADERS ---------- */

    .custom-title {
        color: #52f59c;
        font-size: 1.25rem;
        font-weight: 750;
    }

    .blip-title {
        color: #55aaff;
        font-size: 1.25rem;
        font-weight: 750;
    }

    .model-subtitle {
        color: #8999a8;
        font-size: 0.82rem;
        margin-top: 2px;
    }

    /* ---------- CAPTION BOX ---------- */

    .caption-box {
        margin-top: 18px;
        min-height: 105px;
        padding: 17px;
        border-radius: 12px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.07);
        color: #f3f6fa;
        font-size: 1.02rem;
        line-height: 1.55;
    }

    .caption-label {
        color: #8ea0af;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* ---------- METRICS ---------- */

    .metric-row {
        display: flex;
        gap: 10px;
        margin-top: 18px;
    }

    .metric {
        flex: 1;
        background: rgba(255,255,255,0.035);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .metric-name {
        font-size: 0.67rem;
        color: #7f919f;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #f2f7fb;
        margin-top: 3px;
    }

    /* ---------- STATUS ---------- */

    .status {
        text-align: center;
        margin-top: 12px;
        color: #7d91a0;
        font-size: 0.85rem;
    }

    .status-online {
        color: #52f59c;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #687987;
        margin-top: 35px;
        font-size: 0.8rem;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        border-radius: 30px;
        min-height: 50px;
        border: 1px solid #31f5e5;
        background: rgba(0, 230, 220, 0.07);
        color: #45f5e5;
        font-weight: 800;
        font-size: 1rem;
        box-shadow:
            0 0 12px rgba(49,245,229,0.15);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #65fff2;
        background: rgba(49,245,229,0.14);
        box-shadow:
            0 0 24px rgba(49,245,229,0.3);
        color: white;
    }

    /* ---------- UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    /* ---------- SLIDER ---------- */

    div[data-baseweb="slider"] {
        margin-bottom: 5px;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 900px) {
        .brand-main {
            font-size: 2rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="brand">
        <div class="brand-main">CAPTIOCOMPARE</div>
        <div class="brand-sub">
            AI Image Caption Model Benchmarking
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status">
        <span class="status-online">●</span>
        Upload an image and compare your custom model with BLIP
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# =========================================================
# LOAD CUSTOM MODEL
# =========================================================

@st.cache_resource
def load_custom_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab = Vocabulary.load(C.VOCAB_FILE)

    checkpoint = torch.load(
        C.BEST_MODEL,
        map_location=device,
        weights_only=False,
    )

    cfg = checkpoint["config"]

    decoder = CaptionDecoder(
        cfg["vocab_size"],
        cfg["feature_dim"],
        cfg["embed_dim"],
        cfg["hidden_dim"],
        cfg["num_layers"],
        cfg["dropout"],
        vocab.pad_idx,
    ).to(device)

    decoder.load_state_dict(
        checkpoint["model_state"]
    )

    decoder.eval()

    encoder = EncoderCNN().to(device)
    encoder.eval()

    return (
        encoder,
        decoder,
        vocab,
        device,
        checkpoint,
    )


# =========================================================
# LOAD BLIP
# =========================================================

@st.cache_resource
def load_blip_model():

    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
    )

    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(
        model_name
    )

    model = BlipForConditionalGeneration.from_pretrained(
        model_name
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    return processor, model, device


# =========================================================
# CUSTOM CAPTION
# =========================================================

def generate_custom_caption(
    image,
    encoder,
    decoder,
    vocab,
    device,
):

    with torch.inference_mode():

        image_tensor = (
            encoder.transform(image)
            .unsqueeze(0)
            .to(device)
        )

        features = encoder(image_tensor)

        token_ids = decoder.generate(
            features,
            vocab.start_idx,
            vocab.end_idx,
            max_len=C.MAX_LEN + 1,
        )[0]

        caption = vocab.decode(token_ids)

    return caption.strip()


# =========================================================
# BLIP CAPTION
# =========================================================

def generate_blip_caption(
    image,
    processor,
    model,
    device,
    num_beams,
    temperature,
):

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=False,
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True,
    )

    return caption.strip()


# =========================================================
# LOAD MODELS
# =========================================================

try:

    with st.spinner("Loading AI models..."):

        (
            encoder,
            decoder,
            vocab,
            custom_device,
            checkpoint,
        ) = load_custom_model()

        processor, blip_model, blip_device = (
            load_blip_model()
        )

except Exception as error:

    st.error(
        "Could not load the AI models."
    )

    st.exception(error)

    st.stop()


# =========================================================
# TOP CONTROL AREA
# =========================================================

left, right = st.columns(
    [1.15, 1],
    gap="large",
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

with left:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### 📤 Upload Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        label_visibility="collapsed",
    )

    image = None

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True,
        )

    else:

        st.info(
            "Upload JPG, JPEG, PNG or WEBP image."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# GENERATION SETTINGS
# =========================================================

with right:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### ⚙️ Generation Settings"
    )

    st.markdown(
        "#### BLIP Beam Search"
    )

    num_beams = st.slider(
        "Beam Search",
        min_value=1,
        max_value=8,
        value=5,
        step=1,
    )

    st.markdown(
        "#### Length / Temperature"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )

    st.caption(
        "Beam search and temperature currently control BLIP generation. "
        "Your custom LSTM uses its existing decoder."
    )

    st.write("")

    run_button = st.button(
        "✨ RUN COMPARISON",
        type="primary",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# RUN COMPARISON
# =========================================================

if run_button:

    if image is None:

        st.warning(
            "Please upload an image first."
        )

    else:

        custom_caption = ""
        blip_caption = ""

        custom_time = 0
        blip_time = 0

        # ---------------------------------------------
        # CUSTOM MODEL
        # ---------------------------------------------

        with st.spinner(
            "Running Custom ResNet50 + LSTM..."
        ):

            start = time.perf_counter()

            try:

                custom_caption = generate_custom_caption(
                    image,
                    encoder,
                    decoder,
                    vocab,
                    custom_device,
                )

            except Exception as error:

                custom_caption = (
                    "Custom model failed."
                )

                st.error(
                    f"Custom model error: {error}"
                )

            custom_time = (
                time.perf_counter()
                - start
            )

        # ---------------------------------------------
        # BLIP
        # ---------------------------------------------

        with st.spinner(
            "Running pretrained BLIP..."
        ):

            start = time.perf_counter()

            try:

                blip_caption = generate_blip_caption(
                    image,
                    processor,
                    blip_model,
                    blip_device,
                    num_beams,
                    temperature,
                )

            except Exception as error:

                blip_caption = (
                    "BLIP model failed."
                )

                st.error(
                    f"BLIP error: {error}"
                )

            blip_time = (
                time.perf_counter()
                - start
            )

        # =================================================
        # RESULTS
        # =================================================

        st.write("")
        st.markdown(
            "## 🧠 Caption Comparison"
        )

        custom_col, blip_col = st.columns(
            2,
            gap="large",
        )

        # =================================================
        # CUSTOM CARD
        # =================================================

        with custom_col:

            st.markdown(
                f"""
                <div class="model-card custom-card">

                    <div class="custom-title">
                        🟢 CUSTOM MODEL
                    </div>

                    <div class="model-subtitle">
                        ResNet50 Encoder + LSTM Decoder
                    </div>

                    <div class="caption-box">

                        <div class="caption-label">
                            Generated Caption
                        </div>

                        {custom_caption}

                    </div>

                    <div class="metric-row">

                        <div class="metric">
                            <div class="metric-name">
                                Inference
                            </div>
                            <div class="metric-value">
                                {custom_time:.2f}s
                            </div>
                        </div>

                        <div class="metric">
                            <div class="metric-name">
                                Device
                            </div>
                            <div class="metric-value">
                                {custom_device.type.upper()}
                            </div>
                        </div>

                        <div class="metric">
                            <div class="metric-name">
                                Epoch
                            </div>
                            <div class="metric-value">
                                {checkpoint["epoch"]}
                            </div>
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # =================================================
        # BLIP CARD
        # =================================================

        with blip_col:

            st.markdown(
                f"""
                <div class="model-card blip-card">

                    <div class="blip-title">
                        🔵 BLIP MODEL
                    </div>

                    <div class="model-subtitle">
                        Salesforce BLIP Image Captioning
                    </div>

                    <div class="caption-box">

                        <div class="caption-label">
                            Generated Caption
                        </div>

                        {blip_caption}

                    </div>

                    <div class="metric-row">

                        <div class="metric">
                            <div class="metric-name">
                                Inference
                            </div>
                            <div class="metric-value">
                                {blip_time:.2f}s
                            </div>
                        </div>

                        <div class="metric">
                            <div class="metric-name">
                                Device
                            </div>
                            <div class="metric-value">
                                {blip_device.type.upper()}
                            </div>
                        </div>

                        <div class="metric">
                            <div class="metric-name">
                                Beams
                            </div>
                            <div class="metric-value">
                                {num_beams}
                            </div>
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.write("")

with st.expander(
    "ℹ️ Project & Model Information"
):

    info1, info2, info3 = st.columns(3)

    with info1:

        st.markdown(
            """
            **Custom Model**

            - ResNet50 Encoder
            - LSTM Decoder
            - Flickr8k trained
            - Best checkpoint: Epoch 27
            """
        )

    with info2:

        st.markdown(
            """
            **BLIP Model**

            - Salesforce BLIP
            - Pretrained vision-language model
            - Local inference
            - No image search
            """
        )

    with info3:

        st.markdown(
            f"""
            **System**

            - Custom device: `{custom_device.type.upper()}`
            - BLIP device: `{blip_device.type.upper()}`
            - Vocabulary: `{len(vocab):,}`
            - Validation loss: `{checkpoint["val_loss"]:.4f}`
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        CAPTIOCOMPARE • AI Image Caption Model Benchmarking
        <br>
        Custom ResNet50 + LSTM • Pretrained BLIP
    </div>
    """,
    unsafe_allow_html=True,
)