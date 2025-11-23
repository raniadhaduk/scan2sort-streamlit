import io
import json
import os
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
import google.generativeai as genai

# ============================================
# 0. Configure Gemini (using Streamlit secrets)
# ============================================

# In Streamlit Cloud, you will set this in:
# Settings → Secrets → add GOOGLE_API_KEY
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not set in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"
model = genai.GenerativeModel(MODEL_NAME)

# ============================================
# 1. Bin definitions (for UI)
# ============================================

BIN_ASSETS = {
    "green": {
        "bin_type": "Green Bin (Organics)",
        "color": "#2ecc71",
        "label": "Green Bin",
    },
    "blue": {
        "bin_type": "Recycling (Blue Bin)",
        "color": "#3498db",
        "label": "Blue Bin (Recycling)",
    },
    "black": {
        "bin_type": "Garbage",
        "color": "#2c3e50",
        "label": "Garbage",
    },
    "hazard": {
        "bin_type": "Hazardous Waste",
        "color": "#e74c3c",
        "label": "Hazardous Waste",
    },
    "unknown": {
        "bin_type": "Unknown",
        "color": "#7f8c8d",
        "label": "Unknown – please check local rules",
    },
}

# ============================================
# 2. Simple waste classifier (from description)
# ============================================

def classify_waste_from_description(desc: str) -> List[Dict[str, Any]]:
    """
    Simple classifier based on the text description from Gemini.
    You can later replace this with your full multi-agent logic if you want.
    """
    d = desc.lower()

    # Multi-part example: paper cup + plastic lid
    if "paper cup" in d and "plastic lid" in d:
        return [
            {
                "bin_type": BIN_ASSETS["green"]["bin_type"],
                "color": BIN_ASSETS["green"]["color"],
                "label": BIN_ASSETS["green"]["label"],
                "items": ["paper cup"],
                "explanation": "Paper coffee cups go in the green bin in Ottawa.",
            },
            {
                "bin_type": BIN_ASSETS["blue"]["bin_type"],
                "color": BIN_ASSETS["blue"]["color"],
                "label": BIN_ASSETS["blue"]["label"],
                "items": ["plastic lid"],
                "explanation": "Clean plastic drink lids go in the blue recycling bin.",
            },
        ]

    # Food / organics → green bin
    if any(word in d for word in ["banana peel", "apple core", "food scraps", "coffee grounds"]):
        return [
            {
                "bin_type": BIN_ASSETS["green"]["bin_type"],
                "color": BIN_ASSETS["green"]["color"],
                "label": BIN_ASSETS["green"]["label"],
                "items": [desc],
                "explanation": "Food scraps and organics go in the green bin.",
            }
        ]

    # Containers → blue bin
    if any(word in d for word in ["plastic bottle", "water bottle", "can", "glass jar", "plastic container"]):
        return [
            {
                "bin_type": BIN_ASSETS["blue"]["bin_type"],
                "color": BIN_ASSETS["blue"]["color"],
                "label": BIN_ASSETS["blue"]["label"],
                "items": [desc],
                "explanation": "Clean containers go in the blue recycling bin.",
            }
        ]

    # Hazardous
    if any(word in d for word in ["battery", "propane", "chemical", "paint can"]):
        return [
            {
                "bin_type": BIN_ASSETS["hazard"]["bin_type"],
                "color": BIN_ASSETS["hazard"]["color"],
                "label": BIN_ASSETS["hazard"]["label"],
                "items": [desc],
                "explanation": "This looks hazardous. Please bring it to a hazardous waste drop-off.",
            }
        ]

    # Default to garbage
    return [
        {
            "bin_type": BIN_ASSETS["black"]["bin_type"],
            "color": BIN_ASSETS["black"]["color"],
            "label": BIN_ASSETS["black"]["label"],
            "items": [desc],
            "explanation": "This item most likely belongs in the garbage bin.",
        }
    ]

# ============================================
# 3. Gemini triage: waste vs not-waste vs blurry
# ============================================

VISION_CLASSIFIER_PROMPT = """
You are a strict image triage assistant for a waste-sorting app.

Your job:
1. Decide if this image is a CLEAR picture of ONE OR A FEW WASTE OBJECTS (like cup, bottle, can, food scraps, packaging).
2. Detect if the image is too blurry, dark, or unclear to identify any object.
3. Detect if the image mainly shows a PERSON, PEOPLE, ANIMALS, or general scenery instead of waste.

Respond in EXACT JSON with keys:
- "category": one of ["waste", "not_waste", "blurry"]
- "description": a short description of the main object(s) if "waste", otherwise a short reason.

Do NOT include any other text, just JSON.
"""

def gemini_classify_image(pil_image: Image.Image) -> Dict[str, Any]:
    resp = model.generate_content(
        [
            VISION_CLASSIFIER_PROMPT,
            pil_image,
        ]
    )

    text = resp.text.strip()

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Not a JSON object")
        return data
    except Exception:
        # If parsing fails, treat as blurry / error
        return {"category": "blurry", "description": "Could not confidently parse response."}

# ============================================
# 4. Streamlit UI
# ============================================

st.set_page_config(page_title="Scan2Sort – Waste Sorting Assistant", page_icon="♻️")

st.title("Scan2Sort – Waste Sorting Assistant")
st.write(
    "Upload a photo of a **waste item** (cup, bottle, food scraps, packaging). "
    "I'll tell you which bin it should go in and why."
)

uploaded_file = st.file_uploader("Upload an image of your waste item", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
    except Exception:
        st.error("Could not open this image file. Please try with a different image.")
        st.stop()

    if st.button("Classify"):
        with st.spinner("Analyzing image..."):
            triage = gemini_classify_image(image)
            category = triage.get("category", "blurry")
            desc = triage.get("description", "").strip()

        # Handle blurry / unclear images
        if category == "blurry":
            st.error("The image is unclear. Please upload a clearer photo of the waste item.")
            st.info("Tip: Make sure the item is in focus and fills most of the frame.")
        # Handle wrong content (humans, pets, scenery)
        elif category == "not_waste":
            st.error("I can only analyze waste items.")
            st.info("Please upload a photo of a waste item like a cup, bottle, packaging, or food scraps.")
        elif category == "waste":
            st.success("Looks good! This appears to be a waste item.")
            st.caption(f"Detected description: _{desc}_")

            # Classify into bin(s)
            bins = classify_waste_from_description(desc)

            # Simple scoring: 10 points per bin
            points = 10 * len(bins)

            st.subheader("Result")

            for b in bins:
                # Nice colored bin card
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 12px;
                        padding: 12px 16px;
                        margin-bottom: 8px;
                        background: #f9fafb;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <div style="
                            width: 32px;
                            height: 32px;
                            border-radius: 8px;
                            background: {b['color']};
                            border: 2px solid rgba(0,0,0,0.05);
                        "></div>
                        <div>
                            <div style="font-weight: 600; font-size: 14px;">
                                {b['label']}
                            </div>
                            <div style="font-size: 13px; color: #4b5563;">
                                Items: {", ".join(b['items'])}
                            </div>
                            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
                                {b['explanation']}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div style='text-align: right; font-weight: 600; margin-top: 12px;'>"
                f"Points this scan: {points}</div>",
                unsafe_allow_html=True,
            )
        else:
            # Unexpected status
            st.error("Something went wrong while analyzing the image. Please try again.")
else:
    st.info("Upload a clear photo of a waste item to get started.")
