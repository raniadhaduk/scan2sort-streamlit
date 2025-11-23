import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import streamlit as st
from PIL import Image
import google.generativeai as genai

# ============================================
# 0. Setup & Dependencies (Streamlit + Gemini)
# ============================================

# In Streamlit Cloud, set this in:
# Settings → Secrets → add GOOGLE_API_KEY = "your_key_here"
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not set in Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"
model = genai.GenerativeModel(MODEL_NAME)


# ============================================
# 1. Simple logging helper (observability)
# ============================================

def log_event(event: str, **kwargs):
    # Developer logs (shows in server logs, not UI)
    print(f"[LOG] {event} | " + ", ".join(f"{k}={v}" for k, v in kwargs.items()))


# ============================================
# 2. Data Structures (Agents I/O)
# ============================================

@dataclass
class VisionResult:
    description: str
    confidence: float = 0.7
    notes: str = ""


@dataclass
class PolicyDecision:
    item: str
    bin_type: str
    explanation: str
    tips: str
    source: str  # "material_rules" | "local_db" | "backup_ai"


# ============================================
# 3. Waste Rules DB (Ottawa)
# ============================================

RAW_RULE_CATEGORIES = [
    # --- GREEN BIN (ORGANICS) ---
    (
        "Green Bin (Organics)",
        "Food scraps and compostable kitchen waste go in the green bin.",
        [
            "apple core", "banana peel", "orange peel", "lemon peel", "lime peel",
            "grape stems", "mango peel", "melon rind", "fruit scraps",
            "vegetable peel", "potato peel", "carrot peel", "onion skins",
            "garlic skins", "broccoli stalk", "lettuce core", "cabbage leaves",
            "coffee grounds", "coffee filter", "tea bag", "loose tea leaves",
            "bread crust", "stale bread", "pasta leftovers", "rice leftovers",
            "leftover food", "plate scrapings", "pizza crust",
            "egg shell", "eggshells",
            "chicken bone", "meat bone", "fish bone",
            "paper towel", "paper napkin", "tissue (used)",
            "greasy pizza box", "soiled paper plate", "compostable paper cup",
            "small plant trimmings", "wilted flowers", "dead flowers",
            "pumpkin guts", "pumpkin seeds (cooked)", "corn cob", "corn husk",
            # you can also add "pumpkin", "whole pumpkin" to local rules if you like
        ],
    ),

    # --- GARBAGE ---
    (
        "Garbage",
        "Garbage items go in the regular garbage bin, not in the green bin or recycling.",
        [
            "diaper", "baby diaper", "adult diaper",
            "menstrual pad", "tampon", "panty liner",
            "plastic cutlery", "plastic fork", "plastic spoon", "plastic knife",
            "plastic straw", "drinking straw",
            "chip bag", "chips bag", "crisp packet",
            "candy wrapper", "chocolate wrapper", "snack wrapper",
            "styrofoam", "foam cup", "foam takeout container", "foam tray",
            "toothbrush", "toothpaste tube", "floss", "dental floss",
            "cotton swab", "q-tip", "cotton ball", "makeup wipe",
            "rubber band", "balloon", "latex balloon", "latex glove",
            "broken ceramic mug", "broken mug", "broken plate", "ceramic plate",
            "mirror shard", "small broken mirror",
            "plastic wrap", "plastic film", "cling film", "saran wrap",
            "plastic bag", "grocery bag", "shopping bag",
            "vacuum bag", "vacuum dust", "swept dust",
            "dryer lint", "lint from dryer filter",
            "cigarette butt", "ashtray contents (cold)",
            "disposable razor", "disposable face mask", "nitrile glove",
            "pet waste bag", "dog poop bag", "cat litter clumps (in bag)",
            "old sponge", "kitchen sponge", "scrub sponge",
            "broken toy", "small plastic toy", "rubber toy",
            "pen", "marker", "mechanical pencil",
            "hair tie", "elastic hair band", "broken hair clip",
            "old makeup brush", "mascara tube", "lipstick tube",
        ],
    ),

    # --- BLUE BIN – PAPER / CARDBOARD ---
    (
        "Recycling (Blue Bin)",
        "Clean paper and cardboard go in the blue recycling bin.",
        [
            "newspaper", "magazine", "flyer", "brochure",
            "office paper", "printer paper", "notebook paper",
            "paper envelope", "window envelope",
            "paper bag", "shopping paper bag", "brown paper bag",
            "cardboard box", "shipping box", "corrugated box",
            "cereal box", "pasta box", "snack box", "shoe box",
            "paper egg carton", "cardboard egg carton",
            "toilet paper roll", "paper towel roll", "cardboard tube",
            "paper insert", "cardboard sleeve", "paper packaging",
            "paper file folder", "manila folder",
        ],
    ),

    # --- BLUE BIN – CONTAINERS ---
    (
        "Recycling (Blue Bin)",
        "Clean, empty containers made of glass, metal, or accepted plastics go in the blue bin.",
        [
            "plastic water bottle", "water bottle", "pop bottle", "soda bottle",
            "juice bottle", "sports drink bottle",
            "milk jug", "juice jug", "detergent jug",
            "yogurt tub", "yogurt container", "margarine tub",
            "plastic clamshell", "berry container", "salad container",
            "tin can", "metal can", "soup can", "bean can",
            "aluminum can", "pop can", "beer can",
            "glass jar", "jam jar", "pasta sauce jar",
            "glass bottle", "olive oil bottle", "vinegar bottle",
            "metal jar lid", "metal lid", "metal bottle cap",
            "aluminum tray", "foil tray", "aluminum pie plate",
            "clean aluminum foil", "clean tinfoil",
        ],
    ),

    # --- REFUND / RETURN ---
    (
        "Refund/Return",
        "Many alcoholic beverage containers can be returned for a refund where programs exist.",
        [
            "beer bottle", "beer can", "wine bottle", "spirit bottle",
            "cooler bottle", "cooler can", "lcbo bottle",
        ],
    ),

    # --- HAZARDOUS WASTE ---
    (
        "Hazardous Waste",
        "Hazardous waste must go to a depot or collection program, not in any household bin.",
        [
            "aa battery", "aaa battery", "lithium battery", "button cell battery",
            "rechargeable battery", "power tool battery",
            "syringe", "needle", "epipen", "insulin pen",
            "paint can with paint", "leftover paint",
            "nail polish", "nail polish remover", "acetone",
            "bleach", "drain cleaner", "oven cleaner", "strong cleaner",
            "motor oil", "used motor oil", "antifreeze",
            "pesticide", "herbicide", "weed killer", "bug spray (full)",
            "propane tank", "propane cylinder", "camping fuel canister",
            "pool chemical", "chlorine tablets",
            "gasoline container with fuel", "fuel can with fuel",
            "aerosol can (full)", "spray paint can (full)",
            "medicine", "pill", "tablet", "liquid medicine", "syrup medicine",
        ],
    ),

    # --- ELECTRONIC WASTE ---
    (
        "Electronic Waste",
        "Electronics go to e-waste collection, not household garbage or recycling bins.",
        [
            "old tv", "television", "monitor",
            "laptop", "notebook computer", "desktop computer",
            "tablet", "ipad", "android tablet",
            "smartphone", "cellphone", "mobile phone",
            "printer", "scanner", "fax machine",
            "keyboard", "computer mouse",
            "router", "modem", "wifi router",
            "game console", "xbox", "playstation", "nintendo switch",
            "dvd player", "blu-ray player", "stereo receiver",
            "cable box", "satellite receiver",
            "digital camera", "camcorder",
        ],
    ),

    # --- BULKY / SPECIAL COLLECTION ---
    (
        "Special Collection / Call 3-1-1",
        "Large or bulky items usually require special collection – contact the city or property manager.",
        [
            "sofa", "couch", "loveseat", "armchair",
            "mattress", "box spring", "bed frame",
            "wardrobe", "dresser", "bookshelf",
            "desk", "office chair",
            "fridge", "refrigerator", "freezer",
            "stove", "oven", "range", "cooktop",
            "dishwasher", "washing machine", "dryer",
            "water heater", "hot water tank",
            "air conditioner", "window ac unit",
            "large rug", "carpet roll",
            "door", "interior door", "cabinet door",
            "bathtub (old)", "toilet (old)", "sink (old)",
        ],
    ),

    # --- VARIES (EMPTY/RINSED RULE) ---
    (
        "Varies by material",
        "All items must be empty/rinsed before being placed in recycling or organics.",
        [
            "empty container", "rinsed container", "clean jar", "clean can",
        ],
    ),
]

WASTE_RULES_DB_OTTAWA: List[Dict[str, Any]] = []
for bin_type, notes, items in RAW_RULE_CATEGORIES:
    WASTE_RULES_DB_OTTAWA.append({
        "keywords": items,
        "bin": bin_type,
        "notes": notes,
    })


# ============================================
# 4. Tools (MCP-style)
# ============================================

class LocalWasteRulesTool:
    """
    MCP Tool #1:
      - Looks up Ottawa waste rules from local DB using simple keyword matching.
    """

    def __init__(self, rules_db: List[Dict[str, Any]]):
        self.rules_db = rules_db

    def classify(self, description: str) -> Optional[Dict[str, Any]]:
        text = description.lower()
        best_match = None
        best_score = 0
        for rule in self.rules_db:
            score = sum(1 for kw in rule["keywords"] if kw in text)
            if score > best_score:
                best_score = score
                best_match = rule
        if best_score == 0:
            return None
        return {
            "bin": best_match["bin"],
            "notes": best_match["notes"],
            "source": "local_db",
        }


class BackupWasteAIAssistantTool:
    """
    MCP Tool #2 (backup):
      - Asks Gemini directly for bin + explanation in JSON.
    """

    ALLOWED_BINS = [
        "Garbage",
        "Recycling (Blue Bin)",
        "Green Bin (Organics)",
        "Hazardous Waste",
        "Special Collection / Call 3-1-1",
        "Refund/Return",
        "Varies by material",
        "Unknown",
    ]

    def classify(self, description: str) -> Dict[str, Any]:
        prompt = f"""
You are a waste-sorting assistant for the City of Ottawa, Canada.

Determine where this item should go and respond ONLY as compact JSON.

Item: "{description}"

Return EXACTLY this JSON shape and nothing else:

{{
  "bin": "<one of: Garbage, Recycling (Blue Bin), Green Bin (Organics), Hazardous Waste, Special Collection / Call 3-1-1, Refund/Return, Varies by material, Unknown>",
  "explanation": "<short 2–3 sentence explanation for a resident>"
}}
"""
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        log_event("BackupTool.raw", text=raw)

        # Strip code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        bin_value = "Unknown"
        explanation = raw

        # Try to parse JSON
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                b = data.get("bin", "").strip()
                e = data.get("explanation", "").strip() or explanation
                if b in self.ALLOWED_BINS:
                    bin_value = b
                    explanation = e
        except Exception as e:
            log_event("BackupTool.json_parse_error", error=str(e))

        # Heuristic fallback if bin still Unknown: infer from explanation text
        if bin_value == "Unknown":
            lower = explanation.lower()
            if "green bin" in lower or "organics" in lower or "compost" in lower:
                bin_value = "Green Bin (Organics)"
            elif "recycl" in lower or "blue bin" in lower:
                bin_value = "Recycling (Blue Bin)"
            elif "hazard" in lower or "depot" in lower:
                bin_value = "Hazardous Waste"
            elif "special collection" in lower or "large item" in lower:
                bin_value = "Special Collection / Call 3-1-1"

        return {
            "bin": bin_value,
            "notes": explanation,
            "source": "backup_ai",
        }


LOCAL_WASTE_TOOL = LocalWasteRulesTool(WASTE_RULES_DB_OTTAWA)
BACKUP_AI_TOOL = BackupWasteAIAssistantTool()


# ============================================
# 5. Extra Material Rules (for multi-part items)
# ============================================

MATERIAL_RULES: Dict[str, Dict[str, str]] = {
    "paper cup": {
        "bin": "Green Bin (Organics)",
        "notes": "Paper coffee cups go in the green bin in Ottawa."
    },
    "coffee cup": {
        "bin": "Green Bin (Organics)",
        "notes": "Paper coffee cups go in the green bin in Ottawa."
    },
    "plastic lid": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean plastic drink lids go in the blue bin."
    },
    "yogurt container": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean plastic yogurt containers go in the blue bin."
    },
    "foil lid": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean foil lids go in the blue bin (scrunched into a ball if small)."
    },
    "glass jar": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean glass jars go in the blue bin."
    },
    "metal lid": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Metal lids go in the blue bin."
    },
    "takeout bowl": {
        "bin": "Green Bin (Organics)",
        "notes": "Many fiber/cardboard takeout bowls go in the green bin if food-soiled."
    },
    "takeout container": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean plastic takeout containers go in the blue bin."
    },
}


# ============================================
# 6. Multi-material handling helpers
# ============================================

def split_multimaterial_item(description: str) -> List[str]:
    """
    Detect known multi-material patterns like:
      - 'paper cup with plastic lid'
      - 'yogurt container with foil lid'
      - 'glass jar with metal lid'
    """
    desc = description.lower()

    patterns = [
        ("paper cup", "plastic lid"),
        ("coffee cup", "plastic lid"),
        ("cardboard cup", "plastic lid"),
        ("yogurt container", "foil lid"),
        ("glass jar", "metal lid"),
        ("takeout bowl", "plastic lid"),
        ("takeout container", "plastic lid"),
    ]

    for a, b in patterns:
        if a in desc and b in desc:
            return [a, b]

    return [desc]  # fallback: single-component


def classify_single_component(component_desc: str) -> PolicyDecision:
    """
    Policy-time tool orchestration:
      1) Check MATERIAL_RULES (hard-coded, high-confidence)
      2) Try LocalWasteRulesTool (MCP tool #1)
      3) If still unknown, call BackupWasteAIAssistantTool (MCP tool #2)
    """
    comp = component_desc.lower().strip()

    # 1️⃣ Material rules
    if comp in MATERIAL_RULES:
        r = MATERIAL_RULES[comp]
        return PolicyDecision(
            item=component_desc,
            bin_type=r["bin"],
            explanation=r["notes"],
            tips=r["notes"],
            source="material_rules",
        )

    # 2️⃣ Local tool
    local_res = LOCAL_WASTE_TOOL.classify(component_desc)
    if local_res is not None:
        return PolicyDecision(
            item=component_desc,
            bin_type=local_res["bin"],
            explanation=local_res["notes"],
            tips=local_res["notes"],
            source=local_res["source"],
        )

    log_event("LocalTool.miss", item=component_desc)

    # 3️⃣ Backup MCP tool (Gemini JSON)
    backup_res = BACKUP_AI_TOOL.classify(component_desc)

    return PolicyDecision(
        item=component_desc,
        bin_type=backup_res["bin"],
        explanation=backup_res["notes"],
        tips="This suggestion is based on AI backup reasoning. Check Ottawa's tools if unsure.",
        source=backup_res["source"],
    )


def classify_item(description: str) -> List[PolicyDecision]:
    components = split_multimaterial_item(description)
    return [classify_single_component(c) for c in components]


# ============================================
# 7. Vision Agent (describe waste item from image)
# ============================================

VISION_SYSTEM_PROMPT = (
    "You are an assistant that identifies household waste items from images. "
    "Describe the item in ONE short sentence, including materials if possible, "
    "for example: 'paper cup with plastic lid', 'plastic yogurt container with foil lid'. "
    "Do NOT give disposal instructions. Only describe what you see."
)

def vision_agent_pil(img: Image.Image) -> VisionResult:
    """
    Agent 1: Vision Agent (using PIL image directly)
    """
    resp = model.generate_content(
        [
            VISION_SYSTEM_PROMPT,
            img,
        ]
    )
    desc = resp.text.strip()
    log_event("VisionAgent.output", description=desc)

    return VisionResult(description=desc, confidence=0.7, notes="")


# ============================================
# 8. Triage: waste vs not-waste vs blurry
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

def triage_image(img: Image.Image) -> Dict[str, Any]:
    """
    Use Gemini to decide:
      - waste
      - not_waste (e.g., person, animal, scenery)
      - blurry

    Robust:
      - Handles JSON inside ``` code fences
      - If JSON parse fails, use keyword heuristics
    """
    resp = model.generate_content(
        [
            VISION_CLASSIFIER_PROMPT,
            img,
        ]
    )

    text = resp.text.strip()
    log_event("Triage.raw_response", text=text)

    # 1) Strip code fences if Gemini wrapped JSON in ```...```
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2) Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "category" in data:
            return data
    except Exception as e:
        log_event("Triage.json_parse_error", error=str(e))

    # 3) Fallback: heuristic classification based on plain text
    lower = text.lower()

    if any(k in lower for k in ["blurry", "blurred", "out of focus", "too dark", "unclear"]):
        return {"category": "blurry", "description": text}

    if any(k in lower for k in ["person", "people", "face", "selfie", "human", "dog", "cat", "animal"]):
        return {"category": "not_waste", "description": text}

    return {"category": "waste", "description": text}


# ============================================
# 9. Policy Agent
# ============================================

def policy_agent(vision: VisionResult) -> List[PolicyDecision]:
    """
    Agent 2: Policy Agent
    - Calls the MCP tools orchestrated via classify_item()
    """
    decisions = classify_item(vision.description)
    log_event(
        "PolicyAgent.decisions",
        description=vision.description,
        decisions=str([asdict(d) for d in decisions]),
    )
    return decisions


# ============================================
# 10. Bin Assets (for user-facing UI)
# ============================================

BIN_ASSETS = {
    "Green Bin (Organics)": {
        "name": "Green Bin",
        "emoji": "🟩",
        "color": "#2ecc71",
    },
    "Recycling (Blue Bin)": {
        "name": "Blue Bin (Recycling)",
        "emoji": "🟦",
        "color": "#3498db",
    },
    "Garbage": {
        "name": "Garbage",
        "emoji": "⬛",
        "color": "#2c3e50",
    },
    "Hazardous Waste": {
        "name": "Hazardous Waste",
        "emoji": "🟥",
        "color": "#e74c3c",
    },
    "Special Collection / Call 3-1-1": {
        "name": "Special Collection",
        "emoji": "🟨",
        "color": "#fbbf24",
    },
    "Refund/Return": {
        "name": "Return for Refund",
        "emoji": "🟪",
        "color": "#a855f7",
    },
    "Varies by material": {
        "name": "Check Material",
        "emoji": "⚪",
        "color": "#9ca3af",
    },
    "Unknown": {
        "name": "Unknown – check city tools",
        "emoji": "❓",
        "color": "#6b7280",
    },
}


# ============================================
# 11. Customer-facing Pipeline
#      Vision Agent → Policy Agent
# ============================================

def run_full_pipeline(img: Image.Image) -> Dict[str, Any]:
    """
    Vision Agent → Policy Agent
    Returns UI-friendly dict.
    """
    # 1) Vision agent (describe)
    vision_res = vision_agent_pil(img)

    # 2) Policy agent (bins)
    policy_decisions = policy_agent(vision_res)

    # Group decisions by bin type
    bins_view: Dict[str, Dict[str, Any]] = {}
    for d in policy_decisions:
        btype = d.bin_type
        if btype not in bins_view:
            asset = BIN_ASSETS.get(btype, {
                "name": btype,
                "emoji": "❓",
                "color": "#6b7280",
            })
            bins_view[btype] = {
                "bin_type": btype,
                "display_name": asset["name"],
                "emoji": asset["emoji"],
                "color": asset["color"],
                "items": [],
                "explanations": [],
            }
        bins_view[btype]["items"].append(d.item)
        bins_view[btype]["explanations"].append(d.explanation)

    return {
        "bins": list(bins_view.values()),
        "raw_description": vision_res.description,
    }


# ============================================
# 12. Streamlit UI
# ============================================

st.set_page_config(page_title="Scan2Sort – Waste Sorting Assistant", page_icon="♻️")

st.title("Scan2Sort – Waste Sorting Assistant")
st.write(
    "Upload a photo of a **waste item** (cup, bottle, food scraps, packaging). "
    "The AI agent will identify the item and tell you which bin to use."
)

uploaded_file = st.file_uploader("Upload an image of your waste item", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
    except Exception:
        st.error("Could not open this image file. Please try with a different image.")
        st.stop()

    if st.button("Classify"):
        with st.spinner("Analyzing image..."):
            # First triage: waste vs not-waste vs blurry
            triage = triage_image(image)
            category = triage.get("category", "blurry")
            triage_desc = triage.get("description", "").strip()

        if category == "blurry":
            st.error("The image is unclear. Please upload a clearer photo of the waste item.")
            st.info("Tip: Make sure the item is in focus and fills most of the frame.")
        elif category == "not_waste":
            st.error("This does not look like a waste item.")
            st.info("Please upload a photo of a waste item like a cup, bottle, packaging, or food scraps.")
        elif category == "waste":
            # Run full agent pipeline
            with st.spinner("Classifying..."):
                result = run_full_pipeline(image)

            st.success("Result")
            st.caption(f"Detected description: _{result['raw_description']}_")

            # Show bins
            for b in result["bins"]:
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
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 18px;
                        ">{b['emoji']}</div>
                        <div>
                            <div style="font-weight: 600; font-size: 14px;">
                                {b['display_name']}
                            </div>
                            <div style="font-size: 13px; color: #4b5563;">
                                Items: {", ".join(b['items'])}
                            </div>
                            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
                                {b['explanations'][0]}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error("Something went wrong while analyzing the image. Please try again.")
else:
    st.info("Upload a clear photo of a waste item to get started.")
