"""Curated static knowledge base for the 38 PlantVillage classes.

This is reference literature-style content only (not model output and not the
prediction itself). Severity estimation is intentionally not provided because
no severity annotations exist in the training data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CarePlan:
    """Structured, class-specific guidance distilled from plant-protection
    reference literature (extension-service style recommendations)."""

    actions: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)   # pesticide/fungicide options
    nutrition: list[str] = field(default_factory=list)  # fertilizer / soil amendment
    prevention: list[str] = field(default_factory=list)


# Healthy crops get a general maintenance plan rather than "no treatment needed".
HEALTHY_PLAN = CarePlan(
    actions=[
        "No disease treatment needed — keep monitoring weekly.",
        "Water at the base in the morning; avoid wetting foliage.",
        "Remove fallen leaves and debris to deny pathogens a winter host.",
    ],
    products=[],
    nutrition=[
        "Feed with a balanced fertilizer (e.g. 10-10-10 NPK) at the label rate for the crop.",
        "Mulch 5–8 cm deep to stabilize moisture and soil temperature.",
    ],
    prevention=[
        "Rotate crops where applicable and sanitize tools between plants.",
    ],
)


def _p(c: CarePlan, **kw) -> CarePlan:
    """Merge helper: override only provided fields."""
    return CarePlan(
        actions=kw.get("actions", c.actions),
        products=kw.get("products", c.products),
        nutrition=kw.get("nutrition", c.nutrition),
        prevention=kw.get("prevention", c.prevention),
    )


# Base plans per pathogen/pest group, then per-crop nuances on top.
FUNGICIDE = "labeled fungicide"

_PLAN_BASE: dict[str, CarePlan] = {
    "scab": CarePlan(
        actions=[
            "Remove and destroy infected leaves and fruit — do not compost.",
            "Prune the canopy to open airflow and speed leaf drying.",
            "Rake and dispose of fallen leaves before budbreak.",
        ],
        products=[
            "Myclobutanil or captan at label rates from green tip through first cover.",
            "Copper/lime-sulfur dormant spray before budbreak to cut overwintering inoculum.",
        ],
        nutrition=[
            "Balanced NPK per soil test; avoid excess nitrogen that favors soft growth.",
        ],
        prevention=[
            "Plant scab-resistant cultivars where available.",
        ],
    ),
    "black_rot_fruit": CarePlan(
        actions=[
            "Prune out mummified fruit, cankers and dead wood with clean cuts.",
            "Remove infected fruit from the tree and the ground.",
        ],
        products=[
            "Thiophanate-methyl or myclobutanil from bloom until 2–4 weeks before harvest.",
        ],
        nutrition=["Balanced fertilizer per soil test."],
        prevention=["Sanitize pruning tools with 70% alcohol between trees."],
    ),
    "rust": CarePlan(
        actions=[
            "Inspect leaves weekly during wet spring periods.",
            "Remove nearby alternate hosts (e.g. juniper/cedar) where practical.",
        ],
        products=[
            "Myclobutanil or propiconazole at label rates starting at pink bud; repeat per label.",
        ],
        nutrition=["Balanced NPK; avoid late-season nitrogen."],
        prevention=["Choose rust-resistant cultivars for new plantings."],
    ),
    "powdery_mildew": CarePlan(
        actions=[
            "Improve air circulation; avoid overhead irrigation.",
            "Prune out heavily infected shoots.",
        ],
        products=[
            "Sulfur or potassium bicarbonate early; myclobutanil or tebuconazole if pressure builds.",
        ],
        nutrition=["Avoid excess nitrogen — lush growth is more susceptible."],
        prevention=["Space plants for airflow; water at the base."],
    ),
    "leaf_spot_cercospora": CarePlan(
        actions=[
            "Rotate crops and bury residue to reduce inoculum.",
            "Improve drainage and avoid overhead irrigation.",
        ],
        products=[
            "Azoxystrobin or propiconazole at tassel/first symptoms; follow label intervals.",
        ],
        nutrition=["Balanced fertility per soil test; ensure adequate potassium."],
        prevention=["Plant resistant hybrids/cultivars."],
    ),
    "corn_rust": CarePlan(
        actions=[
            "Scout weekly from V6 through dent; note pustule density.",
            "Residue management to lower overwintering spores.",
        ],
        products=[
            "Azoxystrobin + propiconazole only under early, severe pressure.",
        ],
        nutrition=["Balanced NPK; avoid stress from nutrient gaps."],
        prevention=["Use rust-resistant hybrids."],
    ),
    "northern_leaf_blight": CarePlan(
        actions=[
            "Rotate away from corn and till under infected residue.",
            "Scout lower leaves at V10–VT.",
        ],
        products=[
            "Azoxystrobin, propiconazole or pyraclostrobin at tassel if lesions reach the ear leaf.",
        ],
        nutrition=["Balanced fertility; fix drainage to reduce leaf wetness."],
        prevention=["Resistant hybrids are the primary defense."],
    ),
    "grape_black_rot": CarePlan(
        actions=[
            "Remove mummies and infected canes; tighten training to speed drying.",
            "Start protective sprays at budbreak and keep to veraison.",
        ],
        products=[
            "Myclobutanil, mancozeb or captan on a 10–14 day label schedule during wet weather.",
        ],
        nutrition=["Balanced NPK; avoid excessive vigor."],
        prevention=["Canopy pruning for airflow; remove wild grapes nearby."],
    ),
    "esca": CarePlan(
        actions=[
            "Prune out affected wood with clean cuts; dispose of debris.",
            "Avoid water stress in summer; do not over-crop vines.",
        ],
        products=[
            "No effective chemical cure — manage as a trunk disease; sore-shoot paints/trunk surgery per specialist.",
        ],
        nutrition=["Maintain balanced nutrition and even moisture."],
        prevention=["Buy certified clean planting stock; protect pruning wounds."],
    ),
    "grape_leaf_blight": CarePlan(
        actions=[
            "Rake and destroy fallen leaves.",
            "Open the canopy with summer pruning.",
        ],
        products=[
            "Protective fungicide (mancozeb or captan) in humid spells.",
        ],
        nutrition=["Balanced fertility per soil test."],
        prevention=["Improve ventilation and reduce leaf wetness duration."],
    ),
    "hlb": CarePlan(
        actions=[
            "No cure exists — remove infected trees to slow spread.",
            "Report suspected HLB to your plant-protection authority.",
            "Control the Asian citrus psyllid vector rigorously.",
        ],
        products=[
            "Labeled psyllid insecticides (rotate modes of action) per local guidance.",
        ],
        nutrition=[
            "Supplemental nutrition keeps infected trees productive longer (per citrus extension programs).",
        ],
        prevention=["Buy certified disease-free nursery trees only."],
    ),
    "bacterial_spot": CarePlan(
        actions=[
            "Remove visibly infected leaves/fruit; sanitize hands and tools.",
            "Avoid overhead irrigation and working in the block when foliage is wet.",
        ],
        products=[
            "Copper hydroxide (often mixed with mancozeb) at label rates; alternate with a labeled biofungicide.",
            "Streptomycin is not registered in many regions for this use — check local labels.",
        ],
        nutrition=["Avoid excess nitrogen vigor; balanced fertility."],
        prevention=["Certified disease-free seed/seedlings; 2–3 year rotation."],
    ),
    "potato_early_blight": CarePlan(
        actions=[
            "Remove infected lower leaves; avoid leaf wetness.",
            "Rotate crops (2–3 years) and bury plant debris.",
        ],
        products=[
            "Chlorothalonil or azoxystrobin when first lesions appear on lower leaves.",
        ],
        nutrition=[
            "Maintain even nitrogen; keep potassium adequate — stressed plants blight sooner.",
        ],
        prevention=["Drip irrigate; mulch to reduce soil splash."],
    ),
    "late_blight": CarePlan(
        actions=[
            "Act immediately — this disease spreads explosively in cool, wet weather.",
            "Destroy infected haulms (cut and bag, do not compost).",
            "Report suspected outbreaks to agricultural authorities.",
        ],
        products=[
            "Mancozeb or chlorothalonil protectants; cymoxanil or propamocarb systemics where registered.",
        ],
        nutrition=["Balanced fertility; avoid lush canopy from excess nitrogen."],
        prevention=["Certified seed; destroy volunteer potatoes/tomatoes."],
    ),
    "leaf_scorch": CarePlan(
        actions=[
            "Renovate beds after harvest; remove infected runners and leaves.",
            "Drip irrigate; avoid overhead wetting.",
        ],
        products=[
            "Captan or thiram on a label schedule in humid seasons.",
        ],
        nutrition=["Balanced fertility; avoid drought stress."],
        prevention=["Mulch to keep fruit and foliage off wet soil."],
    ),
    "leaf_mold": CarePlan(
        actions=[
            "Ventilate greenhouses; drop humidity below 85%.",
            "Remove and destroy infected leaves.",
        ],
        products=[
            "Chlorothalonil or a labeled leaf-mold fungicide; biofungicides (Bacillus) as protectants.",
        ],
        nutrition=["Balanced fertility; avoid overwatering."],
        prevention=["Resistant cultivars; wide spacing; staking for airflow."],
    ),
    "septoria": CarePlan(
        actions=[
            "Remove infected lower leaves promptly; mulch to block soil splash.",
            "Water at the base; avoid wetting foliage.",
        ],
        products=[
            "Chlorothalonil or copper at first symptoms; repeat per label interval.",
        ],
        nutrition=["Balanced fertility; calcium and potassium support leaf resilience."],
        prevention=["2-year rotation; resistant varieties."],
    ),
    "spider_mites": CarePlan(
        actions=[
            "Rinse plants with water to knock down mites and dust.",
            "Avoid broad-spectrum insecticides that kill natural enemies.",
        ],
        products=[
            "Insecticidal soap, horticultural oil or a labeled miticide (bifenthrin/abamectin where registered); rotate modes of action.",
            "Releases of Phytoseiulus persimilis predatory mites in greenhouses.",
        ],
        nutrition=["Avoid drought-stressed plants — mites explode on stressed foliage."],
        prevention=["Monitor leaf undersides weekly in hot, dry spells."],
    ),
    "target_spot": CarePlan(
        actions=[
            "Reduce humidity and leaf wetness; remove infected leaves.",
            "Dispose of crop debris after harvest.",
        ],
        products=[
            "Chlorothalonil or fluopyram/tebuconazole labeled for target spot.",
        ],
        nutrition=["Balanced fertility; avoid stress."],
        prevention=["Ventilation and wide spacing; sanitize stakes/trays."],
    ),
    "tylcv": CarePlan(
        actions=[
            "Remove and destroy infected plants immediately — virus has no cure.",
            "Install fine insect netting (50-mesh) over vents and doors.",
            "Control whitefly vectors with yellow sticky traps plus sprays.",
        ],
        products=[
            "Whitefly-targeted insecticides (imidacloprid/dinotefuran as soil drench or foliar; rotate modes of action) per label.",
            "Horticultural oil or insecticidal soap against nymphs.",
        ],
        nutrition=["Keep young transplants vigorous; balanced starter fertilizer."],
        prevention=["TYLCV-resistant hybrids; control weeds that host whiteflies."],
    ),
    "tomato_mosaic": CarePlan(
        actions=[
            "Rogue out infected plants including roots.",
            "Disinfect hands, tools and stakes (10% bleach or 20% nonfat dry milk solution).",
            "Control weed reservoirs around the field.",
        ],
        products=[
            "No curative spray for the virus; prevent only (sanitation + resistant varieties).",
        ],
        nutrition=["Balanced fertility to keep plants competitive."],
        prevention=["Buy certified virus-indexed seed/seedlings; handle plants rarely and gently."],
    ),
    "squash_pm": CarePlan(
        actions=[
            "Water at the base in the morning; improve airflow.",
            "Prune the worst-infected leaves.",
        ],
        products=[
            "Sulfur or potassium bicarbonate early; myclobutanil or horticultural oil if spreading.",
        ],
        nutrition=["Avoid excess nitrogen."],
        prevention=["Powdery-mildew-tolerant varieties."],
    ),
}

_BASE_FOR_CLASS: dict[str, str] = {
    "Apple___Apple_scab": "scab",
    "Apple___Black_rot": "black_rot_fruit",
    "Apple___Cedar_apple_rust": "rust",
    "Cherry_(including_sour)___Powdery_mildew": "powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "leaf_spot_cercospora",
    "Corn_(maize)___Common_rust_": "corn_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "northern_leaf_blight",
    "Grape___Black_rot": "grape_black_rot",
    "Grape___Esca_(Black_Measles)": "esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape_leaf_blight",
    "Orange___Haunglongbing_(Citrus_greening)": "hlb",
    "Peach___Bacterial_spot": "bacterial_spot",
    "Pepper,_bell___Bacterial_spot": "bacterial_spot",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "late_blight",
    "Squash___Powdery_mildew": "squash_pm",
    "Strawberry___Leaf_scorch": "leaf_scorch",
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "potato_early_blight",  # same pathogen, same strategy
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
    "Tomato___Target_Spot": "target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tylcv",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus_": "tylcv",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic",
}


def _crop_nuance(class_name: str) -> dict:
    if class_name.startswith("Apple"):
        return {
            "nutrition": ["Fruit trees: feed in early spring per soil test; avoid late nitrogen."],
        }
    if class_name.startswith(("Tomato", "Potato")):
        return {
            "nutrition": [
                "Side-dress with balanced fertilizer after first fruit set; calcium nitrate helps against blossom-end rot.",
            ],
        }
    if class_name.startswith(("Pepper", "Squash")):
        return {
            "nutrition": [
                "Balanced fertilizer every 2–3 weeks; avoid overwatering.",
            ],
        }
    if class_name.startswith("Corn"):
        return {
            "nutrition": [
                "Side-dress nitrogen at V6 and again at tassel per soil test.",
            ],
        }
    if class_name.startswith("Grape"):
        return {
            "nutrition": [
                "Balanced vineyard fertilizer in spring; leaf analysis is best.",
            ],
        }
    if class_name.startswith("Orange"):
        return {
            "nutrition": [
                "Citrus-specific fertilizer with micronutrients (Zn, Mn, Fe) 3x per year.",
            ],
        }
    return {}


# class -> {common_name, one-line description, treatment guidance}
DISEASE_KB: dict[str, dict[str, str]] = {
    "Apple___Apple_scab": {
        "common_name": "Apple scab (Venturia inaequalis)",
        "description": "Dark olive-to-brown lesions and corky scabs on leaves and fruit.",
        "treatment": "Remove infected leaves, improve airflow by pruning, and apply a labeled scab fungicide in wet spells.",
    },
    "Apple___Black_rot": {
        "common_name": "Apple black rot (Botryosphaeria obtusa)",
        "description": "Frogeye leaf spots and black, rotted fruit lesions.",
        "treatment": "Prune out dead wood and mummified fruit; apply a registered protective fungicide during bloom.",
    },
    "Apple___Cedar_apple_rust": {
        "common_name": "Cedar apple rust (Gymnosporangium juniperi-virginianae)",
        "description": "Bright orange-yellow spots with black pycnia on leaves.",
        "treatment": "Reduce nearby cedar junipers where possible; use a preventive rust fungicide at the correct phenology stage.",
    },
    "Apple___healthy": {"common_name": "Apple (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed; maintain good orchard hygiene."},
    "Blueberry___healthy": {"common_name": "Blueberry (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed; maintain good bush care."},
    "Cherry_(including_sour)___Powdery_mildew": {
        "common_name": "Cherry powdery mildew (Podosphaera clandestina)",
        "description": "White, powdery fungal growth on young leaves and shoots.",
        "treatment": "Improve air circulation, avoid overhead irrigation, and apply a powdery-mildew fungicide when disease pressure is high.",
    },
    "Cherry_(including_sour)___healthy": {"common_name": "Cherry (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "common_name": "Corn gray leaf spot (Cercospora zeae-maydis)",
        "description": "Narrow, tan-gray rectangular lesions parallel to leaf veins.",
        "treatment": "Rotate crops, plant resistant hybrids, and apply a labeled corn-leaf-spot fungicide at tassel when risk is high.",
    },
    "Corn_(maize)___Common_rust_": {
        "common_name": "Corn common rust (Puccinia sorghi)",
        "description": "Small, scattered, cinnamon-brown pustules on both leaf surfaces.",
        "treatment": "Usually late-season and mild; use resistant hybrids and apply a rust fungicide only under severe, early pressure.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "common_name": "Corn northern leaf blight (Exserohilum turcicum)",
        "description": "Large, cigar-shaped tan lesions with dark margins on leaves.",
        "treatment": "Rotate crops, till in residue, use resistant hybrids, and apply a foliar fungicide if lesions reach the ear leaf pre-tassel.",
    },
    "Corn_(maize)___healthy": {"common_name": "Maize (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Grape___Black_rot": {
        "common_name": "Grape black rot (Guignardia bidwellii)",
        "description": "Circular tan leaf spots with dark borders; berries shrivel into black mummies.",
        "treatment": "Remove mummies and infected canes, improve canopy ventilation, and apply a protective grape fungicide from budbreak to veraison.",
    },
    "Grape___Esca_(Black_Measles)": {
        "common_name": "Grape esca / black measles (wood decay complex)",
        "description": "Interveinal tiger-stripe leaf chlorosis and black bark lesions; fruit shrivals and fails to color.",
        "treatment": "Prune out affected wood with clean cuts, avoid excessive summer stress, and manage as a trunk disease complex.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "common_name": "Grape isariopsis leaf spot",
        "description": "Irregular brown blotches that cause premature leaf drop.",
        "treatment": "Rake and remove fallen leaves, open the canopy, and use a protective grape fungicide under humid conditions.",
    },
    "Grape___healthy": {"common_name": "Grape (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Orange___Haunglongbing_(Citrus_greening)": {
        "common_name": "Citrus greening (Huanglongbing, HLB)",
        "description": "Asymmetric blotchy leaf mottling and lopsided, bitter, poorly colored fruit.",
        "treatment": "No cure exists; manage the psyllid vector, remove infected trees to slow spread, and report to the plant-protection authority.",
    },
    "Peach___Bacterial_spot": {
        "common_name": "Peach bacterial spot (Xanthomonas arboricola pv. pruni)",
        "description": "Small, angular, water-soaked leaf spots with shot-holing and cracked fruit.",
        "treatment": "Use resistant cultivars, copper-based sprays in spring, and avoid high-nitrogen vigor.",
    },
    "Peach___healthy": {"common_name": "Peach (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Pepper,_bell___Bacterial_spot": {
        "common_name": "Bell pepper bacterial spot (Xanthomonas spp.)",
        "description": "Small, dark, raised leaf spots with yellow haloes; spots on fruit.",
        "treatment": "Use clean seed, rotate crops, apply copper or labelled bactericide, and avoid overhead irrigation.",
    },
    "Pepper,_bell___healthy": {"common_name": "Bell pepper (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Potato___Early_blight": {
        "common_name": "Potato early blight (Alternaria solani)",
        "description": "Concentric 'target-board' brown lesions with yellow halos on older leaves.",
        "treatment": "Rotate crops, irrigate/front fertilize to avoid stress, and apply a labelled potato fungicide as lesions appear on lower leaves.",
    },
    "Potato___Late_blight": {
        "common_name": "Potato late blight (Phytophthora infestans)",
        "description": "Large water-soaked, necrotic leaf blotches with white sporulation on the underside.",
        "treatment": "Apply a labelled late-blight fungicide immediately, destroy infected haulm, and notify authorities given its epidemic potential.",
    },
    "Potato___healthy": {"common_name": "Potato (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Raspberry___healthy": {"common_name": "Raspberry (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Soybean___healthy": {"common_name": "Soybean (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Squash___Powdery_mildew": {
        "common_name": "Squash powdery mildew (Podosphaera xanthii)",
        "description": "White powdery spots that expand to coat leaves, causing yellowing.",
        "treatment": "Improve airflow, water at the base, and apply a powdery-mildew fungicide or sulfur early.",
    },
    "Strawberry___Leaf_scorch": {
        "common_name": "Strawberry leaf scorch (Diplocarpon earlianum)",
        "description": "Purple-to-brown blotches and scorched leaf edges.",
        "treatment": "Renovate beds, remove infected runners, and apply a labelled strawberry fungicide in humid seasons.",
    },
    "Strawberry___healthy": {"common_name": "Strawberry (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
    "Tomato___Bacterial_spot": {
        "common_name": "Tomato bacterial spot (Xanthomonas perforans)",
        "description": "Small, dark, sunken spots with yellow haloes on leaves and fruit.",
        "treatment": "Use disease-free seed/seedlings, copper or labelled bactericide, and avoid overhead wetting.",
    },
    "Tomato___Early_blight": {
        "common_name": "Tomato early blight (Alternaria solani)",
        "description": "Concentric brown rings on lower leaves with yellow halos.",
        "treatment": "Mulch to reduce splash, rotate crops, and apply a labelled early-blight fungicide when the first lesions appear.",
    },
    "Tomato___Late_blight": {
        "common_name": "Tomato late blight (Phytophthora infestans)",
        "description": "Greasy dark lesions expand fast; white fuzzy growth under humid conditions.",
        "treatment": "Act immediately with a labelled late-blight fungicide (e.g. chlorothalonil/mancozeb + systemic), and destroy heavily infected plants.",
    },
    "Tomato___Leaf_Mold": {
        "common_name": "Tomato leaf mold (Passalora fulva)",
        "description": "Pale-yellow upper-leaf spots with olive-brown fuzzy mould beneath.",
        "treatment": "Increase greenhouse ventilation, lower humidity, and apply a labelled leaf-mold fungicide.",
    },
    "Tomato___Septoria_leaf_spot": {
        "common_name": "Tomato septoria leaf spot (Septoria lycopersici)",
        "description": "Small, circular grey-brown spots with dark borders and dark pycnidia.",
        "treatment": "Remove lower infected leaves, avoid wetting foliage, and apply a labelled tomato fungicide.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "common_name": "Tomato two-spotted spider mite",
        "description": "Fine stippling, webbing and bronzing on leaves in hot, dry conditions.",
        "treatment": "Release predatory mites, avoid mite-blooming broad-spectrum insecticides, and manage dust and heat stress.",
    },
    "Tomato___Target_Spot": {
        "common_name": "Tomato target spot (Corynespora cassiicola)",
        "description": "Ringed, bull's-eye necrotic spots and yellow margins.",
        "treatment": "Reduce humidity and leaf wetness, remove inoculum, and apply a labelled tomato fungicide under warm, wet conditions.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "common_name": "Tomato yellow leaf curl virus (TYLCV)",
        "description": "Upward-cupped yellow leaves with stunted growth, vectored by whiteflies.",
        "treatment": "Control whiteflies with nets/insecticides, remove infected plants, and use TYLCV-resistant hybrids.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus_": {  # normalized duplicate (with trailing underscore)
        "common_name": "Tomato yellow leaf curl virus (TYLCV)",
        "description": "Upward-cupped yellow leaves with stunted growth, vectored by whiteflies.",
        "treatment": "Control whiteflies with nets/insecticides, remove infected plants, and use TYLCV-resistant hybrids.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "common_name": "Tomato mosaic virus (ToMV)",
        "description": "Mottled light/dark mosaic on leaves, sometimes leaf distortion and stunting.",
        "treatment": "Use resistant varieties, sanitize hands/tools and stakes, and rogue out infected plants and weeds.",
    },
    "Tomato___healthy": {"common_name": "Tomato (healthy)", "description": "No pest or disease symptoms detected.", "treatment": "No treatment needed."},
}

GENERIC_GUIDANCE = (
    "No specific guidance is recorded for this class in the built-in reference."
    " A trained specialist should confirm the diagnosis and recommend treatment."
)


def kb_for(class_name: str) -> dict[str, str]:
    return DISEASE_KB.get(class_name, {
        "common_name": class_name.replace("___", " - ").replace("_", " "),
        "description": "Class from the PlantVillage dataset.",
        "treatment": GENERIC_GUIDANCE,
    })


def care_plan_for(class_name: str) -> dict:
    """Structured care plan (actions / products / nutrition / prevention) for a
    predicted class. Healthy classes get a maintenance plan; unknown classes
    get generic honest guidance."""
    if "healthy" in class_name:
        plan = HEALTHY_PLAN
    else:
        base = _PLAN_BASE.get(_BASE_FOR_CLASS.get(class_name, ""))
        if base is None:
            plan = CarePlan(
                actions=[
                    "Confirm the diagnosis with a local agricultural extension officer or specialist.",
                    "Remove and destroy severely infected tissue; do not compost it.",
                ],
                products=[],
                nutrition=["Balanced fertilizer per soil test."],
                prevention=["Crop rotation, sanitation and resistant varieties."],
            )
        else:
            plan = _p(base, **_crop_nuance(class_name))
    return {
        "actions": list(plan.actions),
        "products": list(plan.products),
        "nutrition": list(plan.nutrition),
        "prevention": list(plan.prevention),
        "scope": "Static reference guidance distilled from extension literature; verify product "
                 "registration in your region and follow the label exactly.",
    }
