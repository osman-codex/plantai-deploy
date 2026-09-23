"""Curated static knowledge base for the 38 PlantVillage classes.

This is reference literature-style content only (not model output and not the
prediction itself). Severity estimation is intentionally not provided because
no severity annotations exist in the training data.
"""
from __future__ import annotations

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