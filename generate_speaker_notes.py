"""
Generates ADAS_AEB_Speaker_Notes.docx — complete, slide-by-slide speaking notes
for the 9-slide IIT Roorkee presentation.

Run:
    python3 generate_speaker_notes.py
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUTPUT = "/home/atharv/ADAS-Pedestrian-AEB/reports/ADAS_AEB_Speaker_Notes.docx"


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)


def add_slide_header(doc, slide_num, total, title, duration):
    """Teal banner row with slide number, title, and estimated duration."""
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    r = table.rows[0]

    r.cells[0].text = f"SLIDE {slide_num} / {total}"
    r.cells[1].text = title
    r.cells[2].text = f"≈ {duration} min"

    for i, cell in enumerate(r.cells):
        set_cell_bg(cell, '005050')
        for para in cell.paragraphs:
            para.alignment = [WD_ALIGN_PARAGRAPH.LEFT,
                               WD_ALIGN_PARAGRAPH.CENTER,
                               WD_ALIGN_PARAGRAPH.RIGHT][i]
            for run in para.runs:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.bold      = True
                run.font.size      = Pt(11)

    # Fix column widths
    widths = [Inches(1.2), Inches(5.0), Inches(1.3)]
    for i, cell in enumerate(r.cells):
        cell.width = widths[i]

    doc.add_paragraph()


def add_section(doc, label, color_hex='005050'):
    """Teal sub-section label (e.g. TALKING POINTS, KEY EMPHASES)."""
    p = doc.add_paragraph()
    run = p.add_run(f"  {label}  ")
    run.font.bold      = True
    run.font.size      = Pt(10)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # background shading via XML
    rPr = run._r.get_or_add_rPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  color_hex)
    rPr.append(shd)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(2)


def add_note(doc, text, indent=0.3):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent   = Inches(indent)
    p.paragraph_format.space_before  = Pt(1)
    p.paragraph_format.space_after   = Pt(1)
    run = p.add_run(text)
    run.font.size = Pt(10.5)


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    for run in p.runs:
        run.font.size = Pt(10.5)


def add_transition(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(f"TRANSITION → {text}")
    run.font.bold      = True
    run.font.italic    = True
    run.font.size      = Pt(10)
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(8)


def add_emphasis_box(doc, text):
    """Pale teal info box for key numbers or formulas."""
    table = doc.add_table(rows=1, cols=1)
    table.style = 'Table Grid'
    cell = table.rows[0].cells[0]
    cell.text = text
    set_cell_bg(cell, 'E0F5F5')
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.size  = Pt(10)
            run.font.bold  = True
            run.font.color.rgb = RGBColor(0x00, 0x50, 0x50)
    doc.add_paragraph()


def hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'),  '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '00A0A0')
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(6)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    doc = Document()

    # ── Document-level styles ──────────────────────────────────────────────────
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10.5)

    # Page margins
    for section in doc.sections:
        section.top_margin    = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin   = Inches(0.90)
        section.right_margin  = Inches(0.90)

    # Title page ────────────────────────────────────────────────────────────────
    title_para = doc.add_heading('Speaker Notes', level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        'Physics-Aware Pedestrian Automatic Emergency Braking (AEB) System\n'
        'Department of Mechanical & Industrial Engineering  |  IIT Roorkee  |  May 2026\n'
        'MIT-115 End-Term Presentation  —  9 Slides  |  Total ≈ 20–25 minutes')
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.font.size = Pt(11)
    doc.add_paragraph()
    hr(doc)

    add_emphasis_box(doc,
        "HOW TO USE THESE NOTES\n"
        "Each slide has four sections:\n"
        "  • TALKING POINTS  — what to say, in roughly the order you should say it\n"
        "  • KEY EMPHASES    — numbers or phrases worth repeating for impact\n"
        "  • ANTICIPATE      — likely audience questions and short answers\n"
        "  • TRANSITION      — how to bridge to the next slide\n\n"
        "Estimated total delivery time: 20–25 minutes at a comfortable pace.\n"
        "Slide 1 (title) is 1 min; result slides 4–8 are 2.5–3 min each.")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 1 — TITLE
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 1, 9,
        "Physics-Aware Pedestrian AEB System", "1")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "Good [morning/afternoon], everyone. I'm Atharv Priyadarshi. "
        "Today I'll be presenting our end-term project for MIT-115: "
        "a Physics-Aware Pedestrian Automatic Emergency Braking system — "
        "or AEB — developed entirely in Python, from raw video through braking physics "
        "to injury analysis.")
    add_note(doc,
        "The word 'physics-aware' is the key differentiator. "
        "Every commercial AEB system you read about in product brochures uses "
        "a simple kinematic formula — distance equals v-squared over 2a. "
        "We'll show today that formula can underestimate stopping distance by 15–30 percent, "
        "which means it would trigger braking too late in real-world conditions.")
    add_note(doc,
        "Our system replaces that approximation with four physics layers: "
        "the Pacejka tyre model for grip, Newton's cooling law for brake temperature, "
        "dlib facial landmarks for driver drowsiness, and the Head Injury Criterion "
        "for pedestrian biomechanics. We then validate the whole pipeline against "
        "Euro NCAP's four standardised pedestrian scenarios.")
    add_note(doc,
        "This slide also shows our team and department affiliation. "
        "The project runs January through July 2026 — you'll see the timeline on Slide 3.")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "Physics-aware ≠ kinematic. This distinction is the thesis of the talk.")
    add_note(doc, "Four integrated physics layers — not one formula, not a lookup table.")
    add_note(doc, "End-to-end: from camera pixel to injury severity score.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: Is this a simulation or real vehicle?  "
        "A: This is a software simulation validated against published reference data "
        "and Euro NCAP benchmarks; hardware deployment on a Jetson Orin is listed "
        "as future work.")

    add_transition(doc,
        "\"Let me walk you through what we set out to achieve "
        "and how the system is structured — Slide 2.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 2 — OBJECTIVES & METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 2, 9,
        "Research Objectives and Methodology", "2")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "This slide has three parts: five research objectives on the upper left, "
        "a methodology summary on the lower left, and the 14-step pipeline flowchart "
        "on the right. Let me walk through all three.")
    add_note(doc,
        "Objective 1 is detection: we fine-tune YOLOv8n — the nano variant, "
        "fast enough for real-time — on 6,000 images from the Berkeley DeepDrive dataset. "
        "Our design target was 75% mean Average Precision at IoU 0.5 for pedestrians. "
        "I'll show the full breakdown on Slide 4.")
    add_note(doc,
        "Objective 2 moves to physics. We implement longitudinal braking using "
        "the Pacejka Magic Formula — a tyre model used in professional racing simulators. "
        "It captures the non-linear relationship between wheel slip and lateral force "
        "that the simple mu-times-N-times-g model completely ignores. "
        "We run the integrator at a 1-millisecond timestep to catch ABS pressure "
        "cycling accurately.")
    add_note(doc,
        "Objective 3 adds thermodynamics. Brake rotors heat up during repeated "
        "emergency stops, and hot pads produce less friction — this is brake fade. "
        "We model heating with the energy equation and cooling with Newton's law, "
        "then feed the resulting friction penalty directly back into the AEB trigger. "
        "No commercial AEB does this.")
    add_note(doc,
        "Objective 4 is the adaptive controller. "
        "The base TTC trigger is 1.5 seconds, adjusted in real time for weather, "
        "driver alertness, and brake thermal state. "
        "In the worst case — fog, drowsy driver, faded brakes — "
        "the threshold rises to 3.5 seconds.")
    add_note(doc,
        "Objective 5 closes the loop with validation: "
        "four Euro NCAP pedestrian scenarios across five impact speeds, "
        "and the HIC biomechanical model to quantify injury severity.")
    add_note(doc,
        "Below the objectives is the methodology summary — five points that describe "
        "how each layer is actually implemented. "
        "Detection uses YOLOv8n with CLAHE preprocessing and SORT tracking. "
        "Depth estimation uses the pinhole model to convert bounding-box pixel height "
        "into a 3D world position and project it onto a top-down BEV radar. "
        "Braking physics runs a five-force Pacejka integrator at 1 ms resolution with ABS. "
        "Thermodynamics tracks rotor temperature continuously and outputs a fade penalty. "
        "Validation runs a 20-combination Euro NCAP matrix and scores outcomes with HIC.")
    add_note(doc,
        "On the right: the 14-step pipeline flowchart gives the visual equivalent. "
        "Steps 1–7 in the left column cover perception — from raw frame to "
        "ego-corridor-filtered tracks and weather classification. "
        "The L-connector routes into the right column: driver monitoring, "
        "brake thermodynamics, TTC, AEB decision, Pacejka forces, NCAP validation, "
        "and the BEV radar output. Every frame, all 14 steps run in sequence.")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc,
        "Five objectives map directly to five methodology points — "
        "each objective has a concrete implementation behind it.")
    add_note(doc,
        "1 ms timestep and Newton's cooling law — these are the two methodological "
        "choices that separate this from a kinematic AEB.")
    add_note(doc,
        "The flowchart is the complete per-frame execution order — "
        "perception feeds physics feeds decision, no shortcuts.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: Why YOLOv8 nano instead of a larger variant?  "
        "A: The 'n' variant runs at 15–20 fps on a Tesla T4; larger models "
        "exceed our real-time budget. The mAP gap to YOLOv8s is only ~3 percentage "
        "points for this dataset, which we judged acceptable.")
    add_note(doc,
        "Q: Why SORT and not DeepSORT or ByteTrack?  "
        "A: SORT's Kalman+Hungarian pipeline is computationally minimal and "
        "gives us persistent track IDs at our frame rate. ReID appearance features "
        "add latency without meaningful accuracy gain at the short occlusion durations "
        "typical in these scenarios.")
    add_note(doc,
        "Q: Is the pinhole depth model accurate enough for AEB?  "
        "A: For adult pedestrians at known height (1.7 m), error is under 10% "
        "within 30 metres — sufficient for TTC computation at AEB-relevant ranges. "
        "Stereo or radar fusion is the accuracy upgrade path listed in future work.")

    add_transition(doc,
        "\"Before the results, here's how we planned and executed the work — Slide 3.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 3 — TIMELINE
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 3, 9,
        "Project Timeline — Gantt Chart", "1–1.5")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "The Gantt chart covers January through July 2026 — seven months, "
        "twelve parallel task streams, and the yellow 'NOW' marker sits at "
        "the end of June, confirming we are on schedule for this presentation.")
    add_note(doc,
        "The project follows a logical build-up. "
        "January: dataset preparation — converting BDD100K annotations "
        "from the native format into YOLO-compatible bounding boxes. "
        "This took about six weeks including quality checks and class rebalancing.")
    add_note(doc,
        "February: model training — 30 epochs on a Tesla T4 cloud GPU. "
        "Training converged by epoch 22, at which point validation loss plateaued "
        "and we stopped early to avoid overfitting.")
    add_note(doc,
        "March and April: the perception stack — SORT tracker, MiDaS depth, "
        "lane detection, and ego corridor filtering all run in parallel threads. "
        "Also in March, we began vehicle dynamics development alongside perception, "
        "so the physics model was ready to integrate by April.")
    add_note(doc,
        "April–May: the intelligence layer — brake thermodynamics, weather classifier, "
        "driver monitoring, and the adaptive AEB controller itself. "
        "These four modules were developed and unit-tested independently "
        "before integration.")
    add_note(doc,
        "Late May to June: HIC biomechanics, Euro NCAP validation runs, "
        "report writing, and this presentation.")
    add_note(doc,
        "One deliberate choice: we excluded CARLA simulation integration "
        "from this iteration. Running a photorealistic simulator "
        "would have consumed two additional months we didn't have; "
        "it remains first priority in future work.")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "12 tasks, 7 months — all on schedule.")
    add_note(doc, "Perception and physics developed in parallel to save time.")
    add_note(doc, "CARLA deferred — but algorithm is simulator-ready.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: Did you encounter any schedule slippage?  "
        "A: MiDaS depth integration took one extra week because of coordinate "
        "system mismatches between the depth map and the YOLO bounding boxes. "
        "We absorbed it by parallelising weather classifier development.")

    add_transition(doc,
        "\"Now let's go into the results, starting with object detection and tracking.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 4 — DETECTION & TRACKING
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 4, 9,
        "Results: Object Detection and Tracking", "2.5–3")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "The left panel shows two plots. "
        "The bar chart on the left compares mAP at IoU 0.5 — the standard detection metric — "
        "and the stricter mAP at 0.5-to-0.95 across all four classes and the overall mean. "
        "The right plot shows the training loss curves over 30 epochs.")
    add_note(doc,
        "Let me read the headline numbers. "
        "Pedestrian mAP@0.5: 78.3% — this is our primary design target and we exceed it. "
        "Car: 89.1% — cars are larger and more visually distinct, hence the highest score. "
        "Bicycle: 71.2%. Motorcycle: 69.8% — these two are the hardest to separate "
        "because their aspect ratios and silhouettes overlap heavily at distance. "
        "Overall mAP@0.5: 77.1% across all four classes.")
    add_note(doc,
        "The stricter mAP@0.5:0.95 — averaged over IoU thresholds from 0.5 to 0.95 — "
        "is 44.8% overall. That drop is expected: it reflects localisation precision "
        "rather than detection recall, and YOLOv8n's anchor-free head trades some "
        "box tightness for speed.")
    add_note(doc,
        "The training convergence plot on the right shows box loss in blue "
        "and classification loss in red. Both descend smoothly and level off "
        "by epoch 22, which is when we applied early stopping. "
        "No divergence, no oscillation — the learning rate schedule and "
        "CLAHE preprocessing together stabilised training.")
    add_note(doc,
        "For tracking: SORT uses a 7-dimensional Kalman state — "
        "centre x, centre y, scale, aspect ratio, and their time derivatives. "
        "The Hungarian algorithm assigns each new detection to an existing track "
        "if the bounding box Intersection over Union exceeds 0.30. "
        "A new track requires three consecutive hits before we report it to the "
        "AEB engine, eliminating spurious detections. "
        "A track is deleted after one missed frame to avoid latency on "
        "fast-moving pedestrians. "
        "Each confirmed track outputs x1, y1, x2, y2, track ID, and class ID — "
        "exactly what the depth and corridor modules need.")
    add_note(doc,
        "The bottom of the slide shows the formula strip: "
        "IoU is the intersection area divided by union area — the foundation of "
        "detection metrics. "
        "Precision is true positives over true positives plus false positives — "
        "how many detections are real. "
        "Recall is true positives over true positives plus false negatives — "
        "how many real pedestrians we catch. "
        "mAP is the mean AP averaged over all classes and IoU thresholds.")

    add_emphasis_box(doc,
        "KEY NUMBERS: Pedestrian 78.3%  |  Car 89.1%  |  Bicycle 71.2%  |  "
        "Motorcycle 69.8%  |  Overall 77.1%  |  mAP@0.5:0.95 44.8%")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "78.3% pedestrian mAP — exceeds the 75% safety design target.")
    add_note(doc, "Motorcycle/Bicycle confusion is the main error mode — "
                  "common in all monocular detectors at this scale.")
    add_note(doc, "Convergence at epoch 22 — well-behaved training, no overfitting.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: How does 78.3% compare to state of the art?  "
        "A: Top models on full BDD100K pedestrian reach 85–87% mAP@0.5 using "
        "YOLOv8x (10× more parameters) or transformer architectures with "
        "test-time augmentation. Our 78.3% at real-time speed on YOLOv8n "
        "is a strong result for the model class.")
    add_note(doc,
        "Q: Does the system handle occlusions?  "
        "A: SORT's Kalman filter predicts track position during short occlusions. "
        "The max_age=1 setting means we delete after one missed frame, "
        "which is aggressive but avoids ghost tracks that could trigger false AEB.")

    add_transition(doc,
        "\"Detection tells us there is a pedestrian and where it is. "
        "Now we need to know how fast we can stop — Slide 5, vehicle dynamics.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 5 — VEHICLE DYNAMICS
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 5, 9,
        "Results: Vehicle Dynamics and ABS Simulation", "2.5–3")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "This slide answers the question: given a detected pedestrian at a "
        "measured depth, can the vehicle actually stop in time? "
        "And the answer depends critically on which physics model you use.")
    add_note(doc,
        "The bar chart compares stopping distance across five road surfaces "
        "at 50 km/h and 100 km/h. "
        "Let me highlight the key rows. On dry asphalt with friction coefficient 0.85, "
        "at 50 km/h we stop in 16.2 metres. "
        "Published reference ranges for this condition are 13 to 22 metres — "
        "our model is within range. "
        "On ice, friction drops to 0.10 and stopping distance at 50 km/h "
        "explodes to 145.3 metres — again, within the published reference of "
        "60 to 150 metres.")
    add_note(doc,
        "Now, why does the simple kinematic formula fail? "
        "Kinematic uses d = v-squared over 2-mu-g. "
        "It assumes constant friction, ignores weight transfer, ignores rolling resistance, "
        "ignores aerodynamic drag, and assumes the wheels never lock. "
        "Our model adds all four of these effects. "
        "The result is that on the same dry surface at 50 km/h, "
        "the kinematic formula predicts about 14 metres — a 15% underestimate. "
        "On wet or low-friction surfaces the gap grows to 30%. "
        "An AEB calibrated on kinematic distances would trigger 15–30% too late.")
    add_note(doc,
        "The Pacejka Magic Formula captures the tyre's non-linear behaviour. "
        "At low slip — wheel spinning slightly slower than vehicle speed — "
        "lateral force rises quickly. "
        "But past the optimal slip of about 0.1 to 0.2, the tyre saturates "
        "and force actually drops. "
        "ABS exploits this: it cycles brake pressure at 10 Hz to keep the wheel "
        "hovering near that optimal slip point. "
        "Our ABS reduces stopping distance by roughly 25% compared to locked wheels.")
    add_note(doc,
        "The weight transfer term is also important. "
        "When you brake hard, the nose dips and front axle load increases. "
        "We compute delta-N-axle as mass times longitudinal deceleration times "
        "centre-of-gravity height divided by wheelbase. "
        "This shifts more load to the front wheels, boosting front grip — "
        "which is why performance braking distances with physics models are "
        "shorter than naive estimates but still longer than kinematic ones "
        "once all resistances are included.")
    add_note(doc,
        "The formula strip at the bottom summarises the five forces: "
        "Pacejka, aerodynamic drag, rolling resistance, road grade, "
        "and the weight transfer term for each axle.")

    add_emphasis_box(doc,
        "50 km/h dry:  Pacejka 16.2 m  vs  Kinematic ~14 m  (−15%)\n"
        "50 km/h ice:  Pacejka 145.3 m  vs  Kinematic ~66 m  (−55%)\n"
        "ABS reduces stopping distance ≈ 25% vs locked wheels")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "Kinematic underestimates stopping distance by 15–30% — this is dangerous.")
    add_note(doc, "Pacejka + weight transfer + ABS = the correct physics model.")
    add_note(doc, "All 6 test cases validated within published reference ranges.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: What vehicle parameters did you use?  "
        "A: A 1,500 kg compact sedan: wheelbase 2.7 m, CoG height 0.55 m, "
        "frontal area 2.2 m², drag coefficient 0.30. "
        "These are typical mid-size hatchback values.")
    add_note(doc,
        "Q: Did you validate with physical test data?  "
        "A: We validated against published reference ranges from the literature "
        "(ISO 21994, Euro NCAP technical papers). "
        "All six surface-speed combinations fall within those envelopes.")

    add_transition(doc,
        "\"We can model cold brakes accurately. But what happens after "
        "six emergency stops? The brakes overheat — and that changes everything. Slide 6.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 6 — BRAKE THERMODYNAMICS
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 6, 9,
        "Results: Brake Thermodynamics Model", "2.5–3")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "This is perhaps the most novel contribution of our work, "
        "because no published paper on pedestrian AEB includes brake thermodynamics. "
        "Every existing system assumes friction is constant regardless of how many "
        "times you have braked.")
    add_note(doc,
        "The left plot simulates six panic stops from 80 km/h, "
        "spaced roughly 60 seconds apart. "
        "After the sixth stop, the rotor temperature peaks at 428 degrees Celsius — "
        "above the onset threshold of 300 degrees and approaching the severe fade "
        "threshold of 450 degrees.")
    add_note(doc,
        "The thermal model has two equations. "
        "Heating: delta-T equals one-half m-v-squared divided by m-rotor times c-p. "
        "This is just kinetic energy converted to heat in the rotor. "
        "We use a rotor mass of 40 kilograms and specific heat capacity of "
        "450 joules per kilogram per Kelvin — standard cast iron values.")
    add_note(doc,
        "Cooling: T of t equals T-environment plus delta-T times e to the minus k-t. "
        "Newton's cooling law. The cooling constant k increases with vehicle speed "
        "because airflow over the disc is proportional to velocity. "
        "At 80 km/h cruise, the rotor cools back below 150 degrees in about 10 minutes.")
    add_note(doc,
        "The right plot is the critical output. "
        "The blue line is friction multiplier — it equals 1.0 below 300 degrees "
        "(full friction), drops linearly to 0.7 at 450 degrees, "
        "continues to 0.3 at 650 degrees, and beyond that braking is almost ineffective.")
    add_note(doc,
        "The red dashed line is the AEB TTC penalty — additional trigger lead time "
        "we add to compensate for reduced friction. "
        "The formula is: penalty equals minimum of 3.0 and 0.5 times "
        "one over mu minus one. "
        "At 428 degrees, mu equals 0.72, so penalty is about 0.19 seconds. "
        "At full fade — 650 degrees and above — the penalty caps at 3.0 seconds, "
        "meaning the AEB fires three full seconds earlier than its baseline trigger. "
        "This cap prevents the threshold from becoming unreasonably large "
        "in extreme conditions.")
    add_note(doc,
        "The practical implication: a vehicle running repeated emergency stops "
        "in a busy urban scenario — think a fire engine in an emergency corridor — "
        "will have significantly degraded braking after just six stops. "
        "Our system detects this in real time and automatically compensates. "
        "A kinematic AEB would still trigger at 1.5 seconds and underbrake.")

    add_emphasis_box(doc,
        "6 panic stops (80 km/h) → 428°C rotor\n"
        "At 428°C: friction −28%  |  AEB TTC penalty +0.19 s\n"
        "At ≥650°C: friction −70%  |  AEB penalty caps at +3.0 s\n"
        "Cooling to safe <150°C: ~10 minutes at 80 km/h cruise")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "This is the first pedestrian AEB model to account for brake fade.")
    add_note(doc, "Penalty feeds directly into the adaptive threshold — closed-loop physics.")
    add_note(doc, "The 3.0 s cap is a safety design choice — prevents pathological edge cases.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: How do you measure rotor temperature in a real vehicle?  "
        "A: Infrared sensors or embedded thermocouple rings on the rotor "
        "are standard in motorsport and are commercially available for production vehicles. "
        "Our model is designed to consume that telemetry as a direct input.")
    add_note(doc,
        "Q: Is the fade curve empirical or derived?  "
        "A: The three-regime piecewise curve is based on published "
        "brake pad material datasheets (organic and semi-metallic compounds). "
        "The exact numbers vary by pad composition; our curve represents "
        "a conservative mid-range estimate.")

    add_transition(doc,
        "\"Now we have the physics. The third intelligence layer is: "
        "who is driving, how alert are they, and what is the environment? Slide 7.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 7 — ADAPTIVE AEB & DRIVER MONITORING
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 7, 9,
        "Results: Adaptive AEB and Driver Monitoring", "3")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "This slide brings together the three correction modules — weather, driver, "
        "and thermal — into the adaptive AEB threshold, and introduces the "
        "Bird's-Eye View radar output.")
    add_note(doc,
        "Start with the stacked bar chart on the upper left. "
        "Each bar represents a driving condition, from clear dry roads on the left "
        "to the worst-case fog plus drowsy driver plus thermally faded brakes on the right. "
        "The teal base layer is always 1.5 seconds — the minimum safe TTC "
        "for a 50 km/h approach on dry roads with a fresh brake system. "
        "The orange layer adds weather delta: 0.0 seconds for clear, "
        "0.3 seconds for rain, 0.5 seconds for fog. "
        "The red layer adds driver drowsiness delta: 0.5 seconds when EAR drops "
        "below 0.25 for more than 2 consecutive seconds. "
        "The purple layer adds the thermal penalty from the previous slide. "
        "The worst-case total — all three deltas maxed out — is 3.5 seconds.")
    add_note(doc,
        "The EAR panel on the upper right shows the Eye Aspect Ratio time series. "
        "EAR is computed from dlib's 68-point facial landmark model. "
        "It is defined as the ratio of vertical eye opening to horizontal eye width. "
        "When both eyes are open normally, EAR sits around 0.30 to 0.35. "
        "During a blink, it drops to near zero for 100 to 200 milliseconds "
        "and recovers — you can see the two sharp dips at 5 and 15 seconds. "
        "When the driver is drowsy, EAR stays depressed below 0.25 for an extended "
        "period — you can see this in the red-shaded region around 20 to 23 seconds. "
        "That sustained depression triggers the DROWSY flag and adds 0.5 seconds "
        "to the AEB threshold.")
    add_note(doc,
        "The bottom panel is our Bird's-Eye View radar — the real-time ADAS "
        "visualisation output. It is generated purely from the monocular camera "
        "using two depth equations. "
        "The longitudinal depth Z equals focal length times known real-world object height "
        "divided by pixel height of the bounding box. "
        "The lateral position X equals pixel x-coordinate minus principal point "
        "times Z divided by focal length. "
        "These give us 3D world coordinates from a single camera with no stereo rig.")
    add_note(doc,
        "On the radar, the display is a top-down view. "
        "Ego vehicle is the blue rectangle at the left. "
        "Depth increases to the right. "
        "Three colour zones: red for critical — less than 9 metres or TTC below 2 seconds; "
        "yellow for warning — 9 to 20 metres; green for safe — beyond 20 metres. "
        "In the example shown, pedestrian P3 at 7 metres has TTC 0.9 seconds — "
        "AEB is triggered. P2 at 15 metres is in the warning zone. "
        "P1 at 25 metres is safe.")

    add_emphasis_box(doc,
        "Threshold formula: 1.5 + Δ_weather + Δ_driver + Δ_thermal\n"
        "Worst case: 1.5 + 0.5 + 0.5 + 1.0 = 3.5 s  (+133% over baseline)\n"
        "BEV depth:  Z = f·H_real / H_px   |   X = (c_x − c_x0)·Z / f\n"
        "EAR < 0.25 sustained >2 s → DROWSY flag → +0.5 s threshold")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "133% threshold increase in worst case — that margin is the safety net.")
    add_note(doc, "BEV from a single camera — no additional hardware required.")
    add_note(doc, "EAR drowsiness is real-time, 68-landmark, frame-by-frame.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: How accurate is monocular depth estimation?  "
        "A: The pinhole formula is accurate when the object size is known — "
        "for adult pedestrians at 1.7 m average height, error is under 10% "
        "within 30 metres. Beyond that, pixel height uncertainty grows and "
        "we recommend stereo or radar fusion (listed as future work).")
    add_note(doc,
        "Q: What weather classifier do you use?  "
        "A: A ResNet-18 fine-tuned on BDD100K weather labels, with a "
        "heuristic fallback based on frame brightness variance and blur. "
        "The classifier targets 95% accuracy — improving it is future work.")

    add_transition(doc,
        "\"All of this — detection, physics, adaptation — is only valid if the "
        "vehicle actually stops before impact. Slide 8 tests that against the "
        "Euro NCAP standard and then measures injury severity if impact does occur.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 8 — EURO NCAP & HIC
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 8, 9,
        "Results: Euro NCAP Validation and Impact Biomechanics", "3")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "Slide 8 is our external validation and the answer to: "
        "'does your system actually save lives?' "
        "The left panel is the Euro NCAP pass/fail matrix. "
        "The right panel is the HIC injury severity curve.")
    add_note(doc,
        "Euro NCAP defines four standardised pedestrian crossing scenarios. "
        "CPFA — Child or Adult crossing Far-Side — the pedestrian emerges from the "
        "far side of traffic. "
        "CPNA — Near-Side Adult — approaches from the near side. "
        "CPNC — Child behind a Near-side parked Car — the most difficult scenario "
        "because the child is occluded until very close range. "
        "CPLA — longitudinal — the pedestrian is walking in the vehicle's lane. "
        "We test each scenario at five speeds: 20, 30, 40, 50, and 60 km/h, "
        "giving a 20-combination test matrix.")
    add_note(doc,
        "Reading the matrix: green equals PASS — the AEB stopped the vehicle "
        "before contact. Red equals FAIL. "
        "At 20 and 30 km/h, all four scenarios pass — 100% pass rate. "
        "At 40 km/h, CPNC fails because the child emerges from behind a parked car "
        "with only 20 metres of detection range — not enough time to stop "
        "from 40 km/h even with maximum braking. "
        "This is the binding constraint, and it is consistent with "
        "published limitations of monocular AEB systems.")
    add_note(doc,
        "CPLA — the longitudinal scenario — passes at all speeds including 60 km/h, "
        "because the pedestrian is in the camera's direct line of sight "
        "from maximum detection range, giving the system full stopping time.")
    add_note(doc,
        "Now the right panel — the HIC curve. "
        "HIC stands for Head Injury Criterion. "
        "It was developed from biomechanical cadaver studies and validated "
        "against real accident data. "
        "The formula is: HIC equals contact time times peak head acceleration "
        "divided by g, raised to the power 2.5. "
        "Contact time is derived from the mechanical spring-mass model of "
        "the skull-bonnet impact.")
    add_note(doc,
        "The horizontal bands show the AIS severity scale: "
        "AIS 1 is Minor — bruising; "
        "AIS 2 is Moderate — mild concussion; "
        "AIS 3 is Serious — skull fracture; "
        "AIS 5 is Critical — life-threatening brain injury; "
        "AIS 6 is Fatal. "
        "The blue curve shows HIC as a function of impact speed. "
        "At 20 km/h, HIC is about 45 — AIS 1, minor injury. "
        "At 50 km/h without AEB, HIC reaches approximately 2,200 — that is AIS 5, Critical.")
    add_note(doc,
        "Now the key result: if AEB reduces impact speed from 50 to 30 km/h — "
        "a partial braking event, not full avoidance — HIC drops from 2,200 to 280. "
        "That is an eight-times reduction. "
        "Injury severity drops from AIS 5 Critical to AIS 2 Moderate. "
        "The pedestrian likely survives a life-threatening injury. "
        "Even when the system cannot fully stop the vehicle, partial braking saves lives.")

    add_emphasis_box(doc,
        "100% NCAP pass rate at 20–30 km/h  |  CPNC (child, parked car) = binding constraint\n"
        "HIC at 50 km/h without AEB: ~2,200  (AIS 5 — Critical)\n"
        "HIC at 30 km/h with AEB:      ~280   (AIS 2 — Moderate)\n"
        "8× HIC reduction  |  AIS 5 → AIS 2  — from life-threatening to moderate injury")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "8× HIC reduction is the human cost headline — lead with it.")
    add_note(doc, "CPNC is the binding constraint — be honest about this limitation.")
    add_note(doc, "Partial braking saves lives even when full avoidance fails.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: How do you compute AIS from HIC?  "
        "A: We use the Abbreviated Injury Scale lookup table with standard HIC thresholds: "
        "HIC < 150 = AIS 1, 150–500 = AIS 2, 500–1000 = AIS 3, "
        "1000–1500 = AIS 4, 1500–2500 = AIS 5, > 2500 = AIS 6.")
    add_note(doc,
        "Q: Does the CPNC failure invalidate the system?  "
        "A: No — it identifies the sensor limitation. "
        "At 40 km/h, even the best radar-equipped commercial AEB systems "
        "have limited CPNC success because detection range from behind a parked car "
        "is physically constrained to 15–20 metres. "
        "Stereo fusion or short-range radar is the engineering solution, "
        "which is our next-step future work.")

    add_transition(doc,
        "\"Let me now summarise what we've achieved and where we go next — Slide 9.\"")

    hr(doc)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 9 — CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════════════
    add_slide_header(doc, 9, 9,
        "Conclusions and Future Work", "2.5–3")

    add_section(doc, "TALKING POINTS")
    add_note(doc,
        "Six conclusions, four future directions. "
        "I'll read each conclusion and then explain why it matters beyond this project.")
    add_note(doc,
        "Conclusion 1: Physics Validated. "
        "The Pacejka plus ABS model matches published stopping distance references "
        "within 5% across six surface-speed combinations. "
        "The simple kinematic formula undershoots by 15 to 30 percent. "
        "For AEB calibration, 15% is not a rounding error — it is the difference "
        "between braking in time and not braking in time.")
    add_note(doc,
        "Conclusion 2: Brake Fade Captured. "
        "After six panic stops, rotor temperature reaches 428 degrees and friction "
        "drops 28 percent. "
        "Our closed-loop compensation adds 0.19 seconds of additional lead time automatically. "
        "This is a new capability not present in any published pedestrian AEB paper we found.")
    add_note(doc,
        "Conclusion 3: Euro NCAP Compliance. "
        "100 percent pass rate at 20 to 30 km/h across all four scenarios. "
        "The child-behind-parked-car scenario is the binding constraint at higher speeds, "
        "consistent with monocular camera sensor limitations across the industry.")
    add_note(doc,
        "Conclusion 4: Adaptive AEB Threshold. "
        "The composite TTC threshold scales from 1.5 to 3.5 seconds — a 133 percent "
        "increase in the worst case. "
        "This ensures the system reacts earlier when it knows conditions are degraded, "
        "rather than waiting for a fixed time-to-contact that was calibrated for ideal conditions.")
    add_note(doc,
        "Conclusion 5: Injury Reduction 8 times. "
        "Even when full avoidance fails — as in CPNC at 40 km/h — "
        "partial braking from 50 to 30 km/h reduces HIC from 2,200 to 280. "
        "Injury severity drops from life-threatening to moderate. "
        "This reframes what 'success' means for an AEB system: "
        "it does not need to prevent all collisions to save lives.")
    add_note(doc,
        "Conclusion 6: Real-Time BEV Radar. "
        "A single monocular camera generates a live, colour-coded, top-down "
        "spatial map of all pedestrians within range, at 15 to 20 frames per second. "
        "This is the standard ADAS display format used by production vehicles "
        "with dedicated radar hardware. "
        "We reproduce it from a camera alone using the pinhole depth equations.")
    add_note(doc,
        "The key formula strip at the bottom summarises the mathematical core "
        "of the entire system in one line: "
        "stopping distance, TTC, rotor heating, HIC, and the adaptive threshold.")
    add_note(doc,
        "Four future work items: "
        "First, stereo or short-range radar fusion to extend CPNC detection range "
        "and achieve 100% NCAP compliance at 40 km/h. "
        "Second, a proper ResNet-18 weather classifier to replace the heuristic — "
        "we target 95% classification accuracy. "
        "Third, deployment on NVIDIA Jetson Orin for on-vehicle testing at 30 fps "
        "in real mixed traffic. "
        "Fourth, Kalman-filtered pedestrian velocity estimation for more accurate TTC "
        "beyond the constant-speed assumption we currently use.")
    add_note(doc,
        "To close: we set out to build an AEB system grounded in physics rather "
        "than approximations. "
        "We achieved that across detection, braking mechanics, thermodynamics, "
        "driver state, and injury modelling — all integrated in a single Python pipeline. "
        "The results validate against international standards and show "
        "meaningful safety gains even in scenarios where full avoidance is not possible. "
        "Thank you.")

    add_emphasis_box(doc,
        "FINAL SUMMARY — five numbers to remember:\n"
        "  78.3%  — pedestrian detection mAP@0.5 (target: ≥75%)\n"
        "  428°C  — rotor temperature after 6 panic stops\n"
        "    3.5 s — adaptive TTC threshold in worst case (+133%)\n"
        "     8×   — HIC reduction via partial braking (AIS 5 → AIS 2)\n"
        "  100%   — Euro NCAP pass rate at 20–30 km/h")

    add_section(doc, "KEY EMPHASES", 'C00000')
    add_note(doc, "The 8× HIC reduction is the single most compelling number — end on it.")
    add_note(doc, "Physics-aware ≠ kinematic — return to the opening thesis.")
    add_note(doc,
        "Partial braking saves lives — this is the philosophical reframe "
        "that the audience should leave with.")

    add_section(doc, "ANTICIPATE", '336633')
    add_note(doc,
        "Q: What is the computational cost of the full pipeline?  "
        "A: The perception stack (YOLO + SORT + MiDaS) runs at 15–20 fps on a Tesla T4. "
        "The physics modules (thermodynamics + Pacejka integrator) run at sub-millisecond "
        "per frame on CPU. Total system latency is dominated by inference.")
    add_note(doc,
        "Q: Can this be productised?  "
        "A: The algorithm is hardware-agnostic. "
        "The Jetson Orin future-work item is the productisation path. "
        "Key gaps: real brake temperature sensors, validated weather training set, "
        "and regulatory certification.")
    add_note(doc,
        "Q: How do you handle multiple pedestrians simultaneously?  "
        "A: SORT maintains independent tracks for each pedestrian. "
        "TTC is computed per track. "
        "AEB is triggered on the minimum TTC across all in-corridor tracks — "
        "the closest threat always wins.")

    # ── Final note ─────────────────────────────────────────────────────────────
    hr(doc)
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run(
        "END OF SPEAKER NOTES\n"
        "Total estimated delivery: 20–25 minutes at a comfortable academic pace.\n"
        "Q&A: 5–10 minutes. "
        "If time is short, compress Slides 3 (Timeline) and 6 (Thermodynamics) first.")
    run.font.size   = Pt(10)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(OUTPUT)
    print(f"Speaker notes saved → {OUTPUT}")
    print(f"Slides covered: 9  |  Sections per slide: 4")
    print(f"Total talking points: comprehensive, slide-by-slide")


if __name__ == "__main__":
    main()
