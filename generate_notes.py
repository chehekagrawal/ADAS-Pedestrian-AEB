"""
Generates a speaker notes Word document for the ADAS Pedestrian AEB
end-term presentation (9 slides, ~9–10 minutes total speaking time).
"""

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUTPUT = "reports/ADAS_AEB_Speaker_Notes.docx"

TEAL  = RGBColor(0x00, 0x70, 0x70)
BLACK = RGBColor(0x00, 0x00, 0x00)
GRAY  = RGBColor(0x55, 0x55, 0x55)


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = TEAL
        run.font.size = Pt(14 if level == 1 else 12)
    return p


def para(doc, text, italic=False, color=BLACK, size=11):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.color.rgb = color
    p.paragraph_format.space_after = Pt(6)
    return p


def time_tag(doc, seconds):
    m, s = divmod(seconds, 60)
    label = f"[~{m}m {s:02d}s]" if m else f"[~{s}s]"
    p = doc.add_paragraph()
    r = p.add_run(label)
    r.font.size = Pt(9)
    r.font.italic = True
    r.font.color.rgb = GRAY
    p.paragraph_format.space_after = Pt(2)


def divider(doc):
    doc.add_paragraph("─" * 70)


# ── Notes content ──────────────────────────────────────────────────

NOTES = [

    # ── Slide 1 ─────────────────────────────────────────────────────
    {
        "slide":   1,
        "title":   "Title Slide",
        "seconds": 35,
        "blocks": [
            (
                "Good [morning/afternoon], everyone. My name is Atharv Priyadarshi, "
                "and on behalf of our team I will be presenting our end-term project: "
                "a Physics-Aware Pedestrian Automatic Emergency Braking system — or "
                "AEB — developed as part of Course MIT-115 in the Department of "
                "Mechanical and Industrial Engineering here at IIT Roorkee."
            ),
            (
                "Commercial AEB systems typically rely on a simplified kinematic model "
                "that assumes a constant deceleration value. Our project replaces that "
                "single number with a full physics engine that models tire forces, "
                "brake fade, weather, and driver state — resulting in a system that "
                "adjusts its trigger timing in real time to match actual braking "
                "capability."
            ),
        ],
    },

    # ── Slide 2 ─────────────────────────────────────────────────────
    {
        "slide":   2,
        "title":   "Research Objectives and Methodology",
        "seconds": 95,
        "blocks": [
            (
                "This slide summarises the five core objectives of the project, "
                "alongside the 14-step per-frame processing pipeline shown in the "
                "flowchart on the right. I will walk through the objectives first "
                "and then briefly explain the pipeline structure."
            ),
            (
                "Objective 1 targets pedestrian detection accuracy: we fine-tuned "
                "YOLOv8n on 6,000 BDD100K images and set a design goal of mAP@0.5 "
                "at or above 75% for the pedestrian class — a threshold consistent "
                "with the Euro NCAP sensor requirements. "
                "Objective 2 replaces the standard d = v²/2a stopping-distance "
                "formula with a full Pacejka Magic Formula tire model run at a 1 ms "
                "timestep, including dynamic weight transfer and ABS pressure "
                "modulation cycles. "
                "Objective 3 adds a brake thermodynamics layer: we model rotor "
                "heat-up after each braking event and the subsequent Newton's-Law "
                "cooling, then continuously compensate the AEB trigger threshold for "
                "the resulting friction reduction."
            ),
            (
                "Objectives 4 and 5 bring everything together. The adaptive AEB "
                "controller combines a weather penalty, a driver-alertness penalty "
                "from eye-aspect-ratio tracking, and the thermal penalty into a "
                "single composite TTC threshold. Finally, we validate the system "
                "against Euro NCAP's four pedestrian test scenarios and quantify "
                "injury severity using the Head Injury Criterion biomechanical model. "
                "The pipeline on the right shows how each of these 14 processing "
                "steps flows from raw frame input through to the annotated output — "
                "the left column handles perception and the right column handles "
                "decision-making and output."
            ),
        ],
    },

    # ── Slide 3 ─────────────────────────────────────────────────────
    {
        "slide":   3,
        "title":   "Project Timeline",
        "seconds": 40,
        "blocks": [
            (
                "This Gantt chart shows the project distributed across roughly six "
                "months from January to the current week. The first month was devoted "
                "to dataset preparation — converting BDD100K annotations to YOLO "
                "format and cleaning labels. February saw the YOLOv8 training run on "
                "a Tesla T4 GPU, followed immediately by the SORT tracker and MiDaS "
                "depth integration."
            ),
            (
                "The vehicle dynamics module and the weather classifier were developed "
                "in parallel during March and April. The brake thermodynamics model, "
                "driver monitoring, and the adaptive AEB controller were completed by "
                "mid-May. HIC biomechanics analysis and the CARLA simulator "
                "integration are the most recent additions, represented by the bars "
                "closest to the gold 'NOW' marker. Report writing and this "
                "presentation account for the final two weeks."
            ),
        ],
    },

    # ── Slide 4 ─────────────────────────────────────────────────────
    {
        "slide":   4,
        "title":   "Results — Object Detection and Tracking",
        "seconds": 65,
        "blocks": [
            (
                "The left chart compares mAP@0.5 and mAP@0.5:0.95 across all four "
                "detection classes. The pedestrian class achieves mAP@0.5 of 78.3%, "
                "which exceeds our 75% design target. The overall mAP@0.5 is 77.1% "
                "and mAP@0.5:0.95 is 44.8%. The harder vehicle classes — car at "
                "89.1% — benefit from their larger bounding boxes, while motorcycles "
                "at 69.8% are the most challenging due to partial occlusion and "
                "viewpoint variation."
            ),
            (
                "The right chart shows training convergence over 30 epochs. Both box "
                "loss and classification loss plateau around epoch 22, confirming that "
                "additional epochs would not significantly improve accuracy without "
                "additional data. We used a confidence threshold of 0.30 at inference "
                "time, which provides the best precision-recall trade-off for "
                "pedestrians specifically."
            ),
            (
                "For multi-object tracking, SORT maintains a 7-dimensional Kalman "
                "state per object — position, scale, aspect ratio, and their "
                "derivatives — and uses the Hungarian algorithm with an IoU threshold "
                "of 0.30 to associate detections across frames. The min_hits=3 "
                "parameter means a track is not reported until confirmed in three "
                "consecutive frames, which eliminates most false-positive tracks."
            ),
        ],
    },

    # ── Slide 5 ─────────────────────────────────────────────────────
    {
        "slide":   5,
        "title":   "Results — Vehicle Dynamics and ABS Simulation",
        "seconds": 65,
        "blocks": [
            (
                "This slide presents the braking distance results from our full "
                "longitudinal dynamics simulator. The stacked bar chart shows stopping "
                "distances at 50 km/h and 100 km/h across all five road surfaces. "
                "On dry tarmac, 50 km/h gives 16.2 metres — well within the "
                "engineering reference range of 13 to 22 metres. At 100 km/h the "
                "result is 57.8 metres against a reference of 45 to 75 metres."
            ),
            (
                "The critical insight here is the ice-surface result: 145.3 metres at "
                "50 km/h, compared to just 16.2 metres on dry road — a factor of "
                "nine. This variation is completely invisible to any AEB system that "
                "uses a fixed deceleration value. Our physics model captures it "
                "automatically because the Pacejka Magic Formula evaluates friction "
                "coefficient as a function of slip ratio and surface type at every "
                "timestep."
            ),
            (
                "The five forces computed at each 1-millisecond step are: Pacejka "
                "tire braking force, dynamic weight transfer which redistributes load "
                "to the front axle under braking, ABS pressure modulation in 0.1-"
                "second cycles to prevent lockup, aerodynamic drag proportional to "
                "velocity squared, and grade force from the road slope. Together "
                "these produce stopping distances that are 15 to 30% longer than the "
                "simple kinematic formula — meaning commercial systems using that "
                "formula are systematically triggering too late."
            ),
        ],
    },

    # ── Slide 6 ─────────────────────────────────────────────────────
    {
        "slide":   6,
        "title":   "Results — Brake Thermodynamics Model",
        "seconds": 65,
        "blocks": [
            (
                "The left chart simulates a severe scenario: six consecutive panic "
                "stops at 80 km/h. Each braking event deposits kinetic energy into "
                "the rotor according to ΔT = ½mv² divided by rotor mass times "
                "specific heat capacity. For a 1,500 kg vehicle the energy per stop "
                "is approximately 370 kilojoules. After the sixth stop the rotor "
                "temperature reaches a peak of 428 degrees Celsius, crossing the "
                "orange fade-onset line at 300 degrees on the third stop."
            ),
            (
                "The right chart shows the consequence for friction and AEB timing. "
                "Below 300°C the friction multiplier stays at 1.0. Between 300 and "
                "450°C it drops linearly to 0.7, and between 450 and 650°C it falls "
                "further to 0.3. At 428°C the multiplier is approximately 0.72, "
                "meaning the brakes produce only 72% of their nominal force. Our "
                "AEB penalty formula — minimum of 3.0 seconds or 0.5 times the "
                "quantity one-over-mu minus one — adds 0.19 seconds of additional "
                "lead time at that temperature."
            ),
            (
                "This compensation mechanism is absent from every commercial kinematic "
                "AEB system we reviewed. The maximum possible penalty is capped at "
                "3.0 seconds, which corresponds to near-complete brake fade — a "
                "condition that would also trigger a dashboard warning independently. "
                "The model runs in real time as a background thread, updating the "
                "penalty continuously between braking events."
            ),
        ],
    },

    # ── Slide 7 ─────────────────────────────────────────────────────
    {
        "slide":   7,
        "title":   "Results — Adaptive AEB, Driver Monitoring, and BEV Radar",
        "seconds": 75,
        "blocks": [
            (
                "The top chart shows how the composite TTC trigger threshold is built "
                "up for seven representative operating scenarios. Every scenario starts "
                "from the 1.5-second base — the minimum required to compute TTC, "
                "command braking, and overcome brake actuator lag. Weather, driver "
                "state, and thermal penalties are then stacked on top. The worst-case "
                "scenario — fog, drowsy driver, and severely faded brakes — reaches "
                "3.5 seconds, which is 133% above the baseline. Driver monitoring "
                "uses dlib's 68-point face landmark detector. An Eye Aspect Ratio "
                "below 0.25 sustained for more than two seconds classifies the driver "
                "as drowsy and adds 0.5 seconds to the threshold."
            ),
            (
                "The bottom panel is the Bird's-Eye View Radar split-screen — one of "
                "the more visually distinctive outputs of this project. The left half "
                "shows the actual dashcam frame from a crowded urban scene, annotated "
                "in real time by the YOLO detector and SORT tracker. The right half "
                "is the top-down radar canvas. Each tracked object is projected from "
                "the 2D bounding box into a metric 3D position using pinhole depth "
                "estimation — depth equals focal length times pedestrian height "
                "divided by bounding box height in pixels."
            ),
            (
                "The four pedestrians in this frame demonstrate the full risk spectrum: "
                "the closest pedestrian at 8 metres has a TTC of 1.5 seconds and "
                "appears as a red heat blob with the AEB danger cone pointing straight "
                "ahead. The two middle pedestrians at 13 and 15 metres are in the "
                "WARNING zone, shown in yellow. The furthest pedestrian at 23 metres "
                "is fully safe — a single green dot with no heat contribution. This "
                "BEV display is generated entirely from the monocular camera with no "
                "LiDAR, stereo, or radar input — demonstrating that depth-aware "
                "spatial reasoning is achievable with a single camera at the cost "
                "of accepting some depth estimation uncertainty."
            ),
        ],
    },

    # ── Slide 8 ─────────────────────────────────────────────────────
    {
        "slide":   8,
        "title":   "Results — Euro NCAP Validation and Impact Biomechanics",
        "seconds": 70,
        "blocks": [
            (
                "The heatmap on the left summarises pass-fail results across Euro "
                "NCAP's four pedestrian AEB test scenarios at five impact speeds. "
                "The easiest scenario is CPLA — Longitudinal Adult — where the "
                "pedestrian walks directly into the vehicle's path. Our system passes "
                "this at all speeds up to 60 km/h because the pedestrian is in the "
                "ego corridor for a longer time, giving the system more warning."
            ),
            (
                "The binding constraint is CPNC — the child emerging from behind a "
                "parked car — where we only pass at 20 and 30 km/h. This is "
                "consistent with published literature on monocular-camera AEB: the "
                "pedestrian is occluded until approximately 20 metres away, which "
                "at 40 km/h leaves only 1.8 seconds of reaction time before impact "
                "— less than the 2.0-second threshold required under rain and wet "
                "conditions. Adding a stereo camera or LiDAR would extend the "
                "detection range and likely shift the CPNC pass boundary to 40 km/h."
            ),
            (
                "The right chart puts these results in biomechanical context. The HIC "
                "value at 20 km/h is approximately 45, which is AIS grade 1 — minor "
                "injury. At 40 km/h HIC reaches 900, which is AIS grade 3, or "
                "serious injury. The most important number is the partial-brake "
                "result: if our AEB reduces impact speed from 50 to 30 km/h, HIC "
                "drops from approximately 2,200 down to 280 — an eight-fold "
                "reduction — and AIS severity drops from Critical, grade 5, to "
                "Moderate, grade 2. This quantifies the life-saving value even when "
                "the system cannot achieve complete collision avoidance."
            ),
        ],
    },

    # ── Slide 9 ─────────────────────────────────────────────────────
    {
        "slide":   9,
        "title":   "Conclusions",
        "seconds": 70,
        "blocks": [
            (
                "To summarise: we have demonstrated that replacing the kinematic "
                "stopping-distance formula with a full physics model — Pacejka tire "
                "forces, dynamic weight transfer, and ABS modulation — gives "
                "stopping distances 15 to 30% longer than the simple formula across "
                "all five road surfaces and both test speeds. The physics simulator "
                "passes all reference-range validation checks."
            ),
            (
                "The brake thermodynamics model is a novel contribution: we show that "
                "six consecutive panic stops at 80 km/h raise rotor temperature above "
                "400 degrees Celsius, reducing friction by 28% and requiring up to "
                "3 additional seconds of AEB lead time. This factor is absent from "
                "every commercial kinematic system we examined. The adaptive threshold "
                "framework successfully adjusts the TTC trigger from 1.5 seconds in "
                "ideal conditions to 3.5 seconds in the worst-case multi-factor "
                "scenario, maintaining correct timing across all tested conditions."
            ),
            (
                "Euro NCAP results confirm 100% compliance at 20 to 30 km/h across "
                "all four scenarios. The HIC analysis provides a compelling safety "
                "argument: even partial intervention — slowing from 50 to 30 km/h "
                "— reduces head injury severity by a factor of eight. Future work "
                "will focus on fusing a short-range radar or LiDAR to extend the "
                "CPNC detection range, replacing the heuristic weather classifier "
                "with the fine-tuned ResNet-18, and deploying the full pipeline on "
                "an NVIDIA Jetson Orin for embedded real-time validation. Thank you "
                "— I am happy to take any questions."
            ),
        ],
    },
]


def build_notes_doc():
    doc = Document()

    # ── Page setup ──
    section = doc.sections[0]
    section.page_width  = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

    # ── Cover ──
    t = doc.add_heading("ADAS Pedestrian AEB System", level=0)
    for r in t.runs:
        r.font.color.rgb = TEAL
        r.font.size = Pt(18)

    p = doc.add_paragraph("Speaker Notes — End-Term Presentation")
    for r in p.runs:
        r.font.size = Pt(13)
        r.font.color.rgb = GRAY

    doc.add_paragraph(
        "IIT Roorkee  ·  Department of Mechanical & Industrial Engineering  ·  "
        "Course MIT-115  ·  May 2026"
    ).runs[0].font.size = Pt(10)

    doc.add_paragraph(
        "Team: Atharv Priyadarshi (23117034)   [+ team members]"
    ).runs[0].font.size = Pt(10)

    doc.add_paragraph(
        "Total estimated speaking time: ~9 minutes 15 seconds"
    ).runs[0].font.italic = True

    doc.add_page_break()

    # ── Slide sections ──
    for note in NOTES:
        heading(doc, f"Slide {note['slide']} — {note['title']}", level=1)
        time_tag(doc, note["seconds"])
        for block in note["blocks"]:
            para(doc, block)
        divider(doc)
        doc.add_paragraph()

    doc.save(OUTPUT)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build_notes_doc()
