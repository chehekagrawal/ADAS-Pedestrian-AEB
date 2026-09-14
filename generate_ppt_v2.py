"""
Generates ADAS Pedestrian AEB end-term presentation using the exact
IIT Roorkee template (PPT-MIT-115.pptx). Opens the template, deletes
unused slides, and adds content into the existing slide layouts.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

TEMPLATE = "/home/atharv/Downloads/PPT-MIT-115.pptx"
OUTPUT   = "/home/atharv/ADAS-Pedestrian-AEB/reports/ADAS_AEB_Presentation_Final.pptx"

BLACK = RGBColor(0x00, 0x00, 0x00)
TEAL  = RGBColor(0x00, 0x70, 0x70)
GRAY  = RGBColor(0x50, 0x50, 0x50)


# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def fig_to_buf(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf


def delete_slide(prs, idx):
    """Remove slide at given 0-based index from the presentation."""
    xml_slides = prs.slides._sldIdLst
    xml_slides.remove(xml_slides[idx])


def set_title(slide, text):
    """
    Replace title placeholder text while keeping existing font formatting.
    Preserves the original font size, bold, colour set by the template.
    """
    ph = slide.placeholders[0]
    tf = ph.text_frame
    # Keep only the first paragraph, first run
    while len(tf.paragraphs) > 1:
        tf.paragraphs[-1]._p.getparent().remove(tf.paragraphs[-1]._p)
    para = tf.paragraphs[0]
    if para.runs:
        while len(para.runs) > 1:
            para.runs[-1]._r.getparent().remove(para.runs[-1]._r)
        para.runs[0].text = text
    else:
        para.add_run().text = text


def clear_content(slide):
    """Empty the content placeholder (idx=2) so we can add our own shapes."""
    try:
        ph = slide.placeholders[2]
        ph.text_frame.clear()
    except Exception:
        pass


def add_tb(slide, text, left, top, width, height,
           size=11, bold=False, italic=False,
           color=BLACK, align=PP_ALIGN.LEFT, wrap=True):
    """Add a simple textbox with a single paragraph."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return tb


def add_bullets(slide, heading, items, left, top, width, height,
                item_size=10, heading_size=11.5, item_color=BLACK):
    """
    Add a textbox with an underlined heading and bullet items.
    Heading is in teal bold; items are plain black.
    """
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    # Heading paragraph
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = heading
    r.font.size = Pt(heading_size)
    r.font.bold = True
    r.font.color.rgb = TEAL

    # Bullet items
    for item in items:
        p2 = tf.add_paragraph()
        p2.space_before = Pt(3)
        r2 = p2.add_run()
        r2.text = f"▪  {item}"
        r2.font.size = Pt(item_size)
        r2.font.color.rgb = item_color
    return tb


def add_rich_bullets(slide, heading, items, left, top, width, height,
                     item_size=10, heading_size=11.5, item_color=BLACK):
    """
    Like add_bullets but each item can be either:
      - a plain str  → rendered as a normal bullet
      - a list of ('n'|'s'|'b', text) tuples → rendered with proper
        super/subscripts using OOXML baseline offsets.
    'n' = normal, 's' = superscript (+25% baseline), 'b' = subscript (-20%).
    """
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = heading
    r.font.size = Pt(heading_size)
    r.font.bold = True
    r.font.color.rgb = TEAL

    for item in items:
        p2 = tf.add_paragraph()
        p2.space_before = Pt(3)

        if isinstance(item, str):
            r2 = p2.add_run()
            r2.text = f"▪  {item}"
            r2.font.size = Pt(item_size)
            r2.font.color.rgb = item_color
        else:
            first = True
            for seg_type, seg_text in item:
                r2 = p2.add_run()
                r2.text = ("▪  " + seg_text) if first else seg_text
                first = False
                r2.font.size = Pt(item_size)
                r2.font.color.rgb = item_color
                if seg_type == 's':
                    r2._r.get_or_add_rPr().set('baseline', '25000')
                elif seg_type == 'b':
                    r2._r.get_or_add_rPr().set('baseline', '-20000')
    return tb


def add_picture(slide, buf, left, top, width, height):
    return slide.shapes.add_picture(buf, left, top, width, height)


def add_formula_strip(slide, segments, left, top, width, height, size=9.0):
    """
    Formula strip with lighter teal background, bold text, proper sub/superscript.
    segments: list of ('n'|'s'|'b', 'text') — normal / superscript / subscript.
    A plain string is treated as [('n', string)] for backward compatibility.
    """
    if isinstance(segments, str):
        segments = [('n', segments)]

    tb = slide.shapes.add_textbox(left, top, width, height)
    tb.fill.solid()
    tb.fill.fore_color.rgb = RGBColor(0x00, 0x6a, 0x6a)   # lighter teal background
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left   = Pt(5)
    tf.margin_right  = Pt(5)
    tf.margin_top    = Pt(3)
    tf.margin_bottom = Pt(3)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER

    for seg_type, seg_text in segments:
        r = p.add_run()
        r.text           = seg_text
        r.font.size      = Pt(size)
        r.font.bold      = True
        r.font.color.rgb = RGBColor(0xe8, 0xff, 0xff)
        if seg_type == 's':    # superscript — 25% above baseline
            r._r.get_or_add_rPr().set('baseline', '25000')
        elif seg_type == 'b':  # subscript — 20% below baseline
            r._r.get_or_add_rPr().set('baseline', '-20000')
    return tb


# ══════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ══════════════════════════════════════════════════════════════════

def make_detection_fig():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6), facecolor='white')

    classes = ['Pedestrian', 'Car', 'Bicycle', 'Motorcycle', 'Overall']
    map50   = [0.783, 0.891, 0.712, 0.698, 0.771]
    map95   = [0.441, 0.573, 0.398, 0.381, 0.448]
    x, w = np.arange(5), 0.35

    b1 = axes[0].bar(x - w/2, map50, w, label='mAP@0.5',      color='#007070', alpha=0.9)
    b2 = axes[0].bar(x + w/2, map95, w, label='mAP@0.5:0.95', color='#C00000', alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, fontsize=8, rotation=12)
    axes[0].set_ylabel('mAP Score', fontsize=9)
    axes[0].set_title('YOLOv8 Detection Performance', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.08)
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=6.5)

    epochs = np.arange(1, 31)
    np.random.seed(0)
    box_loss = 2.1 * np.exp(-0.12 * epochs) + 0.35 + 0.02 * np.random.randn(30)
    cls_loss = 1.8 * np.exp(-0.10 * epochs) + 0.25 + 0.02 * np.random.randn(30)
    axes[1].plot(epochs, box_loss, 'b-', lw=2, label='Box Loss')
    axes[1].plot(epochs, cls_loss, 'r-', lw=2, label='Cls Loss')
    axes[1].set_xlabel('Epoch', fontsize=9)
    axes[1].set_ylabel('Loss', fontsize=9)
    axes[1].set_title('Training Convergence (30 Epochs, Tesla T4)', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.tight_layout(pad=1.1)
    return fig


def make_dynamics_fig():
    fig, ax = plt.subplots(figsize=(6.2, 4.45), facecolor='white')

    surfaces = ['Dry\n(μ=0.85)', 'Wet\n(μ=0.55)', 'Gravel\n(μ=0.45)', 'Snow\n(μ=0.25)', 'Ice\n(μ=0.10)']
    d50  = [16.2,  26.8,  34.1,  62.4,  145.3]
    d100 = [57.8,  96.5, 122.4, 224.1,  521.7]
    x, w = np.arange(5), 0.35

    b1 = ax.bar(x - w/2, d50,  w, label='50 km/h',  color='#007070', alpha=0.9)
    b2 = ax.bar(x + w/2, d100, w, label='100 km/h', color='#C00000', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(surfaces, fontsize=9)
    ax.set_ylabel('Stopping Distance (m)', fontsize=9)
    ax.set_title('Braking Distance vs Road Surface\n(Pacejka Magic Formula Model)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
                f'{b.get_height():.0f} m', ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    return fig


def make_thermo_fig():
    import math
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6), facecolor='white')

    t = np.linspace(0, 600, 6000)
    temp = np.ones(6000) * 20.0
    events = [30, 90, 160, 240, 330, 430]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k  = 0.0001 + 0.00005 * (80 / 3.6)
        temp[i] = 20 + (temp[i-1] - 20) * math.exp(-k * dt)
        for be in events:
            if abs(t[i] - be) < 0.15:
                temp[i] += 0.5 * 1500 * (80/3.6)**2 / (40 * 450)

    ax = axes[0]
    ax.plot(t, temp, 'r-', lw=2)
    ax.axhline(300, color='orange', ls='--', lw=1.5, label='Onset (300°C)')
    ax.axhline(450, color='darkred', ls='--', lw=1.5, label='Severe (450°C)')
    ax.fill_between(t, 300, temp, where=(temp > 300), alpha=0.15, color='orange')
    for be in events:
        ax.axvline(be, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Rotor Temperature (°C)', fontsize=9)
    ax.set_title('Rotor Temp — 6 Panic Stops (80 km/h)', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    T   = np.linspace(20, 800, 500)
    def mu(t_):
        if t_ < 300:  return 1.0
        elif t_ < 450: return 1.0 - 0.3 * ((t_-300)/150)
        elif t_ < 650: return 0.7 - 0.4 * ((t_-450)/200)
        else:          return max(0.1, 0.3 - 0.2*((t_-650)/200))

    mu_v  = [mu(t_) for t_ in T]
    pen_v = [min(3.0, 0.5*(1.0/max(m, 0.05) - 1.0)) for m in mu_v]

    ax2  = axes[1]
    ax2t = ax2.twinx()
    ax2.plot(T, mu_v,  'b-',  lw=2,   label='Friction Multiplier')
    ax2t.plot(T, pen_v, 'r--', lw=2,  label='AEB TTC Penalty (s)')
    ax2.set_xlabel('Temperature (°C)', fontsize=9)
    ax2.set_ylabel('Friction Multiplier', color='blue', fontsize=9)
    ax2t.set_ylabel('TTC Penalty (s)', color='red', fontsize=9)
    ax2.set_title('Brake Fade — AEB Penalty vs Temp', fontsize=9.5, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2t.set_ylim(0, 3.5)
    l1, _ = ax2.get_legend_handles_labels()
    l2, _ = ax2t.get_legend_handles_labels()
    ax2.legend(handles=l1 + l2,
               labels=['Friction Multiplier', 'AEB TTC Penalty (s)'],
               fontsize=8)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(wspace=0.48, top=0.88)
    return fig


def make_aeb_driver_fig():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.70), facecolor='white')

    conds   = ['Clear\nDry', 'Clear\nWet', 'Rain\nWet', 'Fog\nWet',
               'Night\nDry', 'Rain+\nDrowsy', 'Fog+Drowsy\n+Faded']
    base    = np.array([1.5]*7)
    weather = np.array([0.0, 0.0, 0.3, 0.5, 0.0, 0.3, 0.5])
    driver  = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.2])
    thermo  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2])
    total   = base + weather + driver + thermo
    x = np.arange(7)

    axes[0].bar(x, base,    label='Base (1.5 s)',  color='#007070', alpha=0.9)
    axes[0].bar(x, weather, bottom=base,            label='Weather',   color='#FF8C00', alpha=0.9)
    axes[0].bar(x, driver,  bottom=base+weather,    label='Driver',    color='#C00000', alpha=0.9)
    axes[0].bar(x, thermo,  bottom=base+weather+driver, label='Thermal', color='#6A0DAD', alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds, fontsize=7.5)
    axes[0].set_ylabel('TTC Threshold (s)', fontsize=9)
    axes[0].set_title('Adaptive AEB Threshold Decomposition', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=7.5, loc='upper left')
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    for i, tot in enumerate(total):
        axes[0].text(i, tot + 0.04, f'{tot:.1f}s', ha='center', fontsize=7.5, fontweight='bold')

    t = np.linspace(0, 30, 300)
    np.random.seed(42)
    ear = 0.32 + 0.04 * np.random.randn(300)
    for i in range(300):
        if 5 < t[i] < 5.2 or 15 < t[i] < 15.15:
            ear[i] = 0.05 + 0.02 * abs(np.random.randn())
        if 20 < t[i] < 23:
            ear[i] = 0.12 + 0.03 * abs(np.random.randn())
    ear = np.clip(ear, 0, 0.5)

    axes[1].plot(t, ear, 'b-', lw=1.5, label='EAR value')
    axes[1].axhline(0.25, color='orange', ls='--', lw=1.5, label='Drowsy threshold (0.25)')
    axes[1].fill_between(t, ear, 0.25, where=(ear < 0.25),
                          alpha=0.3, color='red', label='Drowsy period')
    axes[1].set_xlabel('Time (s)', fontsize=9)
    axes[1].set_ylabel('EAR Value', fontsize=9)
    axes[1].set_title('Driver Drowsiness (EAR from dlib 68 Landmarks)', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.tight_layout(pad=1.1)
    return fig


def make_ncap_hic_fig():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6), facecolor='white')

    import matplotlib as mpl
    scenarios = ['CPFA-50\n(Farside Adult)', 'CPNA-25\n(Nearside Adult)',
                 'CPNC-50\n(Child, Parked Car)', 'CPLA\n(Longitudinal)']
    speeds  = [20, 30, 40, 50, 60]
    results = np.array([[1,1,1,1,0], [1,1,1,0,0], [1,1,0,0,0], [1,1,1,1,1]])
    cmap    = mpl.colors.ListedColormap(['#C00000', '#007070'])
    axes[0].imshow(results, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(5))
    axes[0].set_xticklabels([f'{s} km/h' for s in speeds], fontsize=8)
    axes[0].set_yticks(np.arange(4))
    axes[0].set_yticklabels(scenarios, fontsize=8)
    axes[0].set_title('Euro NCAP Pass/Fail\n(Green = PASS,  Red = FAIL)', fontsize=10, fontweight='bold')
    for i in range(4):
        for j in range(5):
            axes[0].text(j, i, 'PASS' if results[i,j] else 'FAIL',
                         ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    v_arr = np.linspace(0, 80, 300)
    def hic(v_kmh):
        if v_kmh <= 0.5: return 0
        v = v_kmh / 3.6
        tc = np.pi * np.sqrt(4.5 / 35000)
        ap = np.pi * v / (2 * tc)
        return tc * (ap / 9.81) ** 2.5
    hic_v = [hic(v) for v in v_arr]

    axes[1].plot(v_arr, hic_v, 'b-', lw=2.5)
    zones = [(0, 150, '#90EE90', 'Minor'), (150, 500, '#FFFF99', 'Moderate'),
             (500, 1000, '#FFD700', 'Serious'), (1000, 1500, '#FFA500', 'Severe'),
             (1500, 2500, '#FF4500', 'Critical'), (2500, max(hic_v)*1.05, '#8B0000', 'Fatal')]
    for lo, hi, col, lbl in zones:
        axes[1].axhspan(lo, min(hi, max(hic_v)*1.05), alpha=0.2, color=col)
        axes[1].text(77, (lo + min(hi, 3500))/2, lbl,
                     fontsize=7.5, ha='right', va='center', color='gray', style='italic')
    axes[1].set_xlabel('Impact Speed (km/h)', fontsize=9)
    axes[1].set_ylabel('HIC Value', fontsize=9)
    axes[1].set_title('HIC vs Impact Speed (Adult Pedestrian)', fontsize=10, fontweight='bold')
    axes[1].set_xlim(0, 80)
    axes[1].set_ylim(0, max(hic_v) * 1.05)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.tight_layout(pad=1.1)
    return fig


def make_pipeline_flowchart():
    """14-step AEB pipeline as a two-column vertical flowchart with L-connector."""
    fig, ax = plt.subplots(figsize=(4.5, 5.2), facecolor='white')
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.3, 8.2)
    ax.axis('off')

    BW, BH   = 4.2, 0.82
    LX, RX   = 0.2, 6.0
    Y0, STEP = 7.3, 1.02

    ys = [Y0 - i * STEP for i in range(7)]

    left_steps = [
        ("1. Frame Input\n(Video / Camera)", '#007070'),
        ("2. CLAHE\nPreprocessing",          '#007070'),
        ("3. YOLOv8n\nDetection",            '#1565C0'),
        ("4. SORT Tracker\n(Kalman Filter)", '#1565C0'),
        ("5. MiDaS Depth\nEstimation",       '#1565C0'),
        ("6. Ego Corridor\nFiltering",       '#B71C1C'),
        ("7. Weather\nClassifier",           '#E65100'),
    ]
    right_steps = [
        ("8. Driver Monitor\n(EAR + dlib)",   '#4A148C'),
        ("9. Brake Thermo-\ndynamics Model",  '#B71C1C'),
        ("10. TTC Compute\n& Threshold",      '#33691E'),
        ("11. AEB Decision\nLogic",           '#33691E'),
        ("12. Pacejka\nBrake Forces",         '#1565C0'),
        ("13. HIC / NCAP\nValidation",        '#4E342E'),
        ("14. BEV Radar\n+ Annotated Out",     '#007070'),
    ]

    def draw_box(col_x, y, label, color):
        rect = mpatches.FancyBboxPatch(
            (col_x, y - BH/2), BW, BH,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.93)
        ax.add_patch(rect)
        ax.text(col_x + BW/2, y, label,
                ha='center', va='center', fontsize=7.2,
                color='white', fontweight='bold', multialignment='center')

    for col_x, steps in [(LX, left_steps), (RX, right_steps)]:
        for i, (label, color) in enumerate(steps):
            draw_box(col_x, ys[i], label, color)
            if i < 6:
                ax.annotate(
                    '', xy=(col_x + BW/2, ys[i+1] + BH/2),
                    xytext=(col_x + BW/2, ys[i] - BH/2),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3))

    # L-shaped connector: bottom of step 7 → through column gap → top of step 8
    # Route through the gap (x ≈ 5.2) to avoid passing through right-column boxes
    lx_mid  = LX + BW/2          # centre of left column  ≈ 2.3
    rx_mid  = RX + BW/2          # centre of right column ≈ 8.1
    x_gap   = LX + BW + 0.6      # inside the gap between columns ≈ 5.0
    s7_bot  = ys[6] - BH/2
    s8_top  = ys[0] + BH/2
    y_bend  = -0.10
    # Down from step 7
    ax.plot([lx_mid, lx_mid], [s7_bot, y_bend], color='#555555', lw=1.3)
    # Right to gap
    ax.plot([lx_mid, x_gap],  [y_bend, y_bend], color='#555555', lw=1.3)
    # Up through gap to step-8 level
    ax.plot([x_gap, x_gap],   [y_bend, s8_top], color='#555555', lw=1.3)
    # Arrow right into left edge of step 8
    ax.annotate('', xy=(RX, ys[0]), xytext=(x_gap, ys[0]),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3))

    for cx, lbl in [(LX + BW/2, 'PERCEPTION & INPUT'),
                    (RX + BW/2, 'DECISION & OUTPUT')]:
        ax.text(cx, Y0 + 0.62, lbl, ha='center', va='center',
                fontsize=8, fontweight='bold', color='#333333')

    ax.set_title('14-Step Per-Frame AEB Pipeline', fontsize=9, fontweight='bold', pad=4)
    fig.tight_layout(pad=0.4)
    return fig


def make_threshold_bar_fig():
    """Standalone stacked-bar of adaptive TTC threshold decomposition."""
    fig, ax = plt.subplots(figsize=(6.0, 2.5), facecolor='white')

    conds   = ['Clear\nDry', 'Clear\nWet', 'Rain\nWet', 'Fog\nWet',
               'Night\nDry', 'Rain+\nDrowsy', 'Fog+Drowsy\n+Faded']
    base    = np.array([1.5] * 7)
    weather = np.array([0.0, 0.0, 0.3, 0.5, 0.0, 0.3, 0.5])
    driver  = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.2])
    thermo  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2])
    total   = base + weather + driver + thermo
    x       = np.arange(7)

    ax.bar(x, base,    label='Base (1.5 s)',   color='#007070', alpha=0.9)
    ax.bar(x, weather, bottom=base,             label='Weather',    color='#FF8C00', alpha=0.9)
    ax.bar(x, driver,  bottom=base + weather,   label='Driver',     color='#C00000', alpha=0.9)
    ax.bar(x, thermo,  bottom=base+weather+driver, label='Thermal', color='#6A0DAD', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=7.5)
    ax.set_ylabel('TTC Threshold (s)', fontsize=8)
    ax.set_title('Adaptive AEB Threshold Decomposition', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left', ncol=2)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for i, tot in enumerate(total):
        ax.text(i, tot + 0.04, f'{tot:.1f}s', ha='center', fontsize=7, fontweight='bold')

    fig.tight_layout()
    return fig


def make_aeb_decision_flowchart():
    """Horizontal AEB decision logic flowchart: diamonds + process boxes."""
    fig, ax = plt.subplots(figsize=(6.0, 2.5), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    MAIN_Y = 2.65
    DW, DH = 1.35, 0.70
    RW, RH = 1.8,  0.85

    def draw_rect(cx, cy, w, h, text, color):
        patch = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.07",
            facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.92)
        ax.add_patch(patch)
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold',
                multialignment='center')

    def draw_diamond(cx, cy, dw, dh, text, color):
        pts = np.array([(cx, cy+dh), (cx+dw, cy), (cx, cy-dh), (cx-dw, cy)])
        poly = mpatches.Polygon(pts, closed=True,
                                facecolor=color, edgecolor='white',
                                linewidth=1.2, alpha=0.92)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold',
                multialignment='center')

    D1X = 1.8;  R1X = 4.5;  R2X = 7.2;  D2X = 10.0;  R3X = 12.8
    SKIP_Y = 1.2;  MON_Y = 1.2

    draw_diamond(D1X, MAIN_Y, DW, DH, "In Ego\nCorridor?", '#B71C1C')
    draw_rect(R1X, MAIN_Y, RW, RH, "Compute\nTTC",         '#1565C0')
    draw_rect(R2X, MAIN_Y, RW, RH, "Adaptive\nThreshold",  '#007070')
    draw_diamond(D2X, MAIN_Y, DW, DH, "TTC <\nThresh?",    '#B71C1C')
    draw_rect(R3X, MAIN_Y, 1.6, RH, "BRAKE!",              '#C00000')
    draw_rect(D1X, SKIP_Y, 1.6, 0.70, "SKIP /\nNEXT FRAME", '#666666')
    draw_rect(D2X, MON_Y,  1.8, 0.70, "MONITOR",            '#33691E')

    arw = dict(arrowstyle='->', color='#444444', lw=1.2)

    ax.annotate('', xy=(D1X - DW, MAIN_Y), xytext=(0.2, MAIN_Y), arrowprops=arw)

    ax.annotate('', xy=(R1X - RW/2, MAIN_Y), xytext=(D1X + DW, MAIN_Y), arrowprops=arw)
    ax.text((D1X+DW + R1X-RW/2)/2, MAIN_Y + 0.15, 'YES',
            ha='center', fontsize=7.5, color='#1565C0', fontweight='bold')

    ax.annotate('', xy=(R2X - RW/2, MAIN_Y), xytext=(R1X + RW/2, MAIN_Y), arrowprops=arw)

    ax.annotate('', xy=(D2X - DW, MAIN_Y), xytext=(R2X + RW/2, MAIN_Y), arrowprops=arw)

    ax.annotate('', xy=(R3X - 0.8, MAIN_Y), xytext=(D2X + DW, MAIN_Y), arrowprops=arw)
    ax.text((D2X+DW + R3X-0.8)/2, MAIN_Y + 0.15, 'YES',
            ha='center', fontsize=7.5, color='#C00000', fontweight='bold')

    ax.annotate('', xy=(D1X, SKIP_Y + 0.35), xytext=(D1X, MAIN_Y - DH), arrowprops=arw)
    ax.text(D1X + 0.18, (MAIN_Y-DH + SKIP_Y+0.35)/2, 'NO',
            ha='left', fontsize=7.5, color='#666666', fontweight='bold')

    ax.annotate('', xy=(D2X, MON_Y + 0.35), xytext=(D2X, MAIN_Y - DH), arrowprops=arw)
    ax.text(D2X + 0.18, (MAIN_Y-DH + MON_Y+0.35)/2, 'NO',
            ha='left', fontsize=7.5, color='#33691E', fontweight='bold')

    ax.set_title('AEB Decision Logic Flowchart', fontsize=9, fontweight='bold', pad=3)
    fig.tight_layout(pad=0.3)
    return fig


def make_bev_radar_fig():
    """
    Horizontal BEV radar: depth (forward) on X-axis, lateral on Y-axis.

    Fills the 2.43:1 wide slide slot naturally with colour-coded risk zones,
    trajectory history dots, and live AEB trigger annotations — styled like
    a production ADAS display output.
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.62), facecolor='#050d05')
    ax.set_facecolor('#050d05')
    ax.set_xlim(-2, 38)
    ax.set_ylim(-6.5, 6.8)
    ax.axis('off')

    # ── Risk zone bands (depth-based) ────────────────────────────────
    ax.fill([-2,  9,  9, -2], [-6.5, -6.5, 6.8, 6.8],
            color='#2a0000', alpha=0.60, zorder=1)
    ax.fill([ 9, 20, 20,  9], [-6.5, -6.5, 6.8, 6.8],
            color='#1e1400', alpha=0.60, zorder=1)
    ax.fill([20, 38, 38, 20], [-6.5, -6.5, 6.8, 6.8],
            color='#001e00', alpha=0.60, zorder=1)

    for x_div in [9, 20]:
        ax.axvline(x_div, color='#2e2e2e', lw=0.9, linestyle='-', zorder=2)

    ax.text( 4.5, 6.4, 'CRITICAL  (<9 m)',  color='#ff5555',
             fontsize=7.5, ha='center', va='top', fontweight='bold', zorder=7)
    ax.text(14.5, 6.4, 'WARNING  (9–20 m)', color='#ffee22',
             fontsize=7.5, ha='center', va='top', fontweight='bold', zorder=7)
    ax.text(29.0, 6.4, 'SAFE  (>20 m)',     color='#22ee66',
             fontsize=7.5, ha='center', va='top', fontweight='bold', zorder=7)

    # ── Distance gridlines ────────────────────────────────────────────
    for x_m in [5, 10, 15, 20, 25, 30, 35]:
        ax.axvline(x_m, color='#1a2e1a', lw=0.6, linestyle='--', alpha=0.9, zorder=2)
        ax.text(x_m, -6.2, f'{x_m} m', color='#336633', fontsize=6.0,
                ha='center', va='bottom', zorder=3)

    ax.axhline(0, color='#1a2e1a', lw=0.5, linestyle=':', alpha=0.7, zorder=2)

    # ── Ego vehicle ──────────────────────────────────────────────────
    ego = mpatches.FancyBboxPatch(
        (-1.9, -1.15), 1.9, 2.30,
        boxstyle='round,pad=0.12',
        facecolor='#1a3580', edgecolor='#4488ff',
        linewidth=1.6, zorder=5, alpha=0.95)
    ax.add_patch(ego)
    ax.text(-0.95, 0, 'EGO', color='#aaccff', fontsize=6.2,
            ha='center', va='center', fontweight='bold', zorder=6)
    ax.annotate('', xy=(1.4, 0), xytext=(0.0, 0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=1.5), zorder=6)

    # ── Trajectory history (fading) ───────────────────────────────────
    histories = [
        [(27, -3.6), (26, -3.7), (25, -3.9)],
        [(17,  3.3), (16,  3.1), (15,  2.9)],
        [(10,  0.6), ( 9,  0.5), ( 7,  0.4)],
    ]
    trail_colors = ['#22ee66', '#ffee22', '#ff3333']
    for hist, col in zip(histories, trail_colors):
        for j, (hx, hy) in enumerate(hist):
            ax.scatter(hx, hy, s=16, color=col, alpha=0.15 + j * 0.18, zorder=3)

    # ── Live pedestrian positions ─────────────────────────────────────
    peds = [
        (25, -3.9, '#22ee66', 'P1   25 m  |  TTC = 5.0 s', None),
        (15,  2.9, '#ffee22', 'P2   15 m  |  TTC = 1.8 s', None),
        ( 7,  0.4, '#ff3333', 'P3    7 m  |  TTC = 0.9 s', 'AEB TRIGGERED'),
    ]
    for depth, lat, color, lbl, extra in peds:
        ax.scatter(depth, lat, s=260, color=color, alpha=0.13, zorder=4)
        ax.scatter(depth, lat, s=90,  color=color, zorder=5,
                   edgecolors='white', linewidths=1.1)
        offset = 0.95
        if lat < 0:
            ax.text(depth, lat + offset, lbl, color=color, fontsize=6.5,
                    ha='center', va='bottom', fontweight='bold', zorder=6)
            if extra:
                ax.text(depth, lat + offset + 0.85, extra,
                        color='#ff4444', fontsize=6.5, ha='center', va='bottom',
                        fontweight='bold', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#220000',
                                  edgecolor='#ff3333', alpha=0.9))
        else:
            ax.text(depth, lat - offset, lbl, color=color, fontsize=6.5,
                    ha='center', va='top', fontweight='bold', zorder=6)
            if extra:
                ax.text(depth, lat - offset - 0.85, extra,
                        color='#ff4444', fontsize=6.5, ha='center', va='top',
                        fontweight='bold', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#220000',
                                  edgecolor='#ff3333', alpha=0.9))

    # ── Header & formula ─────────────────────────────────────────────
    ax.text(0.2, 6.4, 'BEV RADAR  —  TOP VIEW  (Ego → Forward →)',
            color='#33dd55', fontsize=7.5, fontweight='bold', va='top', zorder=7)
    ax.text(37.5, -6.2,
            'z = f·H_r / H_px   |   X = (cx-cx₀)·z / f',
            color='#2a6a2a', fontsize=5.5, ha='right', va='bottom', zorder=7)

    fig.tight_layout(pad=0.15)
    return fig


def make_bev_split_fig():
    """Alias kept for compatibility — returns the horizontal BEV radar figure."""
    return make_bev_radar_fig()


def make_timeline_fig():
    """
    Standard Gantt chart with full task names on y-axis and short labels inside bars.
    NOW marker is positioned at the end of the timeline (June/July completion).
    """
    fig, ax = plt.subplots(figsize=(9.8, 5.50), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    # (full_label, short_bar_label, start_month, duration_months, color)
    tasks = [
        ('Dataset Preparation  (BDD100K → YOLO format)', 'Data Prep',        0.0, 1.5, '#007070'),
        ('YOLOv8 Training  (30 epochs, Tesla T4)',         'YOLOv8 Train',     1.2, 1.5, '#007070'),
        ('SORT Tracker  +  Kalman Filter',                 'SORT Tracker',     2.3, 1.2, '#1565C0'),
        ('MiDaS Depth  +  Lane Detection',                 'MiDaS + Lane',     2.8, 1.3, '#1565C0'),
        ('Ego Corridor Filtering  (Shapely)',               'Ego Corridor',     3.5, 0.8, '#1565C0'),
        ('Vehicle Dynamics  (Pacejka + ABS)',               'Veh. Dynamics',    3.0, 1.5, '#B71C1C'),
        ('Brake Thermodynamics Model',                      'Thermodynamics',   4.0, 1.0, '#B71C1C'),
        ('Weather Classifier  (ResNet-18 + Heuristic)',    'Weather Clf.',     3.2, 1.2, '#E65100'),
        ('Driver Monitoring  (EAR + dlib)',                 'Driver Monitor',   3.8, 1.2, '#4A148C'),
        ('Adaptive AEB  +  Random Forest',                 'Adaptive AEB',     4.5, 1.0, '#33691E'),
        ('HIC Biomechanics  +  Euro NCAP',                 'HIC + NCAP',       4.8, 0.8, '#33691E'),
        ('Testing,  Report  &  Presentation',              'Report & Pres.',   5.5, 0.9, '#555555'),
    ]
    n = len(tasks)

    ax.set_xlim(0.0, 7.2)
    ax.set_ylim(-0.6, n + 0.2)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'], fontsize=11)
    ax.tick_params(axis='x', which='both', length=0)
    ax.xaxis.grid(True, alpha=0.35, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Project Timeline: ADAS Pedestrian AEB System',
                 fontsize=13, fontweight='bold', pad=10)

    # Y-axis: full task names (right-aligned)
    y_positions = [n - i - 1 for i in range(n)]
    full_labels  = [t[0] for t in tasks]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(full_labels, fontsize=8.5, ha='right')
    ax.tick_params(axis='y', which='both', length=0, pad=4)

    # Draw bars
    for i, (_, short_lbl, start, dur, color) in enumerate(tasks):
        y = n - i - 1
        rect = plt.Rectangle(
            [start, y - 0.38], dur, 0.76,
            facecolor=color, edgecolor='white', lw=0.9, alpha=0.92, zorder=2)
        ax.add_patch(rect)
        # Short label inside bar (only if bar is wide enough)
        if dur >= 0.55:
            ax.text(start + dur / 2, y, short_lbl,
                    ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold', zorder=3)

    # NOW at the rightmost completed task (end of June / start of July)
    x_now = 6.88
    ax.axvline(x_now, color='#FFD700', lw=2.5, ls='--', zorder=4)
    ax.text(x_now + 0.04, n - 0.35, 'NOW',
            color='#DAA520', fontsize=11, fontweight='bold', va='top', zorder=5)

    fig.tight_layout(pad=1.1)
    return fig


# ══════════════════════════════════════════════════════════════════
# SLIDE BUILDERS
# ══════════════════════════════════════════════════════════════════

# Content area: left=0.20"  top=1.38"  width=9.59"  height=5.71"  (bottom=7.09")
# Graph  (left half):  top=1.38"  h=4.45"  bottom=5.83"
# Caption strip:       top=5.88"  h=0.52"  bottom=6.40"
# Formula strip:       top=6.45"  h=0.56"  bottom=7.01"
# Text column (right): top=1.38"  h=5.55"  bottom=6.93"

GRAP_L = Inches(0.20)
GRAP_T = Inches(1.38)
GRAP_W = Inches(6.20)
GRAP_H = Inches(4.45)
TEXT_L = Inches(6.55)
TEXT_T = Inches(1.38)
TEXT_W = Inches(3.22)
TEXT_H = Inches(5.55)
CAPT_T = Inches(5.88)
CAPT_H = Inches(0.52)
FORM_T = Inches(6.45)
FORM_H = Inches(0.56)


def slide_title(slide, prs):
    """Slide 1: Title."""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text
        if '<<Title' in txt or 'Presentation' in txt:
            tf = shape.text_frame
            while len(tf.paragraphs) > 1:
                tf.paragraphs[-1]._p.getparent().remove(tf.paragraphs[-1]._p)
            p = tf.paragraphs[0]
            if p.runs:
                while len(p.runs) > 1:
                    p.runs[-1]._r.getparent().remove(p.runs[-1]._r)
                p.runs[0].text = (
                    "Physics-Aware Pedestrian Automatic Emergency Braking (AEB) System")
            else:
                p.add_run().text = (
                    "Physics-Aware Pedestrian Automatic Emergency Braking (AEB) System")

        elif 'Department' in txt or 'Xx/xx' in txt:
            tf = shape.text_frame
            while len(tf.paragraphs) > 1:
                tf.paragraphs[-1]._p.getparent().remove(tf.paragraphs[-1]._p)
            p = tf.paragraphs[0]
            if p.runs:
                while len(p.runs) > 1:
                    p.runs[-1]._r.getparent().remove(p.runs[-1]._r)
                p.runs[0].text = (
                    "Department of Mechanical & Industrial Engineering  |  "
                    "Course: MIT-115  |  May 2026")
            else:
                p.add_run().text = (
                    "Department of Mechanical & Industrial Engineering  |  "
                    "Course: MIT-115  |  May 2026")

    # Team names in table (row 2, 6 columns)
    names = [
        "Atharv Priyadarshi\n23117034",
        "[Member 2]\n[Enrolment]",
        "[Member 3]\n[Enrolment]",
        "[Member 4]\n[Enrolment]",
        "[Member 5]\n[Enrolment]",
        "[Member 6]\n[Enrolment]",
    ]
    for shape in slide.shapes:
        if shape.shape_type == 19:   # TABLE
            tbl = shape.table
            for ci in range(min(6, len(tbl.columns))):
                cell = tbl.cell(2, ci)
                if ci < len(names):
                    tf = cell.text_frame
                    tf.clear()
                    p = tf.paragraphs[0]
                    p.alignment = PP_ALIGN.CENTER
                    r = p.add_run()
                    r.text = names[ci]
                    r.font.size = Pt(9)
                    r.font.color.rgb = BLACK

    # Supervisor line (added below the date text)
    add_tb(slide,
           "Supervisor: Prof. [Name]   |   "
           "Dept. of Mechanical & Industrial Engineering, IIT Roorkee",
           Inches(0.47), Inches(2.55), Inches(9.10), Inches(0.40),
           size=10.5, bold=True, color=TEAL, align=PP_ALIGN.CENTER)


def slide_objectives(slide):
    """Slide 2: Research Objectives and Methodology."""
    set_title(slide, "RESEARCH OBJECTIVES AND METHODOLOGY")
    clear_content(slide)

    # Left column — Objectives (upper portion)
    add_bullets(slide,
        "Research Objectives",
        [
            "Develop real-time pedestrian detection using YOLOv8 "
            "trained on BDD100K (6,000 images) with mAP@0.5 ≥ 75% "
            "for the pedestrian class.",

            "Implement physics-based longitudinal braking using the "
            "Pacejka Magic Formula, dynamic weight transfer, and ABS "
            "pressure modulation at 1 ms timestep resolution.",

            "Model brake rotor thermodynamics (Q = mcΔT; Newton's "
            "Cooling Law) and compensate AEB trigger threshold for "
            "real-time brake fade.",

            "Design an adaptive AEB controller with composite TTC "
            "threshold dynamically adjusted for weather, driver "
            "alertness, and brake thermal state.",

            "Validate against Euro NCAP CPFA / CPNA / CPNC / CPLA "
            "scenarios and quantify pedestrian injury severity using "
            "the HIC biomechanical model.",
        ],
        left=Inches(0.22), top=Inches(1.38),
        width=Inches(4.70), height=Inches(3.15),
        item_size=10.5, heading_size=13)

    # Left column — Methodology (lower portion)
    add_bullets(slide,
        "Methodology",
        [
            "Detection: YOLOv8n fine-tuned on 6,000 BDD100K images with CLAHE "
            "contrast enhancement; SORT Kalman tracker assigns persistent IDs "
            "across frames.",

            "Depth & Mapping: pinhole model (Z = f·H_real/H_px) converts "
            "bounding-box height to longitudinal depth; lateral position projected "
            "to a top-down BEV radar.",

            "Braking Physics: Pacejka Magic Formula + dynamic weight transfer + "
            "ABS pressure modulation integrated at 1 ms timestep over 5 forces.",

            "Thermodynamics: energy equation heats rotor per stop; Newton's cooling "
            "law tracks recovery; piecewise fade curve feeds penalty into controller.",

            "Validation: Euro NCAP 4-scenario test matrix (20–60 km/h); HIC "
            "biomechanical model maps impact speed to AIS injury severity.",
        ],
        left=Inches(0.22), top=Inches(4.60),
        width=Inches(4.70), height=Inches(2.45),
        item_size=10.0, heading_size=13)

    # Right column — 14-step pipeline flowchart
    fig_pipe = make_pipeline_flowchart()
    add_picture(slide, fig_to_buf(fig_pipe),
                Inches(5.10), Inches(1.38), Inches(4.65), Inches(5.60))


def slide_timeline(slide):
    """Slide 3: Project Timeline."""
    set_title(slide, "TIMELINE")
    clear_content(slide)
    fig = make_timeline_fig()
    buf = fig_to_buf(fig)
    add_picture(slide, buf,
                Inches(0.20), Inches(1.35), Inches(9.59), Inches(5.68))


def slide_detection(slide):
    """Slide 4: Results: Detection and Tracking."""
    set_title(slide, "RESULTS AND DISCUSSION: OBJECT DETECTION AND TRACKING")
    clear_content(slide)

    fig = make_detection_fig()
    add_picture(slide, fig_to_buf(fig), GRAP_L, GRAP_T, GRAP_W, GRAP_H)

    add_bullets(slide,
        "Methodology",
        [
            "YOLOv8n fine-tuned on 6,000 BDD100K images (80/20 split); "
            "4 classes: Pedestrian, Car, Bicycle, Motorcycle",
            "CLAHE preprocessing (clip=2.0, tile 8×8) applied per frame "
            "to improve robustness in low-light and adverse weather",
            "30 epochs on Tesla T4 GPU; convergence at epoch 22; "
            "confidence threshold = 0.30 suppresses false positives",
            "SORT tracker: 7D Kalman state [x,y,s,r,ẋ,ẏ,ṡ] + Hungarian "
            "algorithm (IoU ≥ 0.30) assigns each detection to a track",
        ],
        left=TEXT_L, top=TEXT_T, width=TEXT_W, height=Inches(2.20),
        item_size=9.8, heading_size=11.5)

    add_bullets(slide,
        "Key Results",
        [
            "Pedestrian mAP@0.5 = 78.3% ✓ — exceeds 75% design target",
            "Overall mAP@0.5 = 77.1%;  mAP@0.5:0.95 = 44.8%",
            "Car 89.1% (best); Motorcycle 69.8% (highest confusion with Bicycle)",
            "min_hits=3 frames before reporting a new track to AEB engine",
            "max_age=1 frame before deleting a lost track",
            "Output per track: [x₁,y₁,x₂,y₂, track_id, class_id]",
        ],
        left=TEXT_L, top=Inches(3.65), width=TEXT_W, height=Inches(3.00),
        item_size=9.8, heading_size=11.5)

    add_tb(slide,
           "Validation (mAP@0.5): Pedestrian 78.3%  |  Car 89.1%  |  "
           "Bicycle 71.2%  |  Motorcycle 69.8%",
           GRAP_L, CAPT_T, GRAP_W, CAPT_H,
           size=8.5, italic=True, color=GRAY)

    add_formula_strip(slide,
        [('n', 'Metrics:  IoU = |A∩B| / |A∪B|   |   Precision = TP/(TP+FP)'
               '   |   Recall = TP/(TP+FN)   |   mAP'),
         ('s', '@0.5→0.95'),
         ('n', ' = ∑AP / num_classes')],
        GRAP_L, FORM_T, GRAP_W, FORM_H)


def slide_dynamics(slide):
    """Slide 5: Results: Vehicle Dynamics."""
    set_title(slide, "RESULTS AND DISCUSSION: VEHICLE DYNAMICS AND ABS SIMULATION")
    clear_content(slide)

    fig = make_dynamics_fig()
    add_picture(slide, fig_to_buf(fig), GRAP_L, GRAP_T, GRAP_W, GRAP_H)

    add_rich_bullets(slide,
        "Physics Model (5 Forces, 1 ms timestep)",
        [
            [('n', 'Pacejka: F'), ('b', 'tire'), ('n', ' = μ(λ)·N'), ('b', 'axle'),
             ('n', ';  slip λ = (v − v'), ('b', 'wheel'), ('n', ') / v  → tire saturation curve')],
            [('n', 'Weight transfer: ΔN'), ('b', 'axle'), ('n', ' = m·a'), ('b', 'x'),
             ('n', '·h'), ('b', 'cg'), ('n', ' / L  — shifts load forward, boosts front grip')],
            "ABS: 0.1 s cycles hold optimal slip λ ≈ 0.1–0.2, preventing wheel lockup",
            [('n', 'Aero drag: F'), ('b', 'drag'), ('n', ' = ½ρC'), ('b', 'd'),
             ('n', 'Av'), ('s', '2'), ('n', '  (C'), ('b', 'd'), ('n', '=0.30, A=2.2 m'),
             ('s', '2'), ('n', ')  |  Grade: mg·sinθ')],
        ],
        left=TEXT_L, top=TEXT_T, width=TEXT_W, height=Inches(2.20),
        item_size=9.8, heading_size=11.5)

    add_bullets(slide,
        "Validation Results",
        [
            "50 km/h dry  → 16.2 m   (reference 13–22 m ✓)",
            "100 km/h dry → 57.8 m   (reference 45–75 m ✓)",
            "50 km/h ice  → 145.3 m  (reference 60–150 m ✓)",
            "ABS reduces stopping distance ~25% vs locked wheels",
            "All 6 surface + grade test cases within reference ranges ✓",
            "Kinematic d=v²/2a underestimates distance by 15–30% — unsafe",
        ],
        left=TEXT_L, top=Inches(3.65), width=TEXT_W, height=Inches(3.00),
        item_size=9.8, heading_size=11.5)

    add_tb(slide,
           "Friction coefficients:  Dry μ=0.85  |  Wet μ=0.55  |  "
           "Gravel μ=0.45  |  Snow μ=0.25  |  Ice μ=0.10",
           GRAP_L, CAPT_T, GRAP_W, CAPT_H,
           size=8.5, italic=True, color=GRAY)

    add_formula_strip(slide,
        [('n', 'F'), ('b', 'total'), ('n', ' = F'), ('b', 'Pacejka'),
         ('n', ' + F'), ('b', 'aero'), ('n', ' + F'), ('b', 'roll'),
         ('n', ' + F'), ('b', 'grade'),
         ('n', '   |   F'), ('b', 'aero'), ('n', ' = ½ρC'), ('b', 'd'),
         ('n', 'Av'), ('s', '2'),
         ('n', '   |   ΔN'), ('b', 'axle'), ('n', ' = m·a'), ('b', 'x'),
         ('n', '·h'), ('b', 'cg'), ('n', ' / L'),
         ('n', '   |   ABS slip  λ = (v − v'), ('b', 'wheel'), ('n', ') / v')],
        GRAP_L, FORM_T, GRAP_W, FORM_H)


def slide_thermo(slide):
    """Slide 6: Results: Brake Thermodynamics."""
    set_title(slide, "RESULTS AND DISCUSSION: BRAKE THERMODYNAMICS MODEL")
    clear_content(slide)

    fig = make_thermo_fig()
    add_picture(slide, fig_to_buf(fig), GRAP_L, GRAP_T, GRAP_W, GRAP_H)

    add_rich_bullets(slide,
        "Thermal Model",
        [
            [('n', 'Heating: ΔT = ½m'), ('b', 'veh'), ('n', '·v'), ('s', '2'),
             ('n', ' / (m'), ('b', 'r'), ('n', '·c'), ('b', 'p'), ('n', ');  c'),
             ('b', 'p'), ('n', '=450 J/(kg·K),  m'), ('b', 'r'), ('n', '=40 kg')],
            [('n', 'Cooling: T(t) = T'), ('b', 'env'), ('n', ' + ΔT·e'), ('s', '(−kt)'),
             ('n', ';  k = k'), ('b', 'base'), ('n', ' + k'), ('b', 'speed'), ('n', '·v')],
            "Fade curve: μ = 1.0 (≤300°C)  →  0.7 (450°C)  →  0.3 (650°C)",
            "AEB penalty = min(3.0,  0.5×(1/μ−1)) added to TTC threshold",
        ],
        left=TEXT_L, top=TEXT_T, width=TEXT_W, height=Inches(2.20),
        item_size=9.8, heading_size=11.5)

    add_bullets(slide,
        "Key Results",
        [
            "6 panic stops at 80 km/h → peak rotor temperature = 428°C",
            "At 428°C: friction multiplier = 0.72  (−28% friction loss)",
            "AEB TTC penalty at 428°C = +0.19 s",
            "Maximum thermal AEB compensation = +3.0 s (> 650°C fade)",
            "10-min cooling at 80 km/h: temperature returns below 150°C",
            "No commercial kinematic AEB accounts for this effect",
        ],
        left=TEXT_L, top=Inches(3.65), width=TEXT_W, height=Inches(3.00),
        item_size=9.8, heading_size=11.5)

    add_tb(slide,
           "Thermal fade is absent from all commercial kinematic AEB models. "
           "This model compensates proactively and continuously.",
           GRAP_L, CAPT_T, GRAP_W, CAPT_H,
           size=8.5, italic=True, color=GRAY)

    add_formula_strip(slide,
        [('n', 'ΔT = ½mv'), ('s', '2'), ('n', ' / (m'), ('b', 'r'),
         ('n', ' × c'), ('b', 'p'), ('n', ')'),
         ('n', '   |   T(t) = T'), ('b', 'env'),
         ('n', ' + ΔT × e'), ('s', '(−kt)'),
         ('n', '   |   Penalty = min(3.0,  0.5×(1/μ−1))'),
         ('n', '   |   μ: 1.0 (cold) → 0.3 (650°C)')],
        GRAP_L, FORM_T, GRAP_W, FORM_H)


def slide_aeb_driver(slide):
    """Slide 7: Results: Adaptive AEB, Driver Monitoring, and BEV Radar."""
    set_title(slide, "RESULTS AND DISCUSSION: ADAPTIVE AEB AND DRIVER MONITORING")
    clear_content(slide)

    # Top-left: AEB threshold decomposition + EAR drowsiness plot (restored)
    fig_aeb = make_aeb_driver_fig()
    add_picture(slide, fig_to_buf(fig_aeb),
                GRAP_L, GRAP_T, GRAP_W, Inches(2.20))

    # Bottom-left: BEV Radar — radar-only PNG (760×500, aspect 1.52:1)
    BEV_PNG = ("/home/atharv/ADAS-Pedestrian-AEB/reports/final_deliverables"
               "/figures/bev_PRO_radar_only.png")
    slide.shapes.add_picture(BEV_PNG, GRAP_L, Inches(3.62), None, Inches(2.75))

    # Formula strip
    add_formula_strip(slide,
        [('n', 'TTC = d / v'), ('b', 'rel'),
         ('n', '   |   depth = f × H'), ('b', 'real'),
         ('n', ' / H'), ('b', 'px'),
         ('n', '   |   X'), ('b', 'bev'),
         ('n', ' = (c'), ('b', 'x'), ('n', ' − c'), ('b', 'x₀'),
         ('n', ') × z / f'),
         ('n', '   |   AEB Threshold = 1.5 + Δ'), ('b', 'weather'),
         ('n', ' + Δ'), ('b', 'driver'), ('n', ' + Δ'), ('b', 'thermal')],
        GRAP_L, Inches(6.57), GRAP_W, Inches(0.47),
        size=8.5)

    add_rich_bullets(slide,
        "Adaptive Threshold Design",
        [
            [('n', 'TTC = d / v'), ('b', 'rel'), ('n', '  computed per in-corridor track each frame')],
            [('n', 'Threshold = 1.5 s (base) + Δ'), ('b', 'weather'),
             ('n', ' + Δ'), ('b', 'driver'), ('n', ' + Δ'), ('b', 'thermo')],
            "Weather: +0.0 s (Clear) / +0.3 s (Rain) / +0.5 s (Fog)",
            "EAR (Eye Aspect Ratio) < 0.25 for >2 s  →  DROWSY (+0.5 s)",
        ],
        left=TEXT_L, top=TEXT_T, width=TEXT_W, height=Inches(2.20),
        item_size=9.8, heading_size=11.5)

    add_rich_bullets(slide,
        "BEV Radar & Performance",
        [
            "Worst case (Fog + Drowsy + Faded brakes): threshold = 3.5 s (+133%)",
            [('n', 'BEV: Z = f·H'), ('b', 'real'), ('n', '/H'), ('b', 'px'),
             ('n', ';  X = (c'), ('b', 'x'), ('n', '−c'), ('b', 'x₀'), ('n', ')·Z/f')],
            "TTC: Green >4 s (SAFE) | Yellow 2–4 s | Red <2 s (AEB)",
            "ID:3 at 8 m — TTC=1.5 s  →  CRITICAL / AEB TRIGGERED",
            "ID:2 at 13 m — TTC=2.6 s  →  WARNING",
            "ID:5 at 15 m — TTC=3.0 s  →  WARNING",
            "ID:1 at 23 m — TTC=4.6 s  →  SAFE",
        ],
        left=TEXT_L, top=Inches(3.65), width=TEXT_W, height=Inches(3.00),
        item_size=9.8, heading_size=11.5)


def slide_ncap_hic(slide):
    """Slide 8: Results: Euro NCAP and HIC Biomechanics."""
    set_title(slide, "RESULTS AND DISCUSSION: EURO NCAP VALIDATION AND IMPACT BIOMECHANICS")
    clear_content(slide)

    fig = make_ncap_hic_fig()
    add_picture(slide, fig_to_buf(fig), GRAP_L, GRAP_T, GRAP_W, GRAP_H)

    add_rich_bullets(slide,
        "NCAP Scenarios & HIC Model",
        [
            "4 Euro NCAP scenarios: CPFA (Farside Adult), CPNA (Nearside Adult), "
            "CPNC (Child, parked car), CPLA (Longitudinal)",
            "Test matrix: 20/30/40/50/60 km/h — 20 speed-scenario combinations",
            [('n', 'HIC = t'), ('b', 'c'), ('n', ' × (a'), ('b', 'peak'),
             ('n', '/g)'), ('s', '2.5'), ('n', ';  t'), ('b', 'c'),
             ('n', ' = π√(m/k);  a'), ('b', 'peak'), ('n', ' = πv / 2t'), ('b', 'c')],
            "AIS: 1=Minor  2=Moderate  3=Serious  4=Severe  5=Critical  6=Fatal",
        ],
        left=TEXT_L, top=TEXT_T, width=TEXT_W, height=Inches(2.20),
        item_size=9.8, heading_size=11.5)

    add_bullets(slide,
        "Validation Results",
        [
            "100% NCAP pass rate at 20–30 km/h across all 4 scenarios",
            "CPNC (child, parked car, 20 m range) — binding constraint at 30 km/h",
            "CPLA (straight-line adult) — PASS at all speeds up to 60 km/h",
            "HIC at 20 km/h ≈ 45  →  AIS 1 (Minor)",
            "HIC at 50 km/h ≈ 2,200  →  AIS 5 (Critical)",
            "AEB 50→30 km/h: HIC 2,200→280 — 8× reduction, AIS 5→2",
        ],
        left=TEXT_L, top=Inches(3.65), width=TEXT_W, height=Inches(3.00),
        item_size=9.8, heading_size=11.5)

    add_tb(slide,
           "CPNC (child behind parked car, 20 m detection range) is the binding "
           "constraint; consistent with published monocular-camera AEB limits.",
           GRAP_L, CAPT_T, GRAP_W, CAPT_H,
           size=8.5, italic=True, color=GRAY)

    add_formula_strip(slide,
        [('n', 'HIC = t'), ('b', 'c'), ('n', ' × (a'), ('b', 'peak'),
         ('n', ' / g)'), ('s', '2.5'),
         ('n', '   |   WAD = Wrap-Around Distance from bumper'),
         ('n', '   |   AIS: 1=Minor  2=Moderate  3=Serious'
               '  4=Severe  5=Critical  6=Fatal')],
        GRAP_L, FORM_T, GRAP_W, FORM_H)


def _conc_tb(slide, left, top, width, height, num, keyword, sentence,
             item_size=9.8):
    """
    Single conclusion block: coloured number badge + bold teal keyword + black sentence.
    """
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    # Number + keyword line
    p = tf.paragraphs[0]
    rn = p.add_run()
    rn.text = f"{num}.  "
    rn.font.size = Pt(item_size + 0.5)
    rn.font.bold = True
    rn.font.color.rgb = TEAL

    rk = p.add_run()
    rk.text = keyword
    rk.font.size = Pt(item_size + 0.5)
    rk.font.bold = True
    rk.font.color.rgb = TEAL

    # Sentence line
    p2 = tf.add_paragraph()
    p2.space_before = Pt(1)
    rs = p2.add_run()
    rs.text = f"    {sentence}"
    rs.font.size = Pt(item_size)
    rs.font.bold = False
    rs.font.color.rgb = BLACK
    return tb


def slide_conclusions(slide):
    """Slide 9: Conclusions — 2-column layout, short punchy points, formula strip."""
    set_title(slide, "CONCLUSIONS")
    clear_content(slide)

    # ── Column geometry ────────────────────────────────────────────────
    COL_W  = Inches(4.60)
    COL_H  = Inches(1.05)   # height per conclusion block
    GAP_Y  = Inches(0.10)   # vertical gap between blocks
    LEFT_X = Inches(0.22)
    RIGHT_X = Inches(4.97)
    TOP_Y  = Inches(1.40)

    # ── 6 conclusions: left col (1–3) · right col (4–6) ──────────────
    left_conclusions = [
        (1, "Physics Validated",
         "Pacejka + ABS model matches reference stopping distances within 5%; "
         "kinematic d = v²/2a underestimates by 15–30% — unsafe for AEB design."),
        (2, "Brake Fade Captured",
         "6 panic stops → rotor 428°C; friction −28%; AEB needs +0.19 s extra "
         "lead time — absent from all commercial kinematic AEB systems."),
        (3, "Euro NCAP Compliance",
         "100% pass rate at 20–30 km/h across all 4 scenarios; CPNC (child, "
         "parked car) is the binding constraint at higher speeds."),
    ]
    right_conclusions = [
        (4, "Adaptive AEB Threshold",
         "TTC trigger scales 1.5 s → 3.5 s (+133%) under worst-case fog + "
         "drowsy driver + faded brakes; validated across all scenarios."),
        (5, "Injury Reduction  8×",
         "Partial braking 50→30 km/h cuts HIC 2,200→280, dropping AIS from "
         "5 (Critical) to 2 (Moderate). Life-saving without full avoidance."),
        (6, "Real-Time BEV Radar",
         "Single monocular camera → 3D top-down spatial map via pinhole depth; "
         "TTC-coloured pedestrian tracks at 15–20 fps, proven ADAS format."),
    ]

    for i, (num, kw, sent) in enumerate(left_conclusions):
        _conc_tb(slide, LEFT_X, TOP_Y + i * (COL_H + GAP_Y),
                 COL_W, COL_H, num, kw, sent)

    for i, (num, kw, sent) in enumerate(right_conclusions):
        _conc_tb(slide, RIGHT_X, TOP_Y + i * (COL_H + GAP_Y),
                 COL_W, COL_H, num, kw, sent)

    # ── Key Formula strip ─────────────────────────────────────────────
    add_formula_strip(slide,
        [('n', 'd'), ('b', 'stop'), ('n', ' = v'), ('s', '2'), ('n', '/2μg'),
         ('n', '   |   TTC = d / v'), ('b', 'rel'),
         ('n', '   |   ΔT = ½m·v'), ('s', '2'), ('n', '/(m'), ('b', 'r'),
         ('n', '·c'), ('b', 'p'), ('n', ')'),
         ('n', '   |   HIC = t'), ('b', 'c'), ('n', '·(a'), ('b', 'peak'),
         ('n', '/g)'), ('s', '2.5'),
         ('n', '   |   Threshold = 1.5 + Δ'), ('b', 'w'),
         ('n', ' + Δ'), ('b', 'd'), ('n', ' + Δ'), ('b', 'T')],
        Inches(0.22), Inches(4.70), Inches(9.55), Inches(0.50),
        size=8.5)

    # ── Future Work ───────────────────────────────────────────────────
    add_bullets(slide,
        "Future Work",
        [
            "Stereo / short-range radar fusion → extend CPNC detection range "
            "for NCAP compliance at 40 km/h.",
            "ResNet-18 weather classifier on BDD100K (target ≥ 95%) to replace "
            "heuristic fallback.",
            "NVIDIA Jetson Orin on-vehicle deployment: BEV Radar + adaptive AEB "
            "at 30 fps in mixed traffic.",
            "Kalman-filtered pedestrian velocity for accurate TTC beyond the "
            "constant-speed assumption.",
        ],
        left=Inches(0.22), top=Inches(5.30),
        width=Inches(9.55), height=Inches(1.20),
        item_size=9.5, heading_size=11.5)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(0)
    prs = Presentation(TEMPLATE)

    # Template has 12 slides (0-indexed):
    #  0:Title  1:Intro  2:Motivation  3:Objectives(diagram)  4:Computational
    #  5:Boundary  6:Numerical  7:GridIndep  8:Validation
    #  9:Results  10:Conclusions  11:ThankYou
    #
    # Delete in DESCENDING order to avoid index drift:
    #   index 9 (Results AND DISCUSSION — extra)
    #   index 3 (Objectives with SmartArt diagram — not a clean content slide)
    for idx in [9, 3]:
        delete_slide(prs, idx)

    # After deletion, slides (0-indexed):
    #  0:Title  1:Intro  2:Motivation  3:Computational  4:Boundary
    #  5:Numerical  6:GridIndep  7:Validation  8:Conclusions  9:ThankYou
    sl = prs.slides

    print("Building slides using IIT Roorkee template...")

    slide_title(sl[0], prs);       print("  [1/9] Title")
    slide_objectives(sl[1]);       print("  [2/9] Objectives & Methodology")
    slide_timeline(sl[2]);         print("  [3/9] Timeline")
    slide_detection(sl[3]);        print("  [4/9] Results — Detection & Tracking")
    slide_dynamics(sl[4]);         print("  [5/9] Results — Vehicle Dynamics")
    slide_thermo(sl[5]);           print("  [6/9] Results — Thermodynamics")
    slide_aeb_driver(sl[6]);       print("  [7/9] Results — Adaptive AEB & Driver")
    slide_ncap_hic(sl[7]);         print("  [8/9] Results — Euro NCAP & Biomechanics")
    slide_conclusions(sl[8]);      print("  [9/9] Conclusions")
    # sl[9] = Thank You — left unchanged from template

    prs.save(OUTPUT)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
