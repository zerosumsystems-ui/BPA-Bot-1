#!/usr/bin/env python3
"""
Generate Expanding Triangle (Megaphone) flowchart PDF.
Brooks Price Action — fade the extremes of a broadening formation.

Al Brooks: "An expanding triangle is a trading range with higher highs and
lower lows. It is a sign of uncertainty. Fade the extremes — when price
tests the upper or lower boundary, look for a reversal bar to enter
against the move."
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages


def draw_box(ax, x, y, text, box_type="process", width=3.2, height=0.55):
    """Draw a styled box with text."""
    colors = {
        "start":    ("#E8EAF6", "#3F51B5"),   # Indigo
        "process":  ("#E3F2FD", "#1976D2"),   # Blue
        "decision": ("#FFF9C4", "#F57F17"),   # Yellow/Orange
        "entry":    ("#C8E6C9", "#2E7D32"),   # Green
        "stop":     ("#FFCDD2", "#C62828"),   # Red
        "warning":  ("#FFF3E0", "#E65100"),   # Deep Orange
        "note":     ("#F3E5F5", "#7B1FA2"),   # Purple
    }
    face, edge = colors.get(box_type, colors["process"])

    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.08", facecolor=face,
        edgecolor=edge, linewidth=2.0, zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#212121', zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#555"):
    """Draw an arrow between two points with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.25,head_length=0.15',
        color=color, linewidth=1.5, zorder=1
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2 + 0.15
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=6.5, color=color,
                ha='left', va='center', fontstyle='italic')


def draw_concept_page(ax):
    """Draw an overview page explaining Expanding Triangles."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, "EXPANDING TRIANGLE (MEGAPHONE) — BPA Concept Overview",
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#1565C0')

    lines = [
        ("What is an Expanding Triangle?", 8.7, True),
        ("A trading range that gets progressively wider over time.", 8.3, False),
        ("Higher highs AND lower lows form a broadening pattern.", 7.9, False),
        ("Also called a Megaphone or Broadening Formation.", 7.5, False),
        ("It reflects growing uncertainty and emotional extremes.", 7.1, False),
        ("", 6.7, False),
        ("Why fade the extremes?", 6.4, True),
        ("Brooks: the expanding edges act as magnets pulling price", 6.0, False),
        ("back to the middle. Each overshoot traps breakout traders", 5.6, False),
        ("who become fuel for the reversal back into the range.", 5.2, False),
        ("The wider the range, the better the reward.", 4.8, False),
        ("", 4.4, False),
        ("How to identify the pattern:", 4.1, True),
        ("1. Find 2+ swing highs — each higher than the last.", 3.7, False),
        ("2. Find 2+ swing lows — each lower than the last.", 3.3, False),
        ("3. The range width is expanding (HH-LL gap growing).", 2.9, False),
        ("4. Not a strong trend — EMA should be relatively flat.", 2.5, False),
        ("", 2.1, False),
        ("Risk Management:", 1.8, True),
        ("Stop: beyond the extreme of the last swing (HH or LL).", 1.4, False),
        ("Target: midpoint of the expanding range or opposite edge.", 1.0, False),
    ]

    for text, y, is_header in lines:
        if is_header:
            ax.text(1.0, y, text, ha='left', va='center',
                    fontsize=9.5, fontweight='bold', color='#1565C0')
        else:
            ax.text(1.2, y, text, ha='left', va='center',
                    fontsize=8, color='#333')


def draw_fade_top_flowchart(ax):
    """Flowchart for fading the top of an Expanding Triangle (short)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.7, "EXPANDING TRIANGLE TOP — FADE SHORT",
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#C62828')

    nodes = [
        (5, 8.8, "START: Scan 20-bar\nlookback for swing points", "start"),
        (5, 7.8, "2+ swing highs found?\nEach higher than the last?", "decision"),
        (5, 6.8, "2+ swing lows found?\nEach lower than the last?", "decision"),
        (5, 5.8, "Range expanding?\n(HH - LL gap widening)", "decision"),
        (5, 4.8, "Current bar near the\nupper edge? (within 0.2% of HH)", "decision"),
        (5, 3.8, "Reversal signal?\nClose < midpoint (bearish close)", "decision"),
        (5, 2.7, "ENTRY: Sell short\nat close or 1 tick below bar low", "entry"),
        (5, 1.6, "Stop: above the new HH extreme\nTarget: range midpoint or LL edge", "stop"),
    ]

    for x, y, text, btype in nodes:
        draw_box(ax, x, y, text, btype)

    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i][0], nodes[i][1] - 0.3
        x2, y2 = nodes[i + 1][0], nodes[i + 1][1] + 0.3
        label = "Yes" if nodes[i][3] == "decision" else ""
        draw_arrow(ax, x1, y1, x2, y2, label)

    # "No" annotations on decision nodes
    no_indices = [1, 2, 3, 4, 5]
    for idx in no_indices:
        x, y = nodes[idx][0], nodes[idx][1]
        ax.annotate("No -> skip", xy=(x + 1.7, y), fontsize=6,
                    color='#888', ha='left', va='center')


def draw_fade_bottom_flowchart(ax):
    """Flowchart for fading the bottom of an Expanding Triangle (long)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.7, "EXPANDING TRIANGLE BOTTOM — FADE LONG",
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#2E7D32')

    nodes = [
        (5, 8.8, "START: Scan 20-bar\nlookback for swing points", "start"),
        (5, 7.8, "2+ swing highs found?\nEach higher than the last?", "decision"),
        (5, 6.8, "2+ swing lows found?\nEach lower than the last?", "decision"),
        (5, 5.8, "Range expanding?\n(HH - LL gap widening)", "decision"),
        (5, 4.8, "Current bar near the\nlower edge? (within 0.2% of LL)", "decision"),
        (5, 3.8, "Reversal signal?\nClose > midpoint (bullish close)", "decision"),
        (5, 2.7, "ENTRY: Buy long\nat close or 1 tick above bar high", "entry"),
        (5, 1.6, "Stop: below the new LL extreme\nTarget: range midpoint or HH edge", "stop"),
    ]

    for x, y, text, btype in nodes:
        draw_box(ax, x, y, text, btype)

    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i][0], nodes[i][1] - 0.3
        x2, y2 = nodes[i + 1][0], nodes[i + 1][1] + 0.3
        label = "Yes" if nodes[i][3] == "decision" else ""
        draw_arrow(ax, x1, y1, x2, y2, label)

    no_indices = [1, 2, 3, 4, 5]
    for idx in no_indices:
        x, y = nodes[idx][0], nodes[idx][1]
        ax.annotate("No -> skip", xy=(x + 1.7, y), fontsize=6,
                    color='#888', ha='left', va='center')


def draw_visual_pattern_page(ax):
    """Draw an ASCII-style visual showing the expanding triangle pattern."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, "EXPANDING TRIANGLE — Visual Pattern",
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#1565C0')

    # Draw expanding triangle shape
    # Upper boundary (rising)
    upper_x = [1.5, 3.5, 5.5, 7.5]
    upper_y = [6.5, 7.0, 7.5, 8.0]
    ax.plot(upper_x, upper_y, '--', color='#C62828', linewidth=2, label='Resistance (rising)')

    # Lower boundary (falling)
    lower_x = [1.5, 3.5, 5.5, 7.5]
    lower_y = [5.5, 5.0, 4.5, 4.0]
    ax.plot(lower_x, lower_y, '--', color='#2E7D32', linewidth=2, label='Support (falling)')

    # Price action zigzag inside the megaphone
    price_x = [1.5, 2.2, 3.0, 3.8, 4.6, 5.4, 6.2, 7.0, 7.5]
    price_y = [6.0, 6.4, 5.2, 6.9, 4.6, 7.4, 4.2, 7.9, 4.1]
    ax.plot(price_x, price_y, '-', color='#1976D2', linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='#1976D2')

    # Annotate swing highs
    for i, (px, py) in enumerate(zip(price_x, price_y)):
        if i in [1, 3, 5, 7]:  # Swing highs
            ax.annotate(f'HH{(i+1)//2}', xy=(px, py + 0.15), fontsize=7,
                       color='#C62828', ha='center', fontweight='bold')
        elif i in [2, 4, 6, 8]:  # Swing lows
            ax.annotate(f'LL{i//2}', xy=(px, py - 0.25), fontsize=7,
                       color='#2E7D32', ha='center', fontweight='bold')

    # Fade arrows at extremes
    ax.annotate('FADE SHORT\nhere', xy=(7.0, 7.9), xytext=(8.5, 8.5),
               fontsize=8, color='#C62828', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#C62828', lw=2),
               ha='center')
    ax.annotate('FADE LONG\nhere', xy=(7.5, 4.1), xytext=(8.8, 3.3),
               fontsize=8, color='#2E7D32', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2),
               ha='center')

    # Midpoint line
    mid_y = [(u + l) / 2 for u, l in zip(upper_y, lower_y)]
    ax.plot(upper_x, mid_y, ':', color='#888', linewidth=1.5)
    ax.text(8.0, mid_y[-1], 'Midpoint\n(target)', fontsize=7, color='#888',
            ha='left', va='center')

    # Legend box
    legend_items = [
        ("Key Rules:", True),
        ("1. Higher highs + lower lows = expanding range", False),
        ("2. Fade the extreme edges, not the middle", False),
        ("3. Reversal bar must close against the move", False),
        ("4. Stop beyond the swing extreme", False),
        ("5. Target: midpoint or opposite edge", False),
        ("6. Best in trading ranges, NOT strong trends", False),
        ("7. Confidence: ~65% (range-bound context)", False),
    ]

    for i, (text, is_header) in enumerate(legend_items):
        y_pos = 2.8 - i * 0.3
        if is_header:
            ax.text(1.0, y_pos, text, fontsize=9, fontweight='bold', color='#1565C0')
        else:
            ax.text(1.2, y_pos, text, fontsize=7.5, color='#333')


def main():
    pdf_path = "BPA_Expanding_Triangle_Flowchart.pdf"

    with PdfPages(pdf_path) as pdf:
        # Page 1: Concept overview
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        draw_concept_page(ax)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 2: Visual pattern diagram
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        draw_visual_pattern_page(ax)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 3: Fade Top flowchart
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        draw_fade_top_flowchart(ax)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 4: Fade Bottom flowchart
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        draw_fade_bottom_flowchart(ax)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

    print(f"Generated: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    main()
