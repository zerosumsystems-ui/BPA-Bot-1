"""Generate PDF with visual diagrams of each fade setup: old (broken) vs new (fixed)."""
from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, "Fade Setup Direction Fix - Visual Guide", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(100, 100, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def diagram(self, lines):
        """Draw a monospace diagram block with background."""
        self.set_fill_color(245, 245, 250)
        self.set_font("Courier", "", 8)
        self.set_text_color(40, 40, 40)
        x0 = self.get_x() + 3
        block = "\n".join(lines)
        self.set_x(x0)
        self.multi_cell(184, 3.8, block, fill=True)
        self.ln(2)

    def label_box(self, text, color):
        """Small colored label."""
        self.set_font("Helvetica", "B", 9)
        r, g, b = color
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.cell(50, 6, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(50, 50, 50)
        self.ln(1)

    def draw_setup(self, title, pattern_desc, pattern_lines,
                   old_dir, old_entry, old_stop, old_target, old_lines,
                   new_dir, new_entry, new_stop, new_target, new_lines,
                   why_old_broke, tier_info):
        """Draw a full setup page: pattern + old + new."""

        self.section(title)
        self.body_text(pattern_desc)

        self.subsection("The Pattern")
        self.diagram(pattern_lines)

        self.label_box("BEFORE (Broken)", (190, 50, 50))
        self.body_text(
            f"Direction: {old_dir}    Entry: {old_entry}    "
            f"Stop: {old_stop}    Target: {old_target}\n"
            f"Bug: {why_old_broke}"
        )
        self.diagram(old_lines)

        self.label_box("AFTER (Fixed)", (40, 120, 60))
        self.body_text(
            f"Direction: {new_dir}    Entry: {new_entry}    "
            f"Stop: {new_stop}    Target: {new_target}\n"
            f"{tier_info}"
        )
        self.diagram(new_lines)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=18)

# =====================================================================
# OVERVIEW PAGE
# =====================================================================
pdf.add_page()
pdf.section("Overview: What Changed")
pdf.body_text(
    "The backtester inferred trade direction from setup names using keyword "
    "substring matching. Keywords like \"Low\" and \"High\" matched inside words "
    "like \"Lower\" and \"Higher\", corrupting the direction for 9 of 12 fade setups.\n\n"
    "The fade logic then inverted the already-wrong direction, producing entries "
    "in the wrong direction with garbled stop/target levels.\n\n"
    "Fix: (1) Use the explicit direction field from the detector when available. "
    "(2) Remove \"High\" and \"Low\" from keyword fallback lists."
)

pdf.section("The 7 Active Fade Setups")
pdf.set_font("Courier", "", 9)
pdf.set_text_color(40, 40, 40)
pdf.multi_cell(0, 5,
    "Setup                                     Tier   R:R\n"
    "----------------------------------------------------\n"
    "Fade Lower Low Double Bottom               A     1.0\n"
    "Fade Higher High Double Top                 A     1.0\n"
    "Fade Consecutive Sell Climaxes (Reversal)   A     1.0\n"
    "Fade Consecutive Buy Climaxes (Reversal)    A     1.0\n"
    "Fade Bull Breakout Pullback                 A     1.0\n"
    "Fade Bear Stairs Reversal (3rd/4th Push)    A     1.0\n"
    "Fade Exhaustive Bull Climax at MM           B     0.75"
)
pdf.ln(3)
pdf.body_text("Each setup is shown on the following pages with a chart diagram, "
              "the broken entry geometry, and the corrected entry geometry.")

# =====================================================================
# 1. FADE LOWER LOW DOUBLE BOTTOM
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="1. Fade Lower Low Double Bottom",
    pattern_desc=(
        "Pattern: Two swing lows at nearly the same price (within 0.3%), "
        "where the second low undercuts the first. This is a weak double bottom. "
        "Original setup buys above signal bar (Long). Fade sells (Short), "
        "betting the pattern fails and price continues down."
    ),
    pattern_lines=[
        "  Price",
        "    |",
        "    |     /\\              /\\",
        "    |    /  \\            /  \\",
        "    |   /    \\    /\\   /    \\",
        "    |  /      \\  /  \\ /      \\",
        "    | /        \\/    \\/        \\",
        "    |          L1    L2         \\",
        "    |                (lower)     \\",
        "    +-----------------------------------> Time",
        "",
        "    L1 = First swing low",
        "    L2 = Second swing low (lower than L1)",
    ],
    old_dir="Long  (WRONG)",
    old_entry="signal_bar.low - 0.01",
    old_stop="scalp target (above)",
    old_target="signal_bar.high + 0.01",
    old_lines=[
        "  Keywords scored: \"Low\" x2 = bear(2) > \"Bottom\" x1 = bull(1)",
        "  --> direction = Short (WRONG, should be Long)",
        "  --> Al Brooks levels calculated for Short entry",
        "  --> Fade flips Short to Long",
        "",
        "    |                                        STOP (old target)",
        "    |    ........................................x...............",
        "    |                                     ^",
        "    |     /\\              /\\          |  (buying here??)",
        "    |    /  \\            /  \\    ENTRY x  <-- signal.low - 0.01",
        "    |   /    \\    /\\   /    \\         |",
        "    |  /      \\  /  \\ /      \\       v",
        "    | /        \\/    \\/        \\ TARGET (old stop, below)",
        "    |          L1    L2         x............................",
        "",
        "  Result: Buying at the LOW with target below = guaranteed loser",
        "  But the garbled geometry made it look like 98% WR",
    ],
    new_dir="Short  (CORRECT)",
    new_entry="signal_bar.high + 0.01",
    new_stop="original target (above)",
    new_target="original stop (below)",
    new_lines=[
        "  Keywords scored: \"Bottom\" x1 = bull(1) > bear(0)",
        "  --> direction = Long (CORRECT)",
        "  --> Al Brooks levels: entry = signal.high + 0.01",
        "  --> Fade flips Long to Short, swaps stop & target",
        "",
        "    |                                     STOP (old 1:1 target)",
        "    |    ........................................x...............",
        "    |                                     |",
        "    |     /\\              /\\          ENTRY x  <-- signal.high + 0.01",
        "    |    /  \\            /  \\              |  (selling here)",
        "    |   /    \\    /\\   /    \\             v",
        "    |  /      \\  /  \\ /      \\",
        "    | /        \\/    \\/        \\ TARGET x  (old stop = signal.low)",
        "    |          L1    L2         \\......x........................",
        "",
        "  Selling high, target low, stop above = proper 1:1 fade",
    ],
    why_old_broke="\"Low\" matched twice via substring (\"Lower\" + \"Low\"), making bear > bull.",
    tier_info="Tier A fade at 1:1 R:R"
)

# =====================================================================
# 2. FADE HIGHER HIGH DOUBLE TOP
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="2. Fade Higher High Double Top",
    pattern_desc=(
        "Pattern: Two swing highs at nearly the same price, where the second "
        "high exceeds the first. A weak double top. Original setup sells below "
        "signal bar (Short). Fade buys (Long), betting the top fails to hold "
        "and price continues up."
    ),
    pattern_lines=[
        "  Price",
        "    |          H1    H2",
        "    |                (higher)",
        "    | \\        /\\    /\\        /",
        "    |  \\      /  \\  /  \\      /",
        "    |   \\    /    \\/    \\    /",
        "    |    \\  /            \\  /",
        "    |     \\/              \\/",
        "    |",
        "    +-----------------------------------> Time",
        "",
        "    H1 = First swing high",
        "    H2 = Second swing high (higher than H1)",
    ],
    old_dir="Short  (WRONG)",
    old_entry="signal_bar.high + 0.01",
    old_stop="scalp target (below)",
    old_target="signal_bar.low - 0.01",
    old_lines=[
        "  Keywords scored: \"High\" x2 = bull(2) > \"Top\" x1 = bear(1)",
        "  --> direction = Long (WRONG, should be Short)",
        "  --> Al Brooks levels calculated for Long entry",
        "  --> Fade flips Long to Short",
        "",
        "    |          H1    H2    TARGET (old stop, above)",
        "    |    ................x...............................",
        "    |                     ^",
        "    | \\        /\\    /\\   ENTRY x <-- signal.high + 0.01",
        "    |  \\      /  \\  /  \\      |  (selling at top??)",
        "    |   \\    /    \\/    \\     v",
        "    |    \\  /            \\ STOP (old target, below)",
        "    |     \\/              x..................................",
        "",
        "  Selling at the high with stop below & target above = inverted",
    ],
    new_dir="Long  (CORRECT)",
    new_entry="signal_bar.low - 0.01",
    new_stop="original target (below)",
    new_target="original stop (above)",
    new_lines=[
        "  Keywords scored: \"Top\" x1 = bear(1) > bull(0)",
        "  --> direction = Short (CORRECT)",
        "  --> Al Brooks levels: entry = signal.low - 0.01",
        "  --> Fade flips Short to Long, swaps stop & target",
        "",
        "    |          H1    H2         TARGET (old stop = signal.high)",
        "    |    .......................x............................",
        "    |                           ^",
        "    | \\        /\\    /\\         |",
        "    |  \\      /  \\  /  \\  ENTRY x <-- signal.low - 0.01",
        "    |   \\    /    \\/    \\       |  (buying here)",
        "    |    \\  /            \\     v",
        "    |     \\/           STOP x  (old target, 1:1 below)",
        "    |    ..................x............................",
        "",
        "  Buying low, target high, stop below = proper 1:1 fade",
    ],
    why_old_broke="\"High\" matched twice via substring (\"Higher\" + \"High\"), making bull > bear.",
    tier_info="Tier A fade at 1:1 R:R"
)

# =====================================================================
# 3. FADE CONSECUTIVE SELL CLIMAXES (REVERSAL)
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="3. Fade Consecutive Sell Climaxes (Reversal)",
    pattern_desc=(
        "Pattern: 3+ massive bear bars (range > 1.5x avg) in 10 bars, then a bull "
        "reversal bar (close > open, close > midpoint). Original buys the reversal (Long). "
        "Fade sells (Short), betting the reversal bar fails and selling resumes."
    ),
    pattern_lines=[
        "  Price",
        "    |\\",
        "    | \\",
        "    |  |  CLIMAX 1 (huge bear bar)",
        "    |  |",
        "    |   \\",
        "    |    |  CLIMAX 2",
        "    |    |",
        "    |     \\",
        "    |      |  CLIMAX 3",
        "    |      |_",
        "    |        | <- Bull reversal bar (close > mid)",
        "    +-----------------------------------> Time",
        "",
        "    Original: Buy the reversal (direction = +1, Long)",
        "    Fade: Sell, betting the reversal fails",
    ],
    old_dir="Long  (WRONG)",
    old_entry="signal.low - 0.01",
    old_stop="scalp target (above)",
    old_target="signal.high + 0.01",
    old_lines=[
        "  Detector provides: direction = +1 (Long)",
        "  BUT keyword scorer sees \"Sell\" --> bear=1, bull=0 --> Short",
        "  (Keyword overrides the explicit detector direction!)",
        "  --> Levels calculated for Short, then fade flips to Long",
        "",
        "    |      |  CLIMAX 3               STOP (above)",
        "    |      |_                     ...x...............",
        "    |        |  reversal bar   ENTRY x  signal.low - 0.01",
        "    |        |                    ...x...............",
        "    |                            TARGET (below)",
        "",
        "  Buying at the low of the reversal bar -- wrong geometry",
    ],
    new_dir="Short  (CORRECT)",
    new_entry="signal.high + 0.01",
    new_stop="original target (above)",
    new_target="original stop (below)",
    new_lines=[
        "  Detector provides: direction = +1 (Long) -- now USED DIRECTLY",
        "  --> Levels calculated for Long (entry = signal.high + 0.01)",
        "  --> Fade flips Long to Short, swaps stop & target",
        "",
        "    |      |  CLIMAX 3              STOP (old target, above)",
        "    |      |_                    ...x.....................",
        "    |        |  reversal   ENTRY x  signal.high + 0.01",
        "    |        |                  |  (selling the failed reversal)",
        "    |                           v",
        "    |                      TARGET x  (old stop = signal.low)",
        "    |                       ...x.....................",
        "",
        "  Selling above reversal bar, target = old stop below = 1:1 fade",
    ],
    why_old_broke="Keyword scorer ignored detector's direction=+1; \"Sell\" in name made it Short.",
    tier_info="Tier A fade at 1:1 R:R"
)

# =====================================================================
# 4. FADE CONSECUTIVE BUY CLIMAXES (REVERSAL)
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="4. Fade Consecutive Buy Climaxes (Reversal)",
    pattern_desc=(
        "Pattern: 3+ massive bull bars in 10 bars, then a bear reversal bar "
        "(close < open, close < midpoint). Original sells the reversal (Short). "
        "Fade buys (Long), betting the reversal fails and buying resumes."
    ),
    pattern_lines=[
        "  Price",
        "    |                         _",
        "    |                        | | <- Bear reversal bar",
        "    |                      /  |",
        "    |                    |   CLIMAX 3",
        "    |                    |",
        "    |                  /",
        "    |                |   CLIMAX 2",
        "    |                |",
        "    |              /",
        "    |            |   CLIMAX 1 (huge bull bar)",
        "    |            |",
        "    +-----------------------------------> Time",
        "",
        "    Original: Sell the reversal (direction = -1, Short)",
        "    Fade: Buy, betting the reversal fails",
    ],
    old_dir="Short  (WRONG)",
    old_entry="signal.high + 0.01",
    old_stop="scalp target (below)",
    old_target="signal.low - 0.01",
    old_lines=[
        "  Detector provides: direction = -1 (Short)",
        "  BUT keyword scorer sees \"Buy\" --> bull=1, bear=0 --> Long",
        "  --> Levels calculated for Long, then fade flips to Short",
        "",
        "    |                       _",
        "    |                      | |    TARGET (above) ???",
        "    |                      | | ...x..................",
        "    |                        ENTRY x  signal.high + 0.01",
        "    |                          ...x..................",
        "    |                        STOP (below)",
        "",
        "  Selling at the high with target ABOVE = inverted geometry",
    ],
    new_dir="Long  (CORRECT)",
    new_entry="signal.low - 0.01",
    new_stop="original target (below)",
    new_target="original stop (above)",
    new_lines=[
        "  Detector provides: direction = -1 (Short) -- now USED DIRECTLY",
        "  --> Levels calculated for Short (entry = signal.low - 0.01)",
        "  --> Fade flips Short to Long, swaps stop & target",
        "",
        "    |                       _     TARGET (old stop = signal.high)",
        "    |                      | | ...x....................",
        "    |                      | |  ^",
        "    |                      |    |",
        "    |                   ENTRY x  signal.low - 0.01",
        "    |                         |  (buying the failed reversal)",
        "    |                         v",
        "    |                    STOP x  (old target, 1:1 below)",
        "    |                     ...x....................",
        "",
        "  Buying below reversal bar, target = old stop above = 1:1 fade",
    ],
    why_old_broke="Keyword scorer ignored detector's direction=-1; \"Buy\" in name made it Long.",
    tier_info="Tier A fade at 1:1 R:R"
)

# =====================================================================
# 5. FADE BULL BREAKOUT PULLBACK
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="5. Fade Bull Breakout Pullback",
    pattern_desc=(
        "Pattern: Strong bull breakout above a 5-bar base, price pulls back to "
        "test the breakout level, signal bar closes above midpoint. Original buys "
        "the pullback (Long). Fade sells (Short), betting the pullback test fails "
        "and price falls back into the range."
    ),
    pattern_lines=[
        "  Price",
        "    |                    /\\",
        "    |                   /  \\   <- pullback",
        "    |          BO bar /     \\    /",
        "    |  ------+------/   test \\_/  <- signal bar",
        "    |  |base |base |.........|........  base top",
        "    |  | range     |",
        "    |  |           |",
        "    |  ------+------",
        "    +-----------------------------------> Time",
        "",
        "    BO bar breaks above base, then price retests base top",
        "    Original: Buy the test (direction = +1, Long)",
    ],
    old_dir="Short  (WRONG)",
    old_entry="signal.high + 0.01",
    old_stop="old target (above)",
    old_target="signal.low - 0.01",
    old_lines=[
        "  Detector provides: direction = +1 (Long)",
        "  Keyword scorer: \"Bull\" = bull(1), \"Break\" = 0 --> Long (correct!)",
        "  ** This setup was actually scored CORRECTLY by keywords **",
        "  ** But let's verify the fade geometry is right: **",
        "",
        "    --> direction = Long (correct)",
        "    --> Levels for Long: entry = signal.high + 0.01",
        "    --> Fade flips to Short, swaps stop & target",
        "",
        "  This one was ALREADY WORKING correctly.",
    ],
    new_dir="Short  (CORRECT)",
    new_entry="signal.high + 0.01",
    new_stop="original target (above)",
    new_target="original stop (below)",
    new_lines=[
        "  Detector provides: direction = +1 --> now used directly",
        "  Result is SAME as before (keywords also scored it right)",
        "",
        "    |                    /\\        STOP (old target, above)",
        "    |                   /  \\ ........x........................",
        "    |          BO bar /     \\  ENTRY x  signal.high + 0.01",
        "    |  ------+------/   test \\_/ |  (selling the failed test)",
        "    |  |base |base |..........|.v....  base top",
        "    |  | range     |     TARGET x  (old stop = signal.low)",
        "    |  |           |",
        "    |  ------+------",
        "",
        "  No change needed -- keywords happened to work here.",
        "  Now uses explicit direction=+1 for reliability.",
    ],
    why_old_broke="Keywords got this one right (\"Bull\"=1 > bear=0). No change in behavior.",
    tier_info="Tier A fade at 1:1 R:R (behavior unchanged, source of truth improved)"
)

# =====================================================================
# 6. FADE BEAR STAIRS REVERSAL (3RD/4TH PUSH)
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="6. Fade Bear Stairs Reversal (3rd/4th Push)",
    pattern_desc=(
        "Pattern: 3+ descending pushes (stair-step lower lows with overlapping bars), "
        "then a strong bull reversal bar. Original buys the reversal (Long). "
        "Fade sells (Short), betting the 'exhaustion' is actually continuation."
    ),
    pattern_lines=[
        "  Price",
        "    |\\    push 1",
        "    | \\  /\\",
        "    |  \\/   \\    push 2",
        "    |        \\  /\\",
        "    |         \\/   \\    push 3",
        "    |               \\  /\\",
        "    |                \\/   \\",
        "    |                      |_| <- bull reversal bar",
        "    +-----------------------------------> Time",
        "",
        "    3 descending pushes, then a reversal",
        "    Original: Buy reversal (direction = +1, Long)",
    ],
    old_dir="Long  (WRONG)",
    old_entry="signal.low - 0.01",
    old_stop="scalp target (above)",
    old_target="signal.high + 0.01",
    old_lines=[
        "  Detector provides: direction = +1 (Long)",
        "  Keyword scorer: \"Bear\" = bear(1), bull=0 --> Short (WRONG)",
        "  --> Levels for Short, then fade flips to Long",
        "",
        "    |                      |_|         STOP (old target, above)",
        "    |                      reversal ...x........................",
        "    |                       ENTRY x  signal.low - 0.01",
        "    |                            |",
        "    |                       TARGET x  (old stop, below)",
        "    |                          ...x........................",
        "",
        "  Buying at the bottom with target below = inverted geometry",
    ],
    new_dir="Short  (CORRECT)",
    new_entry="signal.high + 0.01",
    new_stop="original target (above)",
    new_target="original stop (below)",
    new_lines=[
        "  Detector provides: direction = +1 --> now USED DIRECTLY",
        "  --> Levels for Long: entry = signal.high + 0.01",
        "  --> Fade flips Long to Short, swaps stop & target",
        "",
        "    |                                  STOP (old target, above)",
        "    |                               ...x........................",
        "    |                      |_| ENTRY x  signal.high + 0.01",
        "    |                      rev     |  (selling the failed reversal)",
        "    |                              v",
        "    |                         TARGET x  (old stop = signal.low)",
        "    |                           ...x........................",
        "",
        "  Selling above reversal, target at old stop below = 1:1 fade",
    ],
    why_old_broke="\"Bear\" in name scored bear=1 > bull=0, making it Short. Detector says Long.",
    tier_info="Tier A fade at 1:1 R:R"
)

# =====================================================================
# 7. FADE EXHAUSTIVE BULL CLIMAX AT MM
# =====================================================================
pdf.add_page()
pdf.draw_setup(
    title="7. Fade Exhaustive Bull Climax at MM",
    pattern_desc=(
        "Pattern: Price hits a measured move target (spike doubled), prints a "
        "bar 2.5x larger than the channel average, closes below its midpoint. "
        "Original sells the exhaustion (Short). Fade buys (Long), betting the "
        "\"exhaustion\" was actually a breakout continuation."
    ),
    pattern_lines=[
        "  Price",
        "    |                              |  <- HUGE exhaustion bar",
        "    |            MM target ........|.......  (spike_high + spike)",
        "    |                             /|",
        "    |                 channel   /  |  close below midpoint",
        "    |                        /",
        "    |   spike_high ......../.............",
        "    |              |     /",
        "    |    initial   |   /",
        "    |     spike    | /",
        "    |   spike_low  |/.....................",
        "    +-----------------------------------> Time",
        "",
        "    Original: Sell the exhaustion (direction = -1, Short)",
        "    Fade: Buy, betting the breakout continues",
    ],
    old_dir="Short  (WRONG)",
    old_entry="signal.high + 0.01",
    old_stop="old target (below)",
    old_target="signal.low - 0.01",
    old_lines=[
        "  Detector provides: direction = -1 (Short)",
        "  Keyword scorer: \"Bull\" = bull(1), bear=0 --> Long (WRONG)",
        "  --> Levels for Long, then fade flips to Short",
        "",
        "    |                              |    TARGET (old stop, above)",
        "    |                              |  ..x.........................",
        "    |                        ENTRY x  signal.high + 0.01",
        "    |                              |  (selling at top??)",
        "    |                              |",
        "    |                         STOP x  (old target, below)",
        "    |                           ..x.........................",
        "",
        "  Selling at the high with STOP below TARGET = inverted risk",
    ],
    new_dir="Long  (CORRECT)",
    new_entry="signal.low - 0.01",
    new_stop="original target (below)",
    new_target="original stop (above)",
    new_lines=[
        "  Detector provides: direction = -1 --> now USED DIRECTLY",
        "  --> Levels for Short: entry = signal.low - 0.01",
        "  --> Fade flips Short to Long, swaps stop & target",
        "",
        "    |                              |   TARGET (old stop = signal.high)",
        "    |                              | ..x.........................",
        "    |                              |  ^",
        "    |                              |  |",
        "    |                        ENTRY x  signal.low - 0.01",
        "    |                              |  (buying the failed exhaustion)",
        "    |                              v",
        "    |                         STOP x  (old target, 0.75:1 below)",
        "    |                          ..x.........................",
        "",
        "  Buying low, target at old stop above = 0.75:1 fade (Tier B)",
    ],
    why_old_broke="\"Bull\" in name scored bull=1 > bear=0, making it Long. Detector says Short.",
    tier_info="Tier B fade at 0.75:1 R:R"
)

# =====================================================================
# SUMMARY PAGE
# =====================================================================
pdf.add_page()
pdf.section("Summary of Direction Changes")

pdf.ln(2)
pdf.set_font("Courier", "", 8)
pdf.set_text_color(40, 40, 40)
pdf.multi_cell(0, 4.2,
    "Setup                                  Old Dir   New Dir   Changed?\n"
    "-------------------------------------------------------------------\n"
    "1. Fade Lower Low Double Bottom        Long      Short     YES\n"
    "   (\"Low\" x2 corrupted score)\n"
    "\n"
    "2. Fade Higher High Double Top         Short     Long      YES\n"
    "   (\"High\" x2 corrupted score)\n"
    "\n"
    "3. Fade Consec. Sell Climaxes (Rev)    Long      Short     YES\n"
    "   (\"Sell\" made it Short, detector says Long)\n"
    "\n"
    "4. Fade Consec. Buy Climaxes (Rev)     Short     Long      YES\n"
    "   (\"Buy\" made it Long, detector says Short)\n"
    "\n"
    "5. Fade Bull Breakout Pullback         Short     Short     NO\n"
    "   (Keywords got it right, now uses explicit dir)\n"
    "\n"
    "6. Fade Bear Stairs Rev (3rd/4th)      Long      Short     YES\n"
    "   (\"Bear\" made it Short, detector says Long)\n"
    "\n"
    "7. Fade Exhaustive Bull Climax at MM   Short     Long      YES\n"
    "   (\"Bull\" made it Long, detector says Short)\n"
)

pdf.ln(3)
pdf.section("Impact")
pdf.body_text(
    "6 of 7 active fade setups had their direction fixed. "
    "All backtest results for these setups should be re-run to get "
    "accurate win rates.\n\n"
    "The original (unfaded) setups had ~20% WR at 1:1. "
    "Properly faded, they should now show realistic performance "
    "with correct entry prices and stop/target geometry."
)

pdf.output("/home/user/BPA-Bot-1/fade_direction_fix.pdf")
print("PDF saved: fade_direction_fix.pdf")
