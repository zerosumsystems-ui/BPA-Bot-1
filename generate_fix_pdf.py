"""Generate PDF explaining the fade direction fix."""
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, "Backtester Fade Direction Fix", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_fill_color(240, 240, 245)
        self.set_font("Courier", "", 9)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.set_x(x + 5)
        self.multi_cell(180, 4.5, text, fill=True)
        self.ln(3)

    def arrow_label(self, text, color=(200, 60, 60)):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 6, f"  >>> {text}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def table_row(self, cols, bold=False, fill=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        self.set_text_color(40, 40, 40)
        if fill:
            self.set_fill_color(220, 230, 245)
        widths = [55, 40, 35, 35, 25]
        for i, col in enumerate(cols):
            self.cell(widths[i], 6, col, border=1, fill=fill, align="C")
        self.ln()


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ============================================================
# PAGE 1: THE BUG
# ============================================================
pdf.section("The Problem: Keyword Scoring Inverts Direction")

pdf.body_text(
    "The backtester infers trade direction from setup names using keyword matching. "
    "It counts bull keywords (Bull, Bottom, Buy, High) vs bear keywords (Bear, Top, Sell, Low) "
    "and whichever scores higher determines Long vs Short."
)

pdf.body_text("Example: \"Lower Low Double Bottom\"")

pdf.code_block(
    "buy_keywords  = [\"Bull\", \"Bottom\", \"Buy\", \"High\"]\n"
    "sell_keywords = [\"Bear\", \"Top\",    \"Sell\", \"Low\"]\n"
    "\n"
    "Bull score:  \"Bottom\" x1                    = 1\n"
    "Bear score:  \"Low\" in \"Lower\" + \"Low\" x1    = 2\n"
    "\n"
    "Result: bear_score (2) > bull_score (1)  -->  direction = Short"
)

pdf.arrow_label("BUG: \"Lower Low Double Bottom\" is a LONG setup, scored as Short", (200, 40, 40))

pdf.ln(3)
pdf.body_text(
    "The substring match on \"Low\" hits twice: once in \"Lower\" and once in \"Low\". "
    "This overpowers the single \"Bottom\" match, flipping the direction."
)

pdf.section("How This Breaks Fade Entries")

pdf.body_text(
    "The fade logic takes the original direction and inverts it. But if the original "
    "direction is already wrong, the fade inverts it back to what SHOULD have been "
    "the original direction -- producing nonsense."
)

pdf.code_block(
    "BEFORE (broken):\n"
    "\n"
    "  1. Detect: \"Lower Low Double Bottom\"  (should be Long)\n"
    "  2. Keyword score: bear=2 > bull=1      --> Short  [WRONG]\n"
    "  3. Calculate levels: Short entry at signal_bar.low - 0.01\n"
    "  4. Fade flip: Short --> Long\n"
    "  5. Swap stop/target\n"
    "\n"
    "  Result: \"Fade Lower Low Double Bottom\"\n"
    "    Direction: Long   (should be Short)\n"
    "    Entry: at the LOW of the signal bar  (should be at HIGH)\n"
    "    Stop/Target: completely inverted geometry\n"
    "    Win rate: ~98%  (artificial -- levels are garbled)"
)

pdf.arrow_label("The fade undoes the keyword bug instead of fading the actual setup", (200, 40, 40))

# ============================================================
# PAGE 2: THE FIX
# ============================================================
pdf.add_page()

pdf.section("The Fix: Use Explicit Direction, Clean Up Keywords")

pdf.body_text(
    "Two changes applied at both backtester code paths (intraday + swing):"
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(30, 100, 50)
pdf.cell(0, 6, "  Change 1: Check for explicit direction from the detector first", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.body_text(
    "Most detection functions already provide direction=1 (Long) or direction=-1 (Short). "
    "The backtester now checks for this field before falling back to keyword scoring."
)

pdf.code_block(
    "raw_dir = setup.get(\"direction\")\n"
    "if raw_dir is not None:\n"
    "    direction = \"Long\" if raw_dir == 1 else \"Short\"\n"
    "else:\n"
    "    # Fall back to keyword scoring..."
)

pdf.body_text("Setups that provide explicit direction (now used directly):")
pdf.code_block(
    "  detect_consecutive_climaxes     -> direction = 1 or -1\n"
    "  detect_exhaustive_climax_at_mm  -> direction = 1 or -1\n"
    "  detect_breakout_pullback        -> direction = 1 or -1\n"
    "  detect_bear_stairs              -> direction = 1\n"
    "  detect_quiet_flag_ma_entries    -> (no direction, uses fallback)\n"
    "  detect_weak_breakout_tests      -> (no direction, uses fallback)\n"
    "  detect_double_bottoms_tops      -> (no direction, uses fallback)"
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(30, 100, 50)
pdf.cell(0, 6, "  Change 2: Remove \"High\" and \"Low\" from keyword fallback", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.body_text(
    "For detectors without an explicit direction field, the keyword fallback "
    "now uses [Bull, Bottom, Buy] and [Bear, Top, Sell] -- without High/Low."
)

pdf.code_block(
    "\"Lower Low Double Bottom\":\n"
    "  Bull score:  \"Bottom\" x1  = 1\n"
    "  Bear score:  (nothing)     = 0\n"
    "  Result: Long  [CORRECT]\n"
    "\n"
    "\"Higher High Double Top\":\n"
    "  Bull score:  (nothing)     = 0\n"
    "  Bear score:  \"Top\" x1     = 1\n"
    "  Result: Short  [CORRECT]"
)

# ============================================================
# PAGE 3: BEFORE vs AFTER TABLE
# ============================================================
pdf.add_page()

pdf.section("Before vs After: All 12 Fade Setups")

pdf.body_text(
    "This table shows the original direction assigned by the keyword scorer (before), "
    "the corrected direction (after), and how the fade flip behaves in each case."
)

pdf.ln(2)

# Table header
pdf.set_font("Helvetica", "B", 8)
pdf.set_fill_color(50, 70, 120)
pdf.set_text_color(255, 255, 255)
hdrs = ["Setup Name", "Old Direction", "Old Fade", "New Direction", "New Fade"]
widths = [65, 30, 30, 30, 30]
for i, h in enumerate(hdrs):
    pdf.cell(widths[i], 7, h, border=1, fill=True, align="C")
pdf.ln()

rows = [
    ("Lower Low Double Bottom",            "Short (WRONG)", "Long",  "Long",  "Short"),
    ("Higher High Double Top",             "Long (WRONG)",  "Short", "Short", "Long"),
    ("Consec. Buy Climaxes (Rev)",         "Long (WRONG)",  "Short", "Short", "Long"),
    ("Consec. Sell Climaxes (Rev)",        "Short (WRONG)", "Long",  "Long",  "Short"),
    ("Exhaustive Bull Climax at MM",       "Long (WRONG)",  "Short", "Short", "Long"),
    ("Exhaustive Bear Climax at MM",       "Short (WRONG)", "Long",  "Long",  "Short"),
    ("Bull Breakout Pullback",             "Long (OK)",     "Short", "Long",  "Short"),
    ("Weak Bull Breakout Test",            "Long (WRONG)",  "Short", "Short", "Long"),
    ("Weak Bear Breakout Test",            "Short (WRONG)", "Long",  "Long",  "Short"),
    ("Quiet Bull Flag at MA",              "Long (OK)",     "Short", "Long",  "Short"),
    ("Quiet Bear Flag at MA",              "Short (OK)",    "Long",  "Short", "Long"),
    ("Bear Stairs Rev (3rd/4th Push)",     "Short (WRONG)", "Long",  "Long",  "Short"),
]

pdf.set_text_color(40, 40, 40)
for idx, (name, old_dir, old_fade, new_dir, new_fade) in enumerate(rows):
    fill = idx % 2 == 0
    if fill:
        pdf.set_fill_color(235, 240, 250)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.cell(widths[0], 6, name, border=1, fill=fill)

    # Color old direction red if WRONG
    if "WRONG" in old_dir:
        pdf.set_text_color(190, 40, 40)
        pdf.set_font("Helvetica", "B", 7.5)
    else:
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 7.5)
    pdf.cell(widths[1], 6, old_dir, border=1, fill=fill, align="C")

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.cell(widths[2], 6, old_fade, border=1, fill=fill, align="C")

    # Color new direction green
    pdf.set_text_color(30, 120, 50)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.cell(widths[3], 6, new_dir, border=1, fill=fill, align="C")
    pdf.cell(widths[4], 6, new_fade, border=1, fill=fill, align="C")
    pdf.set_text_color(40, 40, 40)
    pdf.ln()

pdf.ln(5)
pdf.arrow_label("9 of 12 setups had WRONG original direction before the fix", (200, 40, 40))

pdf.ln(3)
pdf.section("What This Means for Backtest Results")

pdf.body_text(
    "The ~98% win rate on fade setups was artificial. The keyword bug pre-inverted "
    "the direction, and the fade logic inverted it again, accidentally producing "
    "the correct (unfaded) direction with garbled stop/target levels.\n\n"
    "With the fix applied, fade setups will now:\n"
    "  - Enter in the correct (opposite) direction\n"
    "  - Have proper Al Brooks stop/target geometry\n"
    "  - Show realistic win rates reflecting actual fade performance\n\n"
    "The original setups had ~20% WR at 1:1 R:R. Properly faded, "
    "they should show ~80% WR -- but with correct entry prices and risk levels."
)

# Save
pdf.output("/home/user/BPA-Bot-1/fade_direction_fix.pdf")
print("PDF saved to /home/user/BPA-Bot-1/fade_direction_fix.pdf")
