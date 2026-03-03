from fpdf import FPDF


def safe(text):
    """Replace Unicode chars unsupported by built-in Helvetica."""
    return (text
        .replace("\u2014", "--")   # em-dash
        .replace("\u2013", "-")    # en-dash
        .replace("\u2018", "'")    # left single quote
        .replace("\u2019", "'")    # right single quote
        .replace("\u201c", '"')    # left double quote
        .replace("\u201d", '"')    # right double quote
        .replace("\u2022", "*")    # bullet
    )


class BPAPdf(FPDF):
    """iPhone-optimized PDF: larger fonts, generous padding, high contrast."""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, safe("BPA Bot -- Al Brooks Price Action: 13 Profitable Setups"), align="C")
            self.ln(14)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(225, 235, 248)
        self.cell(0, 12, f"  {title}", ln=True, fill=True)
        self.ln(4)

    def sub_heading(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 9, text, ln=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 12)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 7, text)
        self.ln(3)

    def bullet(self, label, value):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(6)
        w_label = self.get_string_width(label + ": ") + 1
        self.cell(w_label, 7, label + ": ", ln=False)
        self.set_font("Helvetica", "", 12)
        self.multi_cell(0, 7, value)
        self.ln(1)

    def tier_badge(self, tier, color_rgb):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(*color_rgb)
        self.set_text_color(255, 255, 255)
        self.cell(55, 8, f"  {tier}", fill=True)
        self.set_text_color(40, 40, 40)
        self.ln(11)

    def divider(self):
        self.set_draw_color(200, 200, 200)
        y = self.get_y()
        self.line(10, y, 200, y)
        self.ln(6)


pdf = BPAPdf()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=22)

# ─── COVER PAGE ───
pdf.add_page()
pdf.ln(40)
pdf.set_font("Helvetica", "B", 32)
pdf.set_text_color(25, 25, 112)
pdf.cell(0, 18, "BPA Bot", align="C", ln=True)
pdf.set_font("Helvetica", "", 18)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 12, "Al Brooks Price Action", align="C", ln=True)
pdf.ln(6)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(40, 40, 40)
pdf.cell(0, 14, "All 13 Profitable Setups", align="C", ln=True)
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, "Entries, Stops, Targets & Order Types", align="C", ln=True)
pdf.ln(18)
pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "A beginner-friendly guide to every setup", align="C", ln=True)
pdf.cell(0, 8, "detected by the BPA indicator.", align="C", ln=True)
pdf.ln(4)
pdf.cell(0, 8, "Designed for TradingView (Pine Script v6).", align="C", ln=True)

# ─── KEY CONCEPTS PAGE ───
pdf.add_page()
pdf.section_title("Key Concepts  --  Read This First")

pdf.sub_heading("What is Price Action?")
pdf.body_text(
    "Price action trading means making decisions based on the "
    "price chart itself -- the actual bars (candles), their highs, "
    "lows, opens, and closes -- rather than relying on external "
    "indicators like RSI or MACD. The idea is simple: the chart "
    "already contains all the information you need."
)

pdf.sub_heading("What is a Signal Bar?")
pdf.body_text(
    "The signal bar is the specific candle that completes a setup "
    "pattern. It tells you: \"the pattern is done, now get ready "
    "to enter.\" You never enter ON the signal bar -- you place "
    "your order to enter on the NEXT bar, 1 tick beyond it."
)

pdf.sub_heading("What is a Stop Order? (Entry Type)")
pdf.body_text(
    "All 13 setups use STOP ORDERS for entry -- not market "
    "orders, not limit orders. This is critical:\n\n"
    "Buy Stop: Your order sits ABOVE the current price. It only "
    "fills if price rises to your level. This confirms buyers are "
    "pushing price higher before you commit.\n\n"
    "Sell Stop: Your order sits BELOW the current price. It only "
    "fills if price drops to your level. This confirms sellers are "
    "pushing price lower before you commit.\n\n"
    "Why stop orders? They force the market to PROVE the move "
    "before you enter. If price never reaches your entry, you "
    "simply don't get filled -- no harm done."
)

pdf.add_page()
pdf.sub_heading("Entry, Stop-Loss, and Target")
pdf.body_text(
    "Every setup calculates three price levels:\n\n"
    "Entry: 1 tick beyond the signal bar (above the high for "
    "longs, below the low for shorts).\n\n"
    "Stop-Loss: The opposite side of the signal bar. If you "
    "bought above the high, your stop is below the low. This is "
    "the \"you were wrong\" level.\n\n"
    "Target: Calculated from the risk (distance from entry to "
    "stop) multiplied by the reward ratio. For example, 1:1 R:R "
    "means your target is the same distance as your stop but in "
    "the profitable direction."
)

pdf.sub_heading("Risk-to-Reward (R:R) Ratios")
pdf.body_text(
    "Each setup has a preset R:R based on its historical "
    "reliability:\n\n"
    "1:1 -- You risk $1 to make $1. Used for the highest-"
    "probability setups.\n\n"
    "0.75:1 -- You risk $1 to make $0.75. Slightly less "
    "ambitious target.\n\n"
    "0.5:1 -- You risk $1 to make $0.50. Scalp trades -- quick "
    "in, quick out."
)

pdf.sub_heading("The Three Tiers")
pdf.body_text(
    "Setups are ranked by reliability:\n\n"
    "Tier A (Best): The highest win-rate setups. Trade these "
    "with confidence.\n\n"
    "Tier B (Standard): Solid setups with good odds but slightly "
    "lower edge.\n\n"
    "Tier C (Scalps): Quick, smaller trades. Take profits fast."
)

# ─── SETUP PAGES ───
setups = [
    {
        "num": "1",
        "name": "Double Bottom (DB)",
        "tier": "Tier B -- Standard",
        "tier_color": (70, 130, 180),
        "direction": "LONG (profit if price goes UP)",
        "rr": "0.75 : 1",
        "order": "Buy Stop",
        "what": (
            "A double bottom forms when price drops to a low, "
            "bounces up, then drops back down to approximately the "
            "SAME low and bounces again. The two lows create a "
            "\"floor\" -- a price level where buyers step in and "
            "defend."
        ),
        "why": (
            "Buyers defended the same price level twice. If "
            "sellers couldn't break through that floor on two "
            "attempts, buyers are likely in control. The second "
            "bounce is your cue to go long."
        ),
        "entry": "Buy Stop 1 tick above the signal bar's high (the bar at the 2nd low).",
        "stop": "1 tick below the signal bar's low.",
        "target": "0.75x the risk distance above your entry.",
    },
    {
        "num": "2",
        "name": "Higher Low Double Bottom (HL DB)",
        "tier": "Tier C -- Scalp",
        "tier_color": (160, 160, 160),
        "direction": "LONG (profit if price goes UP)",
        "rr": "1 : 1",
        "order": "Buy Stop",
        "what": (
            "Similar to a regular double bottom, but the second "
            "low is HIGHER than the first. Price didn't even make "
            "it back down to the original floor -- buyers stepped "
            "in earlier."
        ),
        "why": (
            "Buyers are getting more aggressive. They didn't wait "
            "for the old low to buy -- they bought at a higher "
            "price. This shows increasing demand and is a sign of "
            "strength."
        ),
        "entry": "Buy Stop 1 tick above the signal bar's high.",
        "stop": "1 tick below the signal bar's low.",
        "target": "1x the risk distance above your entry.",
    },
    {
        "num": "3",
        "name": "Fade Lower Low Double Bottom",
        "tier": "Tier A -- Best",
        "tier_color": (46, 139, 87),
        "direction": "SHORT (profit if price goes DOWN)",
        "rr": "1 : 1",
        "order": "Sell Stop",
        "what": (
            "This looks like a double bottom, but the second low "
            "is actually LOWER than the first. The \"floor\" "
            "broke. Most traders see a double bottom and expect a "
            "bounce -- but the lower low tells you sellers won."
        ),
        "why": (
            "The failed double bottom traps long traders who "
            "bought the expected bounce. When price breaks below "
            "the first low, those trapped longs panic-sell, adding "
            "fuel to the move down. You fade (trade against) the "
            "double bottom pattern by going short."
        ),
        "entry": "Sell Stop 1 tick below the signal bar's low.",
        "stop": "1 tick above the signal bar's high.",
        "target": "1x the risk distance below your entry.",
    },
    {
        "num": "4",
        "name": "Double Top (DT)",
        "tier": "Tier B -- Standard",
        "tier_color": (70, 130, 180),
        "direction": "SHORT (profit if price goes DOWN)",
        "rr": "1 : 1",
        "order": "Sell Stop",
        "what": (
            "Price rises to a high, pulls back down, then rallies "
            "back to approximately the SAME high and fails again. "
            "The two highs create a \"ceiling\" -- a price level "
            "where sellers step in and defend."
        ),
        "why": (
            "Sellers defended the same price level twice. If "
            "buyers couldn't break through that ceiling on two "
            "attempts, sellers are likely in control."
        ),
        "entry": "Sell Stop 1 tick below the signal bar's low (the bar at the 2nd high).",
        "stop": "1 tick above the signal bar's high.",
        "target": "1x the risk distance below your entry.",
    },
    {
        "num": "5",
        "name": "Lower High Double Top (LH DT)",
        "tier": "Tier B -- Standard",
        "tier_color": (70, 130, 180),
        "direction": "SHORT (profit if price goes DOWN)",
        "rr": "1 : 1",
        "order": "Sell Stop",
        "what": (
            "Similar to a regular double top, but the second high "
            "is LOWER than the first. Price couldn't even reach "
            "the old ceiling -- sellers are stepping in earlier and "
            "at lower prices."
        ),
        "why": (
            "Sellers are getting more aggressive. They didn't "
            "wait for the old high to sell -- they sold at a lower "
            "price. This shows increasing supply pressure and is "
            "a sign of weakness."
        ),
        "entry": "Sell Stop 1 tick below the signal bar's low.",
        "stop": "1 tick above the signal bar's high.",
        "target": "1x the risk distance below your entry.",
    },
    {
        "num": "6",
        "name": "Fade Higher High Double Top",
        "tier": "Tier A -- Best",
        "tier_color": (46, 139, 87),
        "direction": "LONG (profit if price goes UP)",
        "rr": "1 : 1",
        "order": "Buy Stop",
        "what": (
            "This looks like a double top, but the second high is "
            "actually HIGHER than the first. The \"ceiling\" broke "
            "upward. Most traders see a double top and expect a "
            "reversal down -- but the higher high tells you buyers "
            "won."
        ),
        "why": (
            "The failed double top traps short traders who sold "
            "expecting a drop. When price breaks above the first "
            "high, those trapped shorts panic-cover (buy to "
            "close), adding fuel to the move up. You fade the "
            "double top pattern by going long."
        ),
        "entry": "Buy Stop 1 tick above the signal bar's high.",
        "stop": "1 tick below the signal bar's low.",
        "target": "1x the risk distance above your entry.",
    },
    {
        "num": "7",
        "name": "Wedge Bottom (3-Push Down)",
        "tier": "Tier C -- Scalp",
        "tier_color": (160, 160, 160),
        "direction": "LONG (profit if price goes UP)",
        "rr": "0.5 : 1",
        "order": "Buy Stop",
        "what": (
            "Price makes three consecutive LOWER lows -- each one "
            "pushing deeper. Think of it like a ball bouncing down "
            "stairs: push 1, push 2, push 3. By the third push, "
            "selling pressure is exhausted."
        ),
        "why": (
            "After three pushes down, most sellers who wanted to "
            "sell have already sold. There's no one left to push "
            "price lower. The third push is often the final "
            "\"flush\" before buyers take over. This is a classic "
            "exhaustion pattern."
        ),
        "entry": "Buy Stop 1 tick above the 3rd push signal bar's high.",
        "stop": "1 tick below the signal bar's low.",
        "target": "0.5x the risk distance above entry (scalp).",
    },
    {
        "num": "8",
        "name": "Wedge Top (3-Push Up)",
        "tier": "Tier C -- Scalp",
        "tier_color": (160, 160, 160),
        "direction": "SHORT (profit if price goes DOWN)",
        "rr": "0.5 : 1",
        "order": "Sell Stop",
        "what": (
            "Price makes three consecutive HIGHER highs -- each "
            "one reaching a new peak. By the third push, buying "
            "pressure is exhausted."
        ),
        "why": (
            "After three pushes up, most buyers who wanted to buy "
            "have already bought. The third push is often the "
            "final \"squeeze\" before sellers take over. Mirror "
            "image of the wedge bottom."
        ),
        "entry": "Sell Stop 1 tick below the 3rd push signal bar's low.",
        "stop": "1 tick above the signal bar's high.",
        "target": "0.5x the risk distance below entry (scalp).",
    },
    {
        "num": "9",
        "name": "Fade Consecutive Sell Climaxes",
        "tier": "Tier A -- Best",
        "tier_color": (46, 139, 87),
        "direction": "LONG (profit if price goes UP)",
        "rr": "1 : 1",
        "order": "Buy Stop",
        "what": (
            "Three or more unusually large bear (red) bars appear "
            "in a short window. These bars are at least 1.5x the "
            "average bar size -- they represent panic selling. Then "
            "a bullish reversal bar appears (closes above its "
            "midpoint and above its open)."
        ),
        "why": (
            "Parabolic selling is unsustainable. When everyone "
            "panics and sells at once, there are no sellers left. "
            "It's like a rubber band stretched too far -- it snaps "
            "back. The bull reversal bar confirms sellers are "
            "exhausted and buyers are stepping in."
        ),
        "entry": "Buy Stop 1 tick above the bull reversal bar's high.",
        "stop": "1 tick below the reversal bar's low.",
        "target": "1x the risk distance above your entry.",
    },
    {
        "num": "10",
        "name": "Fade Consecutive Buy Climaxes",
        "tier": "Tier A -- Best",
        "tier_color": (46, 139, 87),
        "direction": "SHORT (profit if price goes DOWN)",
        "rr": "1 : 1",
        "order": "Sell Stop",
        "what": (
            "Three or more unusually large bull (green) bars "
            "appear in a short window. These represent euphoric, "
            "parabolic buying. Then a bearish reversal bar appears "
            "(closes below its midpoint and below its open)."
        ),
        "why": (
            "Parabolic buying is unsustainable. When everyone "
            "piles in at once, there are no buyers left to push "
            "price higher. The bear reversal bar confirms the "
            "euphoria is over and sellers are taking control."
        ),
        "entry": "Sell Stop 1 tick below the bear reversal bar's low.",
        "stop": "1 tick above the reversal bar's high.",
        "target": "1x the risk distance below your entry.",
    },
    {
        "num": "11",
        "name": "Breakout Pullback (BOPB)",
        "tier": "Tier A -- Best",
        "tier_color": (46, 139, 87),
        "direction": "LONG (profit if price goes UP)",
        "rr": "1 : 1",
        "order": "Buy Stop",
        "what": (
            "Price consolidates in a range (a base), then a "
            "strong bull bar breaks out above the range. After the "
            "breakout, price pulls back down to test the old "
            "ceiling (resistance). The pullback bar closes in its "
            "upper half -- buyers are defending the breakout level."
        ),
        "why": (
            "Old resistance becomes new support. The breakout "
            "proved buyers are strong enough to push through. The "
            "pullback is just profit-taking -- weaker hands "
            "exiting. When the pullback holds and the bar closes "
            "strong, it confirms the breakout was real and the "
            "trend should continue up."
        ),
        "entry": "Buy Stop 1 tick above the pullback test bar's high.",
        "stop": "1 tick below the test bar's low.",
        "target": "1x the risk distance above your entry.",
    },
    {
        "num": "12",
        "name": "Bear Stairs Reversal",
        "tier": "Tier A -- Best",
        "tier_color": (46, 139, 87),
        "direction": "LONG (profit if price goes UP)",
        "rr": "1 : 1",
        "order": "Buy Stop",
        "what": (
            "Price has been dropping in a staircase pattern -- "
            "making three or more lower lows like steps going "
            "down. Then a strong bull reversal bar appears: it "
            "closes well above its midpoint with a large body "
            "(body > 50% of the bar's range)."
        ),
        "why": (
            "A staircase of lower lows feels like a relentless "
            "downtrend, but each step down requires fresh sellers. "
            "After 3-4 pushes, sellers are exhausted. The strong "
            "bull bar shows buyers have finally overwhelmed the "
            "remaining sellers. It's the point where the staircase "
            "\"breaks.\""
        ),
        "entry": "Buy Stop 1 tick above the bull reversal bar's high.",
        "stop": "1 tick below the reversal bar's low.",
        "target": "1x the risk distance above your entry.",
    },
    {
        "num": "13",
        "name": "Exhaustive Climax at Measured Move",
        "tier": "Tier B -- Standard",
        "tier_color": (70, 130, 180),
        "direction": "SHORT (profit if price goes DOWN)",
        "rr": "0.75 : 1",
        "order": "Sell Stop",
        "what": (
            "An initial price spike establishes a leg size. The "
            "\"measured move target\" is that same distance "
            "projected from the spike's high. When price reaches "
            "that target with a massive exhaustion bar (> 2.5x the "
            "average bar size) that closes in its lower half, the "
            "move is done."
        ),
        "why": (
            "Measured moves are self-fulfilling: many traders use "
            "them as targets, so they take profits there. When the "
            "final push to the target comes as a huge, exhaustive "
            "bar, it signals the last burst of buying. The weak "
            "close (lower half) confirms that buyers couldn't hold "
            "the highs -- smart money is selling into the "
            "excitement."
        ),
        "entry": "Sell Stop 1 tick below the exhaustion bar's low.",
        "stop": "1 tick above the exhaustion bar's high.",
        "target": "0.75x the risk distance below your entry.",
    },
]

for s in setups:
    pdf.add_page()
    # Setup number + name
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(25, 25, 112)
    pdf.cell(0, 12, f"Setup #{s['num']}: {s['name']}", ln=True)
    pdf.ln(3)

    # Tier badge
    pdf.tier_badge(s["tier"], s["tier_color"])

    # Direction + Order + R:R summary
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 9, f"  Direction: {s['direction']}", ln=True, fill=True)
    pdf.cell(0, 9, f"  Order Type: {s['order']}     |     R:R: {s['rr']}", ln=True, fill=True)
    pdf.ln(5)

    # What is it?
    pdf.sub_heading("What Is This Pattern?")
    pdf.body_text(s["what"])

    # Why does it work?
    pdf.sub_heading("Why Does It Work?")
    pdf.body_text(s["why"])

    # How to trade it
    pdf.sub_heading("How to Trade It")
    pdf.bullet("Entry", s["entry"])
    pdf.bullet("Stop-Loss", s["stop"])
    pdf.bullet("Target", s["target"])

# ─── QUICK REFERENCE TABLE ───
pdf.add_page()
pdf.section_title("Quick Reference  --  All 13 Setups")
pdf.ln(2)

# Table header
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(40, 40, 80)
pdf.set_text_color(255, 255, 255)
col_w = [8, 50, 15, 22, 14, 28, 53]
headers = ["#", "Setup", "Dir", "Order", "R:R", "Tier", "Key Idea"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 8, h, border=1, fill=True, align="C")
pdf.ln()

rows = [
    ["1",  "Double Bottom",         "Long",  "Buy Stop",  "0.75", "B Std",  "Two equal lows held"],
    ["2",  "Higher Low DB",         "Long",  "Buy Stop",  "1:1",  "C Scalp", "2nd low higher"],
    ["3",  "Fade LL DB",            "Short", "Sell Stop", "1:1",  "A Best", "Floor broke, traps longs"],
    ["4",  "Double Top",            "Short", "Sell Stop", "1:1",  "B Std",  "Two equal highs held"],
    ["5",  "Lower High DT",        "Short", "Sell Stop", "1:1",  "B Std",  "2nd high lower"],
    ["6",  "Fade HH DT",           "Long",  "Buy Stop",  "1:1",  "A Best", "Ceiling broke, traps shorts"],
    ["7",  "Wedge Bottom",         "Long",  "Buy Stop",  "0.5",  "C Scalp", "3 pushes down exhausted"],
    ["8",  "Wedge Top",            "Short", "Sell Stop", "0.5",  "C Scalp", "3 pushes up exhausted"],
    ["9",  "Fade Sell Climax",     "Long",  "Buy Stop",  "1:1",  "A Best", "Panic selling snaps back"],
    ["10", "Fade Buy Climax",      "Short", "Sell Stop", "1:1",  "A Best", "Euphoria snaps back"],
    ["11", "Breakout Pullback",    "Long",  "Buy Stop",  "1:1",  "A Best", "Old resist = new support"],
    ["12", "Bear Stairs Rev.",     "Long",  "Buy Stop",  "1:1",  "A Best", "Staircase of lows breaks"],
    ["13", "MM Climax",            "Short", "Sell Stop", "0.75", "B Std",  "Hit target w/ exhaustion"],
]

pdf.set_font("Helvetica", "", 8.5)
pdf.set_text_color(30, 30, 30)
for ri, row in enumerate(rows):
    if ri % 2 == 0:
        pdf.set_fill_color(245, 245, 255)
    else:
        pdf.set_fill_color(255, 255, 255)
    for ci, val in enumerate(row):
        align = "C" if ci in (0, 2, 3, 4) else "L"
        if ci == 2:
            if val == "Long":
                pdf.set_text_color(0, 130, 60)
            else:
                pdf.set_text_color(200, 30, 30)
        else:
            pdf.set_text_color(30, 30, 30)
        pdf.cell(col_w[ci], 7, val, border=1, fill=True, align=align)
    pdf.ln()

pdf.ln(10)
pdf.set_font("Helvetica", "B", 13)
pdf.set_text_color(40, 40, 40)
pdf.cell(0, 9, "Golden Rules", ln=True)
pdf.ln(3)
pdf.set_font("Helvetica", "", 12)
rules = [
    "Every entry is a STOP order -- the market must prove the move before you're in.",
    "Entry is always 1 tick beyond the signal bar. Stop is always the opposite side.",
    "Never chase. If the stop order doesn't fill, the setup failed -- move on.",
    "Tier A setups are the highest probability. Focus on these when learning.",
    "Smaller R:R (0.5:1) = take profit quickly. Don't hold scalps for runners.",
    "Always know your risk BEFORE entering. Risk = |Entry - Stop|.",
]
for r in rules:
    pdf.cell(6)
    y_before = pdf.get_y()
    pdf.multi_cell(0, 7, f"- {r}")
    pdf.ln(1)

out_path = "/home/user/BPA-Bot-1/BPA_All_13_Profitable_Setups.pdf"
pdf.output(out_path)
print(f"PDF saved to: {out_path}")
