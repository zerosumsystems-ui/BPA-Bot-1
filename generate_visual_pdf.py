"""
BPA Bot Setup Visual Guide — candlestick pattern diagrams + detection flowcharts.
Includes summary stats, MAE/MFE analysis, per-ticker results, and Python source code.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Flowable, Table, TableStyle,
    Preformatted, KeepTogether,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Group, Polygon
from reportlab.graphics import renderPDF
import datetime
import csv
import os
from collections import defaultdict

OUTPUT = "/sessions/awesome-festive-cray/mnt/BPA-Bot-1/BPA_Bot_Visual_Guide.pdf"
CSV_PATH = "/sessions/awesome-festive-cray/mnt/uploads/backtest_SPY_scalp 1500 days.csv"

# Colors
GREEN   = HexColor("#39ff14")
RED     = HexColor("#ff2e63")
DKGREEN = HexColor("#1a8a00")
DKRED   = HexColor("#cc0033")
BG      = HexColor("#f8f9fa")
DARK    = HexColor("#1a1a2e")
GRAY    = HexColor("#999999")
LTGRAY  = HexColor("#dddddd")
BLUE    = HexColor("#4a90d9")
WHITE   = white

# ─── R:R from SETUP_CONFIG ───
SETUP_RR = {
    "Double Bottom": 0.75,
    "Double Top": 1.0,
    "Higher Low Double Bottom": 1.0,
    "Lower High Double Top": 1.0,
    "Fade Lower Low Double Bottom": 1.0,
    "Fade Higher High Double Top": 1.0,
    "Wedge Bottom": 0.5,
    "Wedge Top": 0.5,
    "Fade Consecutive Sell Climaxes (Reversal)": 1.0,
    "Fade Consecutive Buy Climaxes (Reversal)": 1.0,
    "Fade Bull Breakout Pullback": 1.0,
    "Fade Bear Stairs Reversal (3rd/4th Push)": 1.0,
    "Fade Exhaustive Bull Climax at MM": 0.75,
}

SETUP_TIERS = {
    "Double Bottom": "B",
    "Double Top": "B",
    "Higher Low Double Bottom": "C",
    "Lower High Double Top": "B",
    "Fade Lower Low Double Bottom": "A",
    "Fade Higher High Double Top": "A",
    "Wedge Bottom": "C",
    "Wedge Top": "C",
    "Fade Consecutive Sell Climaxes (Reversal)": "A",
    "Fade Consecutive Buy Climaxes (Reversal)": "A",
    "Fade Bull Breakout Pullback": "A",
    "Fade Bear Stairs Reversal (3rd/4th Push)": "A",
    "Fade Exhaustive Bull Climax at MM": "B",
}


# ─── Read CSV and compute stats ───

def compute_stats():
    """Read backtest CSV and return per-setup stats dict + total trading days."""
    raw = defaultdict(lambda: {
        'count': 0, 'wins': 0, 'pnl': 0.0,
        'r_mult': [], 'mae': [], 'mfe': [], 'mae_r': [], 'mfe_r': [],
        'bars': [], 'risk': [], 'dates': set(),
    })
    all_dates = set()

    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row['Setup']
            d = raw[s]
            d['count'] += 1
            d['wins'] += 1 if row['Winner'] == 'True' else 0
            d['pnl'] += float(row['P&L'])
            d['r_mult'].append(float(row['R Multiple']))
            d['mae'].append(float(row['MAE']))
            d['mfe'].append(float(row['MFE']))
            d['mae_r'].append(float(row['MAE (R)']))
            d['mfe_r'].append(float(row['MFE (R)']))
            d['bars'].append(int(row['Bars Held']))
            d['risk'].append(float(row['Risk/Share']))
            dt = row['Entry Time'].split(' ')[0]
            d['dates'].add(dt)
            all_dates.add(dt)

    total_days = len(all_dates)
    results = {}

    for name, d in raw.items():
        n = d['count']
        gross_win = sum(r for r in d['r_mult'] if r > 0)
        gross_loss = abs(sum(r for r in d['r_mult'] if r < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0

        results[name] = {
            'count': n,
            'wins': d['wins'],
            'win_rate': d['wins'] / n * 100,
            'pnl': d['pnl'],
            'ev': d['pnl'] / n,
            'pf': pf,
            'avg_r': sum(d['r_mult']) / n,
            'avg_mae': sum(d['mae']) / n,
            'avg_mfe': sum(d['mfe']) / n,
            'max_mae': max(d['mae']),
            'max_mfe': max(d['mfe']),
            'avg_mae_r': sum(d['mae_r']) / n,
            'avg_mfe_r': sum(d['mfe_r']) / n,
            'avg_bars': sum(d['bars']) / n,
            'avg_risk': sum(d['risk']) / n,
            'trades_per_day': n / total_days,
            'active_days': len(d['dates']),
            'rr': SETUP_RR.get(name, 1.0),
            'tier': SETUP_TIERS.get(name, "?"),
        }

    # Per-ticker stats
    ticker_stats = defaultdict(lambda: {
        'count': 0, 'wins': 0, 'pnl': 0.0, 'r_mult': [],
        'mae': [], 'mfe': [], 'dates': set(),
    })
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row['Ticker']
            d = ticker_stats[t]
            d['count'] += 1
            d['wins'] += 1 if row['Winner'] == 'True' else 0
            d['pnl'] += float(row['P&L'])
            d['r_mult'].append(float(row['R Multiple']))
            d['mae'].append(float(row['MAE']))
            d['mfe'].append(float(row['MFE']))
            d['dates'].add(row['Entry Time'].split(' ')[0])

    ticker_results = {}
    for ticker, d in ticker_stats.items():
        n = d['count']
        gross_win = sum(r for r in d['r_mult'] if r > 0)
        gross_loss = abs(sum(r for r in d['r_mult'] if r < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0
        ticker_results[ticker] = {
            'count': n,
            'wins': d['wins'],
            'win_rate': d['wins'] / n * 100,
            'pnl': d['pnl'],
            'ev': d['pnl'] / n,
            'pf': pf,
            'avg_r': sum(d['r_mult']) / n,
            'avg_mae': sum(d['mae']) / n,
            'avg_mfe': sum(d['mfe']) / n,
            'active_days': len(d['dates']),
            'trades_per_day': n / len(d['dates']) if d['dates'] else 0,
        }

    # Per-ticker per-setup breakdown
    ticker_setup_stats = defaultdict(lambda: defaultdict(lambda: {
        'count': 0, 'wins': 0, 'pnl': 0.0, 'r_mult': [],
    }))
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row['Ticker']
            s = row['Setup']
            d = ticker_setup_stats[t][s]
            d['count'] += 1
            d['wins'] += 1 if row['Winner'] == 'True' else 0
            d['pnl'] += float(row['P&L'])
            d['r_mult'].append(float(row['R Multiple']))

    return results, total_days, ticker_results, ticker_setup_stats


def candle(g, x, y_open, y_close, y_high, y_low, width=14, bull=True):
    """Draw a single candlestick on a Group. y values are pixel positions."""
    color = DKGREEN if bull else DKRED
    fill = WHITE if bull else color
    cx = x + width / 2
    g.add(Line(cx, y_low, cx, y_high, strokeColor=color, strokeWidth=1))
    body_top = max(y_open, y_close)
    body_bot = min(y_open, y_close)
    body_h = max(body_top - body_bot, 2)
    g.add(Rect(x, body_bot, width, body_h, fillColor=fill, strokeColor=color, strokeWidth=1))


def arrow_down(g, x, y, size=8):
    g.add(Polygon(
        points=[x, y, x + size, y, x + size/2, y - size],
        fillColor=DKRED, strokeColor=DKRED, strokeWidth=0.5,
    ))

def arrow_up(g, x, y, size=8):
    g.add(Polygon(
        points=[x, y, x + size, y, x + size/2, y + size],
        fillColor=DKGREEN, strokeColor=DKGREEN, strokeWidth=0.5,
    ))


def label(g, x, y, text, size=7, color=DARK, anchor="middle"):
    g.add(String(x, y, text, fontSize=size, fillColor=color, textAnchor=anchor))


def dashed_line(g, x1, y1, x2, y2, color=GRAY):
    g.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.8, strokeDashArray=[3, 3]))


# ─── Pattern Drawing Functions ───

def draw_double_bottom(d, w, h):
    g = Group()
    base_y = 30
    candle(g, 20, base_y+90, base_y+70, base_y+95, base_y+65, bull=False)
    candle(g, 40, base_y+75, base_y+55, base_y+80, base_y+50, bull=False)
    candle(g, 65, base_y+50, base_y+30, base_y+55, base_y+20, bull=False)
    candle(g, 85, base_y+35, base_y+55, base_y+60, base_y+30, bull=True)
    candle(g, 105, base_y+55, base_y+65, base_y+70, base_y+50, bull=True)
    candle(g, 130, base_y+60, base_y+35, base_y+65, base_y+20, bull=False)
    candle(g, 155, base_y+30, base_y+55, base_y+60, base_y+22, bull=True)
    dashed_line(g, 55, base_y+20, 175, base_y+20)
    label(g, 65, base_y+8, "Low 1", size=6)
    label(g, 140, base_y+8, "Low 2 (equal)", size=6)
    arrow_up(g, 158, base_y+65)
    label(g, 162, base_y+78, "LONG", size=8, color=DKGREEN)
    d.add(g)


def draw_higher_low_db(d, w, h):
    g = Group()
    base_y = 30
    candle(g, 20, base_y+90, base_y+70, base_y+95, base_y+65, bull=False)
    candle(g, 40, base_y+75, base_y+55, base_y+80, base_y+50, bull=False)
    candle(g, 65, base_y+50, base_y+25, base_y+55, base_y+15, bull=False)
    candle(g, 85, base_y+30, base_y+55, base_y+60, base_y+25, bull=True)
    candle(g, 105, base_y+55, base_y+65, base_y+70, base_y+50, bull=True)
    candle(g, 130, base_y+60, base_y+38, base_y+65, base_y+30, bull=False)
    candle(g, 155, base_y+35, base_y+60, base_y+65, base_y+30, bull=True)
    dashed_line(g, 55, base_y+15, 175, base_y+15)
    dashed_line(g, 120, base_y+30, 175, base_y+30, color=DKGREEN)
    label(g, 65, base_y+3, "Low 1", size=6)
    label(g, 140, base_y+22, "Low 2 (higher)", size=6, color=DKGREEN)
    arrow_up(g, 158, base_y+70)
    label(g, 162, base_y+83, "LONG", size=8, color=DKGREEN)
    d.add(g)


def draw_fade_ll_db(d, w, h):
    g = Group()
    base_y = 30
    candle(g, 20, base_y+90, base_y+70, base_y+95, base_y+65, bull=False)
    candle(g, 40, base_y+75, base_y+55, base_y+80, base_y+50, bull=False)
    candle(g, 65, base_y+50, base_y+30, base_y+55, base_y+22, bull=False)
    candle(g, 85, base_y+35, base_y+55, base_y+60, base_y+30, bull=True)
    candle(g, 105, base_y+55, base_y+60, base_y+65, base_y+50, bull=True)
    candle(g, 130, base_y+55, base_y+25, base_y+60, base_y+12, bull=False)
    candle(g, 155, base_y+30, base_y+20, base_y+35, base_y+10, bull=False)
    dashed_line(g, 55, base_y+22, 175, base_y+22)
    dashed_line(g, 120, base_y+12, 175, base_y+12, color=DKRED)
    label(g, 65, base_y+10, "Low 1", size=6)
    label(g, 140, base_y+2, "Low 2 (lower)", size=6, color=DKRED)
    arrow_down(g, 158, base_y+8)
    label(g, 162, base_y-8, "FADE SHORT", size=7, color=DKRED)
    d.add(g)


def draw_double_top(d, w, h):
    g = Group()
    base_y = 20
    candle(g, 20, base_y+20, base_y+40, base_y+45, base_y+15, bull=True)
    candle(g, 40, base_y+40, base_y+60, base_y+65, base_y+35, bull=True)
    candle(g, 65, base_y+60, base_y+85, base_y+90, base_y+55, bull=True)
    candle(g, 85, base_y+80, base_y+65, base_y+85, base_y+60, bull=False)
    candle(g, 105, base_y+65, base_y+55, base_y+70, base_y+50, bull=False)
    candle(g, 130, base_y+60, base_y+82, base_y+90, base_y+55, bull=True)
    candle(g, 155, base_y+80, base_y+60, base_y+85, base_y+55, bull=False)
    dashed_line(g, 55, base_y+90, 175, base_y+90)
    label(g, 70, base_y+95, "High 1", size=6)
    label(g, 140, base_y+95, "High 2 (equal)", size=6)
    arrow_down(g, 158, base_y+50)
    label(g, 162, base_y+38, "SHORT", size=8, color=DKRED)
    d.add(g)


def draw_lower_high_dt(d, w, h):
    g = Group()
    base_y = 20
    candle(g, 20, base_y+20, base_y+40, base_y+45, base_y+15, bull=True)
    candle(g, 40, base_y+40, base_y+60, base_y+65, base_y+35, bull=True)
    candle(g, 65, base_y+60, base_y+88, base_y+95, base_y+55, bull=True)
    candle(g, 85, base_y+85, base_y+65, base_y+90, base_y+60, bull=False)
    candle(g, 105, base_y+65, base_y+55, base_y+70, base_y+50, bull=False)
    candle(g, 130, base_y+58, base_y+78, base_y+83, base_y+53, bull=True)
    candle(g, 155, base_y+75, base_y+55, base_y+80, base_y+50, bull=False)
    dashed_line(g, 55, base_y+95, 175, base_y+95)
    dashed_line(g, 120, base_y+83, 175, base_y+83, color=DKRED)
    label(g, 70, base_y+100, "High 1", size=6)
    label(g, 140, base_y+87, "High 2 (lower)", size=6, color=DKRED)
    arrow_down(g, 158, base_y+45)
    label(g, 162, base_y+33, "SHORT", size=8, color=DKRED)
    d.add(g)


def draw_fade_hh_dt(d, w, h):
    g = Group()
    base_y = 20
    candle(g, 20, base_y+20, base_y+40, base_y+45, base_y+15, bull=True)
    candle(g, 40, base_y+40, base_y+60, base_y+65, base_y+35, bull=True)
    candle(g, 65, base_y+60, base_y+80, base_y+85, base_y+55, bull=True)
    candle(g, 85, base_y+78, base_y+60, base_y+82, base_y+55, bull=False)
    candle(g, 105, base_y+60, base_y+50, base_y+65, base_y+45, bull=False)
    candle(g, 130, base_y+55, base_y+85, base_y+95, base_y+50, bull=True)
    candle(g, 155, base_y+82, base_y+92, base_y+98, base_y+78, bull=True)
    dashed_line(g, 55, base_y+85, 175, base_y+85)
    dashed_line(g, 120, base_y+95, 175, base_y+95, color=DKGREEN)
    label(g, 70, base_y+90, "High 1", size=6)
    label(g, 138, base_y+100, "High 2 (higher)", size=6, color=DKGREEN)
    arrow_up(g, 158, base_y+100)
    label(g, 162, base_y+113, "FADE LONG", size=7, color=DKGREEN)
    d.add(g)


def draw_wedge_bottom(d, w, h):
    g = Group()
    base_y = 25
    candle(g, 15, base_y+95, base_y+80, base_y+100, base_y+75, bull=False)
    candle(g, 35, base_y+78, base_y+60, base_y+82, base_y+50, bull=False)
    candle(g, 55, base_y+58, base_y+40, base_y+62, base_y+35, bull=False)
    candle(g, 72, base_y+42, base_y+55, base_y+60, base_y+38, bull=True)
    candle(g, 89, base_y+52, base_y+30, base_y+56, base_y+25, bull=False)
    candle(g, 106, base_y+32, base_y+48, base_y+52, base_y+28, bull=True)
    candle(g, 123, base_y+45, base_y+22, base_y+48, base_y+15, bull=False)
    candle(g, 143, base_y+20, base_y+48, base_y+52, base_y+15, bull=True)
    dashed_line(g, 60, base_y+35, 95, base_y+25)
    dashed_line(g, 95, base_y+25, 130, base_y+15)
    label(g, 60, base_y+27, "1", size=7, color=DKRED)
    label(g, 95, base_y+17, "2", size=7, color=DKRED)
    label(g, 130, base_y+7, "3", size=7, color=DKRED)
    arrow_up(g, 148, base_y+57)
    label(g, 152, base_y+70, "LONG", size=8, color=DKGREEN)
    d.add(g)


def draw_wedge_top(d, w, h):
    g = Group()
    base_y = 15
    candle(g, 15, base_y+15, base_y+30, base_y+35, base_y+10, bull=True)
    candle(g, 35, base_y+30, base_y+50, base_y+55, base_y+25, bull=True)
    candle(g, 55, base_y+50, base_y+70, base_y+75, base_y+45, bull=True)
    candle(g, 72, base_y+68, base_y+55, base_y+72, base_y+50, bull=False)
    candle(g, 89, base_y+58, base_y+80, base_y+85, base_y+53, bull=True)
    candle(g, 106, base_y+78, base_y+62, base_y+82, base_y+58, bull=False)
    candle(g, 123, base_y+65, base_y+90, base_y+95, base_y+60, bull=True)
    candle(g, 143, base_y+88, base_y+62, base_y+92, base_y+58, bull=False)
    dashed_line(g, 60, base_y+75, 95, base_y+85)
    dashed_line(g, 95, base_y+85, 130, base_y+95)
    label(g, 60, base_y+79, "1", size=7, color=DKGREEN)
    label(g, 95, base_y+89, "2", size=7, color=DKGREEN)
    label(g, 130, base_y+99, "3", size=7, color=DKGREEN)
    arrow_down(g, 148, base_y+53)
    label(g, 152, base_y+40, "SHORT", size=8, color=DKRED)
    d.add(g)


def draw_fade_sell_climaxes(d, w, h):
    g = Group()
    base_y = 25
    candle(g, 10, base_y+90, base_y+80, base_y+95, base_y+75, bull=False)
    candle(g, 30, base_y+78, base_y+45, base_y+82, base_y+38, 18, bull=False)
    candle(g, 55, base_y+48, base_y+18, base_y+52, base_y+12, 18, bull=False)
    candle(g, 80, base_y+22, base_y-5, base_y+26, base_y-10, 18, bull=False)
    label(g, 40, base_y+86, "Climax", size=6, color=DKRED)
    label(g, 65, base_y+56, "Climax", size=6, color=DKRED)
    label(g, 90, base_y+30, "Climax", size=6, color=DKRED)
    candle(g, 110, base_y-8, base_y+30, base_y+35, base_y-12, 18, bull=True)
    label(g, 119, base_y+40, "Reversal", size=6, color=DKGREEN)
    arrow_up(g, 116, base_y+48)
    label(g, 120, base_y+60, "LONG", size=8, color=DKGREEN)
    d.add(g)


def draw_fade_buy_climaxes(d, w, h):
    g = Group()
    base_y = 10
    candle(g, 10, base_y+15, base_y+25, base_y+30, base_y+10, bull=True)
    candle(g, 30, base_y+25, base_y+58, base_y+62, base_y+20, 18, bull=True)
    candle(g, 55, base_y+55, base_y+85, base_y+90, base_y+50, 18, bull=True)
    candle(g, 80, base_y+82, base_y+108, base_y+115, base_y+78, 18, bull=True)
    label(g, 40, base_y+4, "Climax", size=6, color=DKGREEN)
    label(g, 65, base_y+38, "Climax", size=6, color=DKGREEN)
    label(g, 90, base_y+68, "Climax", size=6, color=DKGREEN)
    candle(g, 110, base_y+110, base_y+80, base_y+115, base_y+75, 18, bull=False)
    label(g, 119, base_y+68, "Reversal", size=6, color=DKRED)
    arrow_down(g, 116, base_y+62)
    label(g, 120, base_y+50, "SHORT", size=8, color=DKRED)
    d.add(g)


def draw_fade_bull_breakout_pullback(d, w, h):
    g = Group()
    base_y = 30
    candle(g, 10, base_y+40, base_y+50, base_y+55, base_y+35, bull=True)
    candle(g, 28, base_y+48, base_y+42, base_y+53, base_y+38, bull=False)
    candle(g, 46, base_y+43, base_y+50, base_y+54, base_y+40, bull=True)
    candle(g, 64, base_y+49, base_y+44, base_y+53, base_y+40, bull=False)
    dashed_line(g, 5, base_y+55, 120, base_y+55, color=BLUE)
    label(g, 5, base_y+58, "Range High", size=5, color=BLUE, anchor="start")
    candle(g, 82, base_y+48, base_y+72, base_y+78, base_y+45, 16, bull=True)
    label(g, 90, base_y+82, "Breakout", size=6, color=DKGREEN)
    candle(g, 105, base_y+70, base_y+58, base_y+74, base_y+54, bull=False)
    label(g, 112, base_y+46, "Pullback", size=6, color=GRAY)
    candle(g, 125, base_y+56, base_y+72, base_y+76, base_y+52, bull=True)
    arrow_up(g, 130, base_y+80)
    label(g, 134, base_y+92, "LONG", size=8, color=DKGREEN)
    d.add(g)


def draw_fade_bear_stairs(d, w, h):
    g = Group()
    base_y = 25
    candle(g, 10, base_y+90, base_y+78, base_y+95, base_y+72, bull=False)
    candle(g, 30, base_y+80, base_y+68, base_y+84, base_y+62, bull=False)
    candle(g, 48, base_y+70, base_y+72, base_y+76, base_y+65, bull=True)
    candle(g, 66, base_y+72, base_y+55, base_y+75, base_y+48, bull=False)
    candle(g, 84, base_y+58, base_y+60, base_y+64, base_y+52, bull=True)
    candle(g, 102, base_y+60, base_y+40, base_y+63, base_y+33, bull=False)
    dashed_line(g, 16, base_y+72, 36, base_y+62)
    dashed_line(g, 36, base_y+62, 72, base_y+48)
    dashed_line(g, 72, base_y+48, 108, base_y+33)
    label(g, 16, base_y+66, "1", size=6, color=DKRED)
    label(g, 36, base_y+56, "2", size=6, color=DKRED)
    label(g, 72, base_y+42, "3", size=6, color=DKRED)
    label(g, 108, base_y+27, "4", size=6, color=DKRED)
    candle(g, 125, base_y+35, base_y+65, base_y+70, base_y+30, 16, bull=True)
    arrow_up(g, 130, base_y+75)
    label(g, 134, base_y+88, "LONG", size=8, color=DKGREEN)
    d.add(g)


def draw_fade_exhaustive_climax_mm(d, w, h):
    g = Group()
    base_y = 10
    candle(g, 10, base_y+15, base_y+40, base_y+45, base_y+10, bull=True)
    candle(g, 28, base_y+38, base_y+55, base_y+60, base_y+35, bull=True)
    candle(g, 46, base_y+53, base_y+48, base_y+57, base_y+44, bull=False)
    candle(g, 62, base_y+50, base_y+55, base_y+58, base_y+46, bull=True)
    candle(g, 78, base_y+53, base_y+70, base_y+74, base_y+50, bull=True)
    candle(g, 96, base_y+68, base_y+82, base_y+86, base_y+65, bull=True)
    candle(g, 116, base_y+80, base_y+108, base_y+115, base_y+78, 18, bull=True)
    candle(g, 140, base_y+106, base_y+85, base_y+112, base_y+82, 16, bull=False)
    dashed_line(g, 5, base_y+110, 170, base_y+110, color=BLUE)
    label(g, 170, base_y+113, "MM Target", size=6, color=BLUE, anchor="start")
    label(g, 125, base_y+68, "Exhaustion", size=6, color=DKRED)
    arrow_down(g, 145, base_y+78)
    label(g, 149, base_y+66, "SHORT", size=8, color=DKRED)
    d.add(g)


# ─── Flowchart Drawing ───

def draw_flowchart(steps, w=220, entry_dir="LONG"):
    box_w = 180
    box_h = 22
    gap = 8
    total_h = len(steps) * (box_h + gap) + 30
    d = Drawing(w, total_h)

    y = total_h - 10
    cx = w / 2

    for i, (text, shape) in enumerate(steps):
        bx = cx - box_w / 2
        by = y - box_h

        if shape == "diamond":
            fill = HexColor("#fff3cd")
            stroke = HexColor("#ffc107")
        elif shape == "entry":
            fill = HexColor("#d4edda") if entry_dir == "LONG" else HexColor("#f8d7da")
            stroke = DKGREEN if entry_dir == "LONG" else DKRED
        else:
            fill = HexColor("#e8eaf6")
            stroke = HexColor("#5c6bc0")

        d.add(Rect(bx, by, box_w, box_h, fillColor=fill, strokeColor=stroke, strokeWidth=1, rx=4, ry=4))
        d.add(String(cx, by + 7, text, fontSize=7, fillColor=DARK, textAnchor="middle"))

        if i < len(steps) - 1:
            arrow_y = by - gap
            d.add(Line(cx, by, cx, arrow_y, strokeColor=GRAY, strokeWidth=0.8))
            d.add(Polygon(
                points=[cx - 3, arrow_y + 4, cx + 3, arrow_y + 4, cx, arrow_y],
                fillColor=GRAY, strokeWidth=0,
            ))

        y = by - gap

    return d


# ─── All Setups ───

SETUPS = [
    {
        "name": "Double Bottom",
        "draw": draw_double_bottom,
        "flow": [
            ("Find swing low (lookback=3 bars left/right)", "box"),
            ("Is there a prior swing low within 0.3%?", "diamond"),
            ("Are the two lows roughly equal?", "diamond"),
            ("Wait for bull reversal bar", "box"),
            ("BUY STOP above signal bar high", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Higher Low Double Bottom",
        "draw": draw_higher_low_db,
        "flow": [
            ("Find swing low (lookback=3)", "box"),
            ("Prior swing low within 0.3%?", "diamond"),
            ("Is second low HIGHER than first?", "diamond"),
            ("Buyers stepping in at higher price", "box"),
            ("BUY STOP above signal bar high", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Fade Lower Low Double Bottom",
        "draw": draw_fade_ll_db,
        "flow": [
            ("Find swing low (lookback=3)", "box"),
            ("Prior swing low within 0.3%?", "diamond"),
            ("Is second low LOWER than first?", "diamond"),
            ("Failed DB — sellers winning", "box"),
            ("SELL STOP below signal bar low (FADE)", "entry"),
        ],
        "entry_dir": "SHORT",
    },
    {
        "name": "Double Top",
        "draw": draw_double_top,
        "flow": [
            ("Find swing high (lookback=3)", "box"),
            ("Prior swing high within 0.3%?", "diamond"),
            ("Are the two highs roughly equal?", "diamond"),
            ("Wait for bear reversal bar", "box"),
            ("SELL STOP below signal bar low", "entry"),
        ],
        "entry_dir": "SHORT",
    },
    {
        "name": "Lower High Double Top",
        "draw": draw_lower_high_dt,
        "flow": [
            ("Find swing high (lookback=3)", "box"),
            ("Prior swing high within 0.3%?", "diamond"),
            ("Is second high LOWER than first?", "diamond"),
            ("Sellers gaining control", "box"),
            ("SELL STOP below signal bar low", "entry"),
        ],
        "entry_dir": "SHORT",
    },
    {
        "name": "Fade Higher High Double Top",
        "draw": draw_fade_hh_dt,
        "flow": [
            ("Find swing high (lookback=3)", "box"),
            ("Prior swing high within 0.3%?", "diamond"),
            ("Is second high HIGHER than first?", "diamond"),
            ("Failed DT — buyers winning", "box"),
            ("BUY STOP above signal bar high (FADE)", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Wedge Bottom (3 Pushes Down)",
        "draw": draw_wedge_bottom,
        "flow": [
            ("Track last 3 swing lows", "box"),
            ("SW3 < SW2 < SW1? (3 consecutive lower lows)", "diamond"),
            ("3rd push exhausts sellers", "box"),
            ("BUY STOP above signal bar high", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Wedge Top (3 Pushes Up)",
        "draw": draw_wedge_top,
        "flow": [
            ("Track last 3 swing highs", "box"),
            ("SW3 > SW2 > SW1? (3 consecutive higher highs)", "diamond"),
            ("3rd push exhausts buyers", "box"),
            ("SELL STOP below signal bar low", "entry"),
        ],
        "entry_dir": "SHORT",
    },
    {
        "name": "Fade Consecutive Sell Climaxes",
        "draw": draw_fade_sell_climaxes,
        "flow": [
            ("10-bar lookback: count bars with body > 1.5x avg range", "box"),
            ("3+ massive bear bars found?", "diamond"),
            ("Is current bar a bull reversal? (close upper half, body > 40%)", "diamond"),
            ("Panic selling exhausted — bears done", "box"),
            ("BUY STOP above signal bar high", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Fade Consecutive Buy Climaxes",
        "draw": draw_fade_buy_climaxes,
        "flow": [
            ("10-bar lookback: count bars with body > 1.5x avg range", "box"),
            ("3+ massive bull bars found?", "diamond"),
            ("Is current bar a bear reversal? (close lower half, body > 40%)", "diamond"),
            ("Euphoric buying exhausted — bulls done", "box"),
            ("SELL STOP below signal bar low", "entry"),
        ],
        "entry_dir": "SHORT",
    },
    {
        "name": "Fade Bull Breakout Pullback",
        "draw": draw_fade_bull_breakout_pullback,
        "flow": [
            ("Compute base range from 8 bars back", "box"),
            ("Did bar[-2] close above range high? (breakout)", "diamond"),
            ("Is current bar within 0.2% of breakout level? (pullback)", "diamond"),
            ("Pullback tests breakout as support", "box"),
            ("BUY STOP above signal bar high", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Fade Bear Stairs Reversal",
        "draw": draw_fade_bear_stairs,
        "flow": [
            ("15-bar lookback: find local minima", "box"),
            ("3+ progressively lower minima? (stair-step)", "diamond"),
            ("Is current bar a strong bull reversal? (body > 50% range)", "diamond"),
            ("Bear channel exhausted on 3rd/4th push", "box"),
            ("BUY STOP above signal bar high", "entry"),
        ],
        "entry_dir": "LONG",
    },
    {
        "name": "Fade Exhaustive Bull Climax at MM",
        "draw": draw_fade_exhaustive_climax_mm,
        "flow": [
            ("20-bar trend window: compute initial spike range", "box"),
            ("Double spike range = measured move target", "box"),
            ("Has price reached MM target?", "diamond"),
            ("Is current bar exhaustion? (range > 2.5x avg, close near low)", "diamond"),
            ("SELL STOP below signal bar low", "entry"),
        ],
        "entry_dir": "SHORT",
    },
]

# Map CSV names to SETUPS names
CSV_NAME_MAP = {
    "Fade Consecutive Sell Climaxes (Reversal)": "Fade Consecutive Sell Climaxes",
    "Fade Consecutive Buy Climaxes (Reversal)": "Fade Consecutive Buy Climaxes",
    "Fade Bear Stairs Reversal (3rd/4th Push)": "Fade Bear Stairs Reversal",
    "Wedge Bottom": "Wedge Bottom (3 Pushes Down)",
    "Wedge Top": "Wedge Top (3 Pushes Up)",
}


class PatternFlowable(Flowable):
    """Custom flowable that draws a candlestick pattern + flowchart side by side."""

    def __init__(self, setup):
        Flowable.__init__(self)
        self.setup = setup
        self.width = 7 * inch
        self.height = 2 * inch

    def draw(self):
        s = self.setup
        c = self.canv

        d = Drawing(200, 130)
        d.add(Rect(0, 0, 200, 130, fillColor=HexColor("#fafafa"), strokeColor=LTGRAY, strokeWidth=0.5, rx=4, ry=4))
        s["draw"](d, 200, 130)
        renderPDF.draw(d, c, 0, 10)

        flow_d = draw_flowchart(s["flow"], w=300, entry_dir=s["entry_dir"])
        renderPDF.draw(flow_d, c, 215, 10)


def shorten_name(csv_name):
    """Shorten setup name for table display."""
    return csv_name.replace("Fade Consecutive ", "F.").replace("(Reversal)", "Rev") \
        .replace("(3rd/4th Push)", "3/4P").replace("Fade ", "F.") \
        .replace("Exhaustive Bull Climax at MM", "Exhaust Climax MM") \
        .replace("Higher Low ", "HL ").replace("Lower High ", "LH ") \
        .replace("Double Bottom", "DB").replace("Double Top", "DT") \
        .replace("Bull Breakout Pullback", "Bull BO PB") \
        .replace("Bear Stairs Reversal", "Bear Stairs Rev") \
        .replace("Lower Low ", "LL ").replace("Higher High ", "HH ") \
        .replace("Wedge Bottom", "Wedge Bot").replace("Wedge Top", "Wedge Top")


def build_pdf():
    stats, total_days, ticker_results, ticker_setup_stats = compute_stats()

    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )

    name_style = ParagraphStyle("Name", fontSize=14, leading=18, textColor=DARK, fontName="Helvetica-Bold")
    tag_style = ParagraphStyle("Tag", fontSize=8, leading=11, textColor=GRAY)
    stats_style = ParagraphStyle("Stats", fontSize=7.5, leading=10, textColor=HexColor("#555555"), fontName="Helvetica-Bold")
    title_style = ParagraphStyle("Title", fontSize=22, leading=28, textColor=DARK, fontName="Helvetica-Bold", alignment=TA_CENTER)
    sub_style = ParagraphStyle("Sub", fontSize=10, textColor=GRAY, alignment=TA_CENTER, spaceAfter=20)
    section_head = ParagraphStyle("SectionHead", fontSize=14, leading=18, textColor=DARK, fontName="Helvetica-Bold", spaceAfter=8)

    story = []

    # ═══════════════════════════════════════════════════════
    #  PAGE 1: TITLE + SUMMARY TABLE
    # ═══════════════════════════════════════════════════════

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("BPA Bot — Setup Visual Guide", title_style))

    total_trades = sum(s['count'] for s in stats.values())
    total_pnl = sum(s['pnl'] for s in stats.values())
    overall_wr = sum(s['wins'] for s in stats.values()) / total_trades * 100
    story.append(Paragraph(
        f"13 Setups  &bull;  SPY 5-Min  &bull;  {total_days:,} Trading Days  &bull;  "
        f"{total_trades:,} Trades  &bull;  {overall_wr:.0f}% Win Rate  &bull;  "
        f"${total_pnl:,.0f} Net P&L  &bull;  {datetime.date.today().strftime('%B %d, %Y')}",
        sub_style))

    # ── Summary Ranking Table ──
    story.append(Paragraph("Setup Performance Summary", section_head))

    # Build table data sorted by total P&L descending
    sorted_names = sorted(stats.keys(), key=lambda n: -stats[n]['pnl'])

    header = ['Setup', 'N', 'Win%', 'R:R', 'EV(R)', 'EV($)', 'PF',
              'Avg\nMAE($)', 'Avg\nMFE($)', 'MAE(R)', 'MFE(R)',
              'Avg\nBars', 'Trades\n/Day', 'Net\nP&L']
    rows = [header]

    for csv_name in sorted_names:
        s = stats[csv_name]
        short = shorten_name(csv_name)

        rows.append([
            short,
            str(s['count']),
            f"{s['win_rate']:.0f}%",
            f"{s['rr']}:1",
            f"{s['avg_r']:.2f}",
            f"${s['ev']:.2f}",
            f"{s['pf']:.1f}",
            f"${s['avg_mae']:.2f}",
            f"${s['avg_mfe']:.2f}",
            f"{s['avg_mae_r']:.2f}",
            f"{s['avg_mfe_r']:.2f}",
            f"{s['avg_bars']:.1f}",
            f"{s['trades_per_day']:.2f}",
            f"${s['pnl']:.0f}",
        ])

    # Column widths — total ~7.1 inches
    col_w = [108, 28, 32, 30, 34, 36, 28, 38, 38, 34, 34, 28, 36, 38]

    # Paragraph styles for cells
    cell_hdr = ParagraphStyle("CellHdr", fontSize=6.5, leading=8, fontName="Helvetica-Bold",
                              textColor=WHITE, alignment=TA_CENTER)
    cell_body = ParagraphStyle("CellBody", fontSize=6.5, leading=8, fontName="Helvetica",
                               textColor=DARK, alignment=TA_CENTER)
    cell_name = ParagraphStyle("CellName", fontSize=6.5, leading=8, fontName="Helvetica-Bold",
                               textColor=DARK, alignment=TA_LEFT)

    # Convert to Paragraphs for wrapping
    tbl_data = []
    for ri, row in enumerate(rows):
        prow = []
        for ci, val in enumerate(row):
            if ri == 0:
                prow.append(Paragraph(val.replace('\n', '<br/>'), cell_hdr))
            elif ci == 0:
                prow.append(Paragraph(val, cell_name))
            else:
                prow.append(Paragraph(val, cell_body))
        tbl_data.append(prow)

    tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)

    # Color rows by EV — green tint for high EV, neutral for mid, light red for low
    tbl_style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, LTGRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]

    # Alternate row colors
    for ri in range(1, len(tbl_data)):
        csv_name = sorted_names[ri - 1]
        s = stats[csv_name]
        if s['avg_r'] >= 0.50:
            bg = HexColor("#e8f5e9")  # Strong green
        elif s['avg_r'] >= 0.20:
            bg = HexColor("#f1f8e9")  # Light green
        else:
            bg = HexColor("#fafafa")  # Neutral
        tbl_style_cmds.append(('BACKGROUND', (0, ri), (-1, ri), bg))

    tbl.setStyle(TableStyle(tbl_style_cmds))
    story.append(tbl)

    story.append(Spacer(1, 12))

    # ── Key Metrics Legend ──
    legend_style = ParagraphStyle("Legend", fontSize=7, leading=10, textColor=GRAY)
    story.append(Paragraph(
        "<b>N</b> = Total Trades  &bull;  <b>Win%</b> = Win Rate  &bull;  "
        "<b>R:R</b> = Risk:Reward Ratio  &bull;  <b>EV(R)</b> = Avg R-Multiple  &bull;  "
        "<b>EV($)</b> = Expected Value per Trade  &bull;  <b>PF</b> = Profit Factor<br/>"
        "<b>MAE</b> = Max Adverse Excursion (avg drawdown during trade)  &bull;  "
        "<b>MFE</b> = Max Favorable Excursion (avg peak unrealized profit)  &bull;  "
        "<b>Trades/Day</b> = Average Signals per Trading Day",
        legend_style))

    story.append(Spacer(1, 12))

    # ── Portfolio-level summary box ──
    story.append(Paragraph("Portfolio Summary", section_head))

    avg_trades_per_day = total_trades / total_days
    best_setup = max(stats.items(), key=lambda x: x[1]['ev'])
    most_active = max(stats.items(), key=lambda x: x[1]['count'])

    sum_rows = [
        ['Total Trades', f'{total_trades:,}', 'Trading Days', f'{total_days:,}'],
        ['Overall Win Rate', f'{overall_wr:.1f}%', 'Avg Trades/Day', f'{avg_trades_per_day:.2f}'],
        ['Total Net P&L', f'${total_pnl:,.2f}', 'Avg P&L/Trade', f'${total_pnl/total_trades:.2f}'],
        ['Best EV Setup', best_setup[0][:28], 'EV', f'${best_setup[1]["ev"]:.2f}/trade'],
        ['Most Active', most_active[0][:28], 'Trades', f'{most_active[1]["count"]}'],
    ]

    sum_cell = ParagraphStyle("SumCell", fontSize=8, leading=10, fontName="Helvetica", textColor=DARK)
    sum_lbl = ParagraphStyle("SumLbl", fontSize=8, leading=10, fontName="Helvetica-Bold", textColor=HexColor("#444444"))

    sum_tbl_data = []
    for row in sum_rows:
        sum_tbl_data.append([
            Paragraph(row[0], sum_lbl), Paragraph(row[1], sum_cell),
            Paragraph(row[2], sum_lbl), Paragraph(row[3], sum_cell),
        ])

    sum_tbl = Table(sum_tbl_data, colWidths=[120, 110, 120, 110])
    sum_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#f5f5f5")),
        ('GRID', (0, 0), (-1, -1), 0.3, LTGRAY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sum_tbl)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════
    #  GLOSSARY / ABBREVIATION KEY
    # ═══════════════════════════════════════════════════════

    story.append(Paragraph("Glossary &amp; Abbreviation Key", section_head))
    story.append(Paragraph(
        "All abbreviations and terms used in this report, explained for all experience levels.",
        ParagraphStyle("GlossDesc", fontSize=8, leading=11, textColor=GRAY, spaceAfter=8)))

    gloss_lbl = ParagraphStyle("GlossLbl", fontSize=7, leading=9.5, fontName="Helvetica-Bold", textColor=DARK)
    gloss_val = ParagraphStyle("GlossVal", fontSize=7, leading=9.5, fontName="Helvetica", textColor=HexColor("#444444"))

    glossary_items = [
        # Column headers / metrics
        ("N", "Total number of trades generated by a setup during the backtest period."),
        ("Win%  (Win Rate)", "Percentage of trades that were profitable. E.g., 80% means 8 out of 10 trades made money."),
        ("R:R  (Risk:Reward Ratio)", "How much you stand to gain relative to your risk. 1:1 means risking $1 to make $1. 0.5:1 means risking $1 to make $0.50."),
        ("EV($)  (Expected Value)", "Average dollar profit per trade. If EV = $0.50, you can expect to make $0.50 on average every time this setup fires."),
        ("EV(R)  (Expected R-Multiple)", "Average profit per trade measured in units of risk (R). EV(R) = 0.5 means each trade earns 0.5x your risk on average."),
        ("PF  (Profit Factor)", "Total gross profit divided by total gross loss. PF > 1.0 = profitable. PF > 2.0 = strong edge. PF > 10 = exceptional."),
        ("MAE  (Max Adverse Excursion)", "The worst drawdown experienced during a trade before it resolved. Lower = better. Shows how much pain you endure per trade."),
        ("MAE(R)", "MAE measured in R-units (multiples of risk). E.g., 0.5R means the trade went 0.5x your risk against you at its worst point."),
        ("MFE  (Max Favorable Excursion)", "The peak unrealized profit during a trade. Higher = better. Shows how far the trade ran in your favor before closing."),
        ("MFE(R)", "MFE measured in R-units. E.g., 1.5R means the trade ran 1.5x your risk in your favor at its peak."),
        ("MFE/MAE Ratio", "MFE divided by MAE. Higher is better. Ratios above 2.0x mean trades run twice as far in your favor as against you."),
        ("Net P&amp;L", "Total dollar profit or loss from all trades of a setup. The bottom line."),
        ("Avg Bars", "Average number of 5-minute bars a trade is held. E.g., 3 bars = 15 minutes average hold time."),
        ("Trades/Day", "Average number of signals generated per trading day across the entire backtest."),
        ("Avg Risk($)", "Average dollar risk per share on each trade (distance from entry to stop loss)."),
        # Setup name abbreviations
        ("F. (Fade)", "Trading the opposite direction of what the pattern suggests. Fading means betting the pattern will fail."),
        ("DB  (Double Bottom)", "A chart pattern where price tests the same low twice and bounces — a bullish reversal signal."),
        ("DT  (Double Top)", "A chart pattern where price tests the same high twice and drops — a bearish reversal signal."),
        ("HL  (Higher Low)", "The second low is above the first low — shows buyers are getting more aggressive."),
        ("LH  (Lower High)", "The second high is below the first high — shows sellers are getting more aggressive."),
        ("LL  (Lower Low)", "The second low is below the first low — shows sellers breaking through support."),
        ("HH  (Higher High)", "The second high is above the first high — shows buyers breaking through resistance."),
        ("MM  (Measured Move)", "A price target calculated by doubling the size of the initial trend move. A mathematical projection."),
        ("BO PB  (Breakout Pullback)", "Price breaks out of a range, then pulls back to test the breakout level as support/resistance."),
        ("3/4P  (3rd/4th Push)", "The 3rd or 4th push in a stair-step pattern — often the final exhaustion move before reversal."),
        ("Rev  (Reversal)", "A change in price direction from bearish to bullish or vice versa."),
        # Tier system
        ("Tier A", "Highest conviction setups — strongest backtested edge, most reliable signals."),
        ("Tier B", "Solid setups with a proven edge, slightly lower conviction than Tier A."),
        ("Tier C", "Usable setups with positive expectancy, but lower EV or fewer occurrences."),
        # General terms
        ("Signal Bar", "The bar (candle) that triggers the trade setup. Entry order is placed beyond this bar."),
        ("BUY STOP / SELL STOP", "An order type that activates only when price moves past a certain level, confirming direction."),
        ("LONG", "Buying — betting the price will go up."),
        ("SHORT", "Selling — betting the price will go down."),
        ("Scalp Target", "A nearby profit target for quick exits (typically the R:R ratio applied to your risk)."),
        ("Bars Held", "How many 5-minute candles the trade lasted from entry to exit."),
    ]

    # Build glossary as a 2-column table
    gloss_tbl_data = []
    for abbr, desc in glossary_items:
        gloss_tbl_data.append([
            Paragraph(abbr, gloss_lbl),
            Paragraph(desc, gloss_val),
        ])

    gloss_tbl = Table(gloss_tbl_data, colWidths=[115, 390])
    gloss_tbl.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -1), 0.3, LTGRAY),
    ] + [('BACKGROUND', (0, ri), (-1, ri), HexColor("#f9f9f9") if ri % 2 == 0 else WHITE)
         for ri in range(len(gloss_tbl_data))]))
    story.append(gloss_tbl)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════
    #  RESULTS PER TICKER
    # ═══════════════════════════════════════════════════════

    story.append(Paragraph("Results Per Ticker", section_head))
    story.append(Paragraph(
        "Performance breakdown by ticker symbol. Each ticker shows overall stats and per-setup results.",
        ParagraphStyle("TickDesc2", fontSize=8, leading=11, textColor=GRAY, spaceAfter=10)))

    for ticker in sorted(ticker_results.keys()):
        tr = ticker_results[ticker]

        ticker_hdr_s = ParagraphStyle("TickerHdr2", fontSize=13, leading=16, textColor=DARK,
                                     fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
        story.append(Paragraph(f"{ticker}", ticker_hdr_s))

        ticker_sum = (
            f"<b>{tr['count']}</b> trades  &bull;  <b>{tr['win_rate']:.1f}%</b> win rate  &bull;  "
            f"EV <b>${tr['ev']:.2f}</b>/trade  &bull;  PF <b>{tr['pf']:.2f}</b>  &bull;  "
            f"Net P&amp;L <b>${tr['pnl']:,.2f}</b>  &bull;  "
            f"Avg MAE <b>${tr['avg_mae']:.3f}</b>  &bull;  Avg MFE <b>${tr['avg_mfe']:.3f}</b>  &bull;  "
            f"{tr['active_days']} active days  &bull;  {tr['trades_per_day']:.2f} trades/day"
        )
        story.append(Paragraph(ticker_sum, ParagraphStyle("TickSum2", fontSize=7.5, leading=10, textColor=HexColor("#444444"))))
        story.append(Spacer(1, 6))

        ts_data = ticker_setup_stats.get(ticker, {})
        if ts_data:
            ts_header = ['Setup', 'N', 'Win%', 'EV($)', 'PF', 'Net P&L']
            ts_rows = [ts_header]

            sorted_setups = sorted(ts_data.keys(), key=lambda s: -ts_data[s]['pnl'])
            for sname in sorted_setups:
                sd = ts_data[sname]
                n = sd['count']
                wr = sd['wins'] / n * 100
                ev = sd['pnl'] / n
                gw = sum(r for r in sd['r_mult'] if r > 0)
                gl = abs(sum(r for r in sd['r_mult'] if r < 0))
                pf = gw / gl if gl > 0 else 999.0
                ts_rows.append([
                    shorten_name(sname),
                    str(n),
                    f"{wr:.0f}%",
                    f"${ev:.2f}",
                    f"{pf:.1f}",
                    f"${sd['pnl']:.2f}",
                ])

            ts_col_w = [170, 40, 50, 55, 45, 65]
            ts_tbl_data2 = []
            for ri, row in enumerate(ts_rows):
                prow = []
                for ci, val in enumerate(row):
                    if ri == 0:
                        prow.append(Paragraph(val, cell_hdr))
                    elif ci == 0:
                        prow.append(Paragraph(val, cell_name))
                    else:
                        prow.append(Paragraph(val, cell_body))
                ts_tbl_data2.append(prow)

            ts_tbl = Table(ts_tbl_data2, colWidths=ts_col_w, repeatRows=1)
            ts_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1a1a2e")),
                ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.3, LTGRAY),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ] + [('BACKGROUND', (0, ri), (-1, ri), HexColor("#fafafa") if ri % 2 == 0 else WHITE)
                 for ri in range(1, len(ts_tbl_data2))]))
            story.append(ts_tbl)

        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════
    #  DETAILED MAE/MFE BREAKDOWN TABLE
    # ═══════════════════════════════════════════════════════

    story.append(Paragraph("Detailed MAE / MFE Analysis", section_head))
    story.append(Paragraph(
        "Maximum Adverse Excursion (MAE) shows how far trades go against you before resolving. "
        "Maximum Favorable Excursion (MFE) shows the peak unrealized profit. Lower MAE and higher MFE = better edge.",
        ParagraphStyle("Desc", fontSize=8, leading=11, textColor=GRAY, spaceAfter=10)))

    det_header = ['Setup', 'Avg\nMAE($)', 'Max\nMAE($)', 'Avg\nMAE(R)', 'Avg\nMFE($)',
                  'Max\nMFE($)', 'Avg\nMFE(R)', 'MFE/\nMAE', 'Avg\nRisk($)', 'Avg\nBars']
    det_rows = [det_header]

    for csv_name in sorted_names:
        s = stats[csv_name]
        short = shorten_name(csv_name)

        mfe_mae_ratio = s['avg_mfe'] / s['avg_mae'] if s['avg_mae'] > 0.001 else 99.0

        det_rows.append([
            short,
            f"${s['avg_mae']:.3f}",
            f"${s['max_mae']:.2f}",
            f"{s['avg_mae_r']:.2f}R",
            f"${s['avg_mfe']:.3f}",
            f"${s['max_mfe']:.2f}",
            f"{s['avg_mfe_r']:.2f}R",
            f"{mfe_mae_ratio:.1f}x",
            f"${s['avg_risk']:.2f}",
            f"{s['avg_bars']:.1f}",
        ])

    det_col_w = [108, 42, 42, 42, 42, 42, 42, 38, 46, 32]

    det_tbl_data = []
    for ri, row in enumerate(det_rows):
        prow = []
        for ci, val in enumerate(row):
            if ri == 0:
                prow.append(Paragraph(val.replace('\n', '<br/>'), cell_hdr))
            elif ci == 0:
                prow.append(Paragraph(val, cell_name))
            else:
                prow.append(Paragraph(val, cell_body))
        det_tbl_data.append(prow)

    det_tbl = Table(det_tbl_data, colWidths=det_col_w, repeatRows=1)
    det_style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, LTGRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]
    for ri in range(1, len(det_tbl_data)):
        bg = HexColor("#fafafa") if ri % 2 == 0 else WHITE
        det_style_cmds.append(('BACKGROUND', (0, ri), (-1, ri), bg))

    det_tbl.setStyle(TableStyle(det_style_cmds))
    story.append(det_tbl)

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<b>MFE/MAE Ratio</b>: Higher is better — shows how much profit potential vs. drawdown risk per trade. "
        "Ratios above 2.0x indicate trades that run strongly in your favor relative to adverse movement.",
        legend_style))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════
    #  PAGES 3+: PATTERN VISUALS + FLOWCHARTS
    # ═══════════════════════════════════════════════════════

    story.append(Paragraph("Setup Pattern Guide", section_head))
    story.append(Spacer(1, 4))

    for i, s in enumerate(SETUPS):
        # Lookup stats for this setup
        setup_name = s["name"]
        # Try direct match first, then reverse CSV_NAME_MAP
        csv_name = None
        if setup_name in stats:
            csv_name = setup_name
        else:
            for k, v in CSV_NAME_MAP.items():
                if v == setup_name:
                    csv_name = k
                    break

        st = stats.get(csv_name) if csv_name else None

        story.append(Paragraph(s["name"], name_style))

        if st:
            dir_label = "LONG" if s["entry_dir"] == "LONG" else "SHORT"
            if "Fade" in s["name"]:
                dir_label += " (Fade)"
            line1 = f"{dir_label}  |  Tier {st['tier']}  |  {st['rr']}:1 R:R"
            line2 = (f"{st['win_rate']:.0f}% Win  |  {st['count']} trades  |  "
                     f"EV +${st['ev']:.2f}  |  PF {st['pf']:.1f}  |  "
                     f"MAE ${st['avg_mae']:.2f}  |  MFE ${st['avg_mfe']:.2f}  |  "
                     f"Avg {st['avg_bars']:.0f} bars")
            story.append(Paragraph(line1, tag_style))
            story.append(Paragraph(line2, stats_style))
        else:
            story.append(Paragraph("Stats not available", tag_style))

        story.append(Spacer(1, 2))
        story.append(PatternFlowable(s))

        if i < len(SETUPS) - 1:
            if (i + 1) % 2 == 0:
                story.append(PageBreak())
            else:
                story.append(Spacer(1, 18))

    # ═══════════════════════════════════════════════════════
    #  PYTHON SOURCE CODE PAGES
    # ═══════════════════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("Python Detection Code", title_style))
    story.append(Paragraph(
        "Source code for each of the 13 profitable setup detectors. "
        "Functions are called by the algo engine on every new 5-min bar.",
        sub_style))

    code_style = ParagraphStyle("Code", fontSize=5.8, leading=7.2, fontName="Courier",
                                 textColor=DARK, backColor=HexColor("#f5f5f5"),
                                 leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=4,
                                 borderPadding=6)
    code_title = ParagraphStyle("CodeTitle", fontSize=11, leading=14, textColor=DARK,
                                 fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
    code_file = ParagraphStyle("CodeFile", fontSize=7, leading=9, textColor=GRAY, spaceAfter=4)

    # Map each setup to its source code
    SETUP_CODE = _get_setup_code()

    for setup_name, code_info in SETUP_CODE.items():
        story.append(PageBreak())
        story.append(Paragraph(setup_name, code_title))
        story.append(Paragraph(f"Source: {code_info['file']}  &mdash;  {code_info['func']}", code_file))

        # Escape code for XML/Paragraph rendering
        code_text = code_info['code']
        code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Wrap in <pre> equivalent using Preformatted
        story.append(Preformatted(code_info['code'], code_style))

    doc.build(story)
    print(f"PDF saved to {OUTPUT}")


def _get_setup_code():
    """Extract source code for each of the 13 setups from the algo files."""
    base = "/sessions/awesome-festive-cray/mnt/BPA-Bot-1"
    code_map = {}

    # ── algo_engine.py: detect_double_bottoms_tops ──
    # Covers: Double Bottom, Higher Low DB, Lower Low DB (Fade LL DB),
    #         Double Top, Lower High DT, Higher High DT (Fade HH DT)
    code = _read_function(f"{base}/algo_engine.py", "def detect_double_bottoms_tops")
    if code:
        code_map["Double Bottom / Higher Low DB / Fade Lower Low DB"] = {
            'file': 'algo_engine.py', 'func': 'detect_double_bottoms_tops()', 'code': code}
        code_map["Double Top / Lower High DT / Fade Higher High DT"] = {
            'file': 'algo_engine.py', 'func': 'detect_double_bottoms_tops()', 'code': code}

    # ── advanced_setups.py: detect_wedge_patterns ──
    code = _read_function(f"{base}/user_algos/advanced_setups.py", "def detect_wedge_patterns")
    if code:
        code_map["Wedge Bottom (3 Pushes Down) / Wedge Top (3 Pushes Up)"] = {
            'file': 'user_algos/advanced_setups.py', 'func': 'detect_wedge_patterns()', 'code': code}

    # ── best_setups.py: detect_consecutive_climaxes ──
    code = _read_function(f"{base}/user_algos/best_setups.py", "def detect_consecutive_climaxes")
    if code:
        code_map["Fade Consecutive Sell Climaxes / Buy Climaxes"] = {
            'file': 'user_algos/best_setups.py', 'func': 'detect_consecutive_climaxes()', 'code': code}

    # ── best_setups.py: detect_breakout_pullback ──
    code = _read_function(f"{base}/user_algos/best_setups.py", "def detect_breakout_pullback")
    if code:
        code_map["Fade Bull Breakout Pullback"] = {
            'file': 'user_algos/best_setups.py', 'func': 'detect_breakout_pullback()', 'code': code}

    # ── range_setups.py: detect_bear_stairs ──
    code = _read_function(f"{base}/user_algos/range_setups.py", "def detect_bear_stairs")
    if code:
        code_map["Fade Bear Stairs Reversal (3rd/4th Push)"] = {
            'file': 'user_algos/range_setups.py', 'func': 'detect_bear_stairs()', 'code': code}

    # ── best_setups.py: detect_exhaustive_climax_at_mm ──
    code = _read_function(f"{base}/user_algos/best_setups.py", "def detect_exhaustive_climax_at_mm")
    if code:
        code_map["Fade Exhaustive Bull Climax at MM"] = {
            'file': 'user_algos/best_setups.py', 'func': 'detect_exhaustive_climax_at_mm()', 'code': code}

    # ── algo_engine.py: detect_wedges (core wedge detection) ──
    code = _read_function(f"{base}/algo_engine.py", "def detect_wedges")
    if code:
        code_map["Wedge Bottom / Wedge Top (Core Detection)"] = {
            'file': 'algo_engine.py', 'func': 'detect_wedges()', 'code': code}

    return code_map


def _read_function(filepath, func_sig):
    """Read a Python function from a file, starting at the def line until the next def or EOF."""
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    start = None
    for i, line in enumerate(lines):
        if func_sig in line:
            start = i
            break

    if start is None:
        return None

    # Read until next top-level def or end of file
    func_lines = [lines[start]]
    for i in range(start + 1, len(lines)):
        line = lines[i]
        # Stop at next top-level function definition
        if line.startswith("def ") and not line.startswith("    "):
            break
        func_lines.append(line)

    # Strip trailing blank lines
    while func_lines and func_lines[-1].strip() == '':
        func_lines.pop()

    return ''.join(func_lines)


if __name__ == "__main__":
    build_pdf()
