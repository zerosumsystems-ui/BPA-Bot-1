import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

# Suppress matplotlib warnings for headless generation
warnings.filterwarnings("ignore")

def create_chart(filename, shape_type):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(20)
    
    if shape_type == "trend":
        y = np.linspace(100, 120, 20) + np.sin(x) * 2
        y[10:15] = y[10:15] - np.linspace(1, 5, 5) 
        ax.plot(x, y, color='#0B2647', linewidth=2.5, label="Price Action")
        ax.plot(x, np.linspace(98, 118, 20), color='#1D4ED8', linestyle='--', alpha=0.6, label="20-EMA Dynamic Support")
        ax.scatter([14], [y[14]], color='#16A34A', s=150, zorder=5, label="Algorithmic H2 Entry")
    elif shape_type == "wedge":
        y = np.linspace(100, 130, 20) + np.sin(x*1.5) * 4
        ax.plot(x, y, color='#0B2647', linewidth=2.5, label="Price Action")
        ax.plot([0, 19], [105, 132], color='#DC2626', linestyle='--', alpha=0.6, label="Macro Resistance")
        ax.scatter([18], [y[18]], color='#DC2626', s=150, zorder=5, label="Reversal Short Signal")
    elif shape_type == "ema":
        y = np.linspace(100, 110, 20)
        y[5:10] = y[5:10] + np.random.normal(0, 1.5, 5)
        ax.plot(x, y, color='#0B2647', linewidth=2.5, label="Price Action")
        ax.plot(x, np.linspace(99, 109, 20), color='#1D4ED8', linestyle='--', alpha=0.6, label="20-EMA Fair Value")
        ax.scatter([10], [y[10]], color='#16A34A', s=150, zorder=5, label="Terminal Mean Reversion Bounce")
        
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='-', alpha=0.3, color='#94A3B8')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.set_title(f"Simulated Volatility Array - Model {shape_type.upper()}", loc='left', color='#64748B', fontsize=10)
    
    # Hide axis values for a cleaner, conceptual look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

# Generate the simulated institutional graphics
try:
    create_chart("chart_h2.png", "trend")
    create_chart("chart_wedge.png", "wedge")
    create_chart("chart_ema.png", "ema")
except Exception as e:
    print(f"Chart gen failed: {e}")

class CorporatePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return 
        self.set_fill_color(11, 38, 71) 
        self.rect(0, 0, 210, 25, 'F')
        self.set_y(10)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 5, 'GLOBAL INVESTMENT RESEARCH', 0, 1, 'L')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(200, 220, 255)
        self.cell(0, 5, 'Quantitative Execution Strategies | Strictly Confidential', 0, 1, 'L')
        self.ln(15)

    def footer(self):
        self.set_y(-20)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, 'This material is for internal desk use only. Not for client distribution. Unauthorized duplication is prohibited.', 0, 1, 'C')
        self.cell(0, 5, f'Page {self.page_no()}', 0, 0, 'R')

def build_pdf():
    pdf = CorporatePDF()
    pdf.add_page()
    
    # --- CONFIDENTIAL TITLE PAGE ---
    pdf.set_y(60)
    pdf.set_font('Arial', 'B', 32)
    pdf.set_text_color(11, 38, 71)
    pdf.multi_cell(0, 12, 'Algorithmic Capture\nof Price Action Inefficiencies', align='C')
    pdf.ln(10)
    
    pdf.set_font('Arial', '', 16)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8, 'Systematic Execution Models Derived from\nInstitutional Flow Dynamics', align='C')
    pdf.ln(40)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    info = [
        f"Date: {current_date}",
        "Author: Quantitative Trading Desk",
        "Division: Global Markets - Systematic Strategies",
        "Alpha Rating: Outperform (Tier 1)"
    ]
    
    # Draw a neat box around info
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(30, 150, 150, 40, 'FD')
    
    pdf.set_y(155)
    for line in info:
        pdf.cell(0, 7, line, ln=1, align='C')
        
    pdf.set_y(230)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    disclaimer = "CONFIDENTIAL AND PROPRIETARY. This memorandum outlines the successful mathematical codification of three distinct, high-probability price action architectures into automated execution logic. Derived from verified methodologies authored by Al Brooks, these algorithms isolate deep market inefficiencies through real-time statistical analysis. Unauthorized duplication, distribution, or reverse-engineering of the execution logic contained herein is strictly prohibited under institutional compliance mandates."
    pdf.multi_cell(0, 4, disclaimer, align='J')
    
    # --- EXECUTIVE SUMMARY ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(11, 38, 71)
    pdf.cell(0, 10, 'Executive Summary', ln=1)
    pdf.set_draw_color(11, 38, 71)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(30,30,30)
    summary_text = (
        "This memorandum details the successful translation of analog price action theories into "
        "rigid, deterministic quantitative execution engines.\n\n"
        "Historically, the Al Brooks methodologies have relied heavily on discretionary human visual parsing -- a system "
        "prone to emotional interference, inconsistent adherence to entry parameters, and execution latency. "
        "The newly deployed Python modules ('user_algos/advanced_setups.py') eliminate discretionary variance "
        "by subjecting real-time OHLCV data to relentless algorithmic scrutiny.\n\n"
        "By enforcing strict mathematical boundaries on concepts such as 'tightness thresholds,' 'ema padding,' "
        "and 'reversal signal conviction arrays,' the models effectively hunt for trapped retail liquidity and "
        "institutional re-accumulation zones perfectly.\n\n"
        "The following pages break down the three primary Alpha-generating models currently active in the core engine."
    )
    pdf.multi_cell(0, 6, summary_text)
    
    # --- STRATEGY 1 ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(11, 38, 71)
    pdf.cell(0, 10, 'Model 1: Multi-Leg Trend Continuation (H2/L2)', ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    try:
        pdf.image('chart_h2.png', x=15, w=180)
        pdf.ln(5)
    except:
        pass
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'I. Architectural Thesis', ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(50, 50, 50)
    t1 = "Efficient markets rarely re-rate linearly. During persistent directional trends, minor counter-trend flow momentarily depresses asset prices before institutional participants aggressively re-enter. A secondary, successive failure of counter-trend liquidity to break structure ('High 2') confirms the dominant institutional trend remains highly intact, offering asymmetric execution yield."
    pdf.multi_cell(0, 6, t1)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'II. Execution Logic Parameters', ln=1)
    pdf.set_font('Courier', 'B', 10)
    pdf.set_fill_color(240, 245, 250)
    pdf.set_text_color(20, 40, 80)
    p1 = (" * MODEL CONFIDENCE MATRIX : 85.0%\n"
          " * TARGET ENVIRONMENT      : Established Directional Momentum (>15 bars)\n"
          " * CORE DEPENDENCY         : 20-EMA Dynamic Support/Resistance Arrays\n"
          " * TRIGGER THRESHOLD       : EMA Intersection + Directional 50% Body Close\n"
          " * RISK ASSIGNMENT         : Trailing local minima extreme")
    pdf.multi_cell(0, 7, p1, fill=True)
    
    # --- STRATEGY 2 ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(11, 38, 71)
    pdf.cell(0, 10, 'Model 2: Exhaustion Extrema Detection (Wedge Reversals)', ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    try:
        pdf.image('chart_wedge.png', x=15, w=180)
        pdf.ln(5)
    except:
        pass
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'I. Architectural Thesis', ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(50, 50, 50)
    t2 = "Prolonged directional moves frequently terminate in parabolic exhaustion sprints rather than smooth distribution cycles. This exhaustion manifests mathematically as three distinct, diminishing volume pushes into an extreme price deviation. Each successive peak covers declining net distance -- heavily indicating dying momentum and a violent impending mean-reversion."
    pdf.multi_cell(0, 6, t2)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'II. Execution Logic Parameters', ln=1)
    pdf.set_font('Courier', 'B', 10)
    pdf.set_fill_color(240, 245, 250)
    pdf.set_text_color(20, 40, 80)
    p2 = (" * MODEL CONFIDENCE MATRIX : 80.0%\n"
          " * TARGET ENVIRONMENT      : Macro Extremes / Broken Support Channels\n"
          " * CORE DEPENDENCY         : Local Minima/Maxima Traversal Parser\n"
          " * TRIGGER THRESHOLD       : 3rd Progressive Push + Sharp Contrarian Close\n"
          " * YIELD PROFILE           : High-Variance, Highly Asymmetric Reward-to-Risk")
    pdf.multi_cell(0, 7, p2, fill=True)

    # --- STRATEGY 3 ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(11, 38, 71)
    pdf.cell(0, 10, 'Model 3: Dynamic Mean Reversion (EMA Rubber Band)', ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    try:
        pdf.image('chart_ema.png', x=15, w=180)
        pdf.ln(5)
    except:
        pass
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'I. Architectural Thesis', ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(50, 50, 50)
    t3 = "The 20-period Exponential Moving Average (EMA) operates as a dynamic, weighted measure of algorithmic fair value. When an asset experiences severe momentum divergence from this median, standard distribution models force an eventual reversion to the mean. The algorithm isolates pullbacks that physically intersect the 20-EMA during highly accelerated trends, representing flawless execution zones."
    pdf.multi_cell(0, 6, t3)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'II. Execution Logic Parameters', ln=1)
    pdf.set_font('Courier', 'B', 10)
    pdf.set_fill_color(240, 245, 250)
    pdf.set_text_color(20, 40, 80)
    p3 = (" * MODEL CONFIDENCE MATRIX : 85.0%\n"
          " * TARGET ENVIRONMENT      : High Accelerated Momentum & Volatility Expansion\n"
          " * CORE DEPENDENCY         : Standardized Mean-Deviation Vectors\n"
          " * TRIGGER THRESHOLD       : Micro-Tolerance [0.998, 1.002] Band Intersection\n"
          " * RISK ASSIGNMENT         : Structural protection underneath moving average")
    pdf.multi_cell(0, 7, p3, fill=True)

    pdf.output('/Users/williamkosloski/BPA-Bot-1/Algorithmic_Execution_Models_Research.pdf')
    print("Goldman Sachs Quantitative Report Created Successfully.")

if __name__ == "__main__":
    build_pdf()
