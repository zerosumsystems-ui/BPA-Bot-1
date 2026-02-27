"""
generate_algo.py — LLM-powered Algo Rule Generator

Uses Gemini to read the Brooks encyclopedia + your training corrections
and generate new Python detection functions for algo_engine.py.

The LLM becomes your algo programmer — it writes fast code, not slow analysis.

Usage:
    python generate_algo.py                     # Generate new rules from encyclopedia
    python generate_algo.py --from-corrections  # Learn from your training corrections
    python generate_algo.py --improve           # Analyze accuracy and improve existing rules
"""

import os
import sys
import json
import pathlib
import argparse
import time

# ─────────────────────────── CONFIG ──────────────────────────────────────────

DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "."))
ENCYCLOPEDIA_PATH = DATA_DIR / "brooks_encyclopedia_learnings.md"
TRAINING_CSV = DATA_DIR / "training_data.csv"
ALGO_ENGINE_PATH = pathlib.Path("algo_engine.py")
GENERATED_RULES_PATH = DATA_DIR / "generated_rules.py"

CODE_GEN_PROMPT = """You are an expert Python developer and Al Brooks Price Action trader.

Your job is to write Python detection functions for a fast algorithmic trading engine.
These functions run on raw OHLC bar data (no LLM at runtime — pure code, must be fast).

## Available Infrastructure

You have access to these classes and functions from algo_engine.py:

```python
@dataclass
class Bar:
    idx: int        # Bar number (1-indexed)
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Properties: is_bull, is_bear, is_doji, body, range, upper_tail, lower_tail,
    #             body_top, body_bottom, closes_near_high, closes_near_low
    # Methods: is_big(avg_range), is_strong_bull(avg_range), is_strong_bear(avg_range),
    #          is_inside(prev_bar), is_outside(prev_bar)

@dataclass
class Setup:
    setup_name: str
    entry_bar: int
    entry_price: float
    order_type: str   # "Stop" or "Limit"
    confidence: float  # 0.0 to 1.0

# Available helpers:
# compute_ema(bars, period=20) -> list[float]
# find_swing_highs(bars, lookback=3) -> list[int]
# find_swing_lows(bars, lookback=3) -> list[int]
```

## Rules for Generated Code

1. Each function must accept `(bars: list[Bar], ema: list[float])` and return `list[Setup]`
2. Functions must be pure Python — no API calls, no imports beyond numpy
3. Use the Bar properties and methods — don't recompute things
4. Set realistic confidence values (0.4-0.7 range)
5. Include docstrings explaining the Al Brooks logic
6. Use stop orders for with-trend entries, limit orders for counter-trend
7. Handle edge cases (insufficient bars, zero ranges, etc.)

## Output Format

Return ONLY valid Python code. No markdown, no explanation. Just the function definitions.
Each function should be named `detect_<pattern_name>` and follow the pattern above.
"""

CORRECTION_LEARNING_PROMPT = """You are analyzing trading bot corrections to improve the detection algorithms.

Below are rows from training_data.csv where a teacher corrected the bot's analysis.
The "bot_*" columns show what the algo detected. The "override_*" columns show the
teacher's corrections. The "teacher_notes" column has explanations.

Analyze the PATTERNS in the corrections:
1. What setups does the bot consistently get wrong?
2. What setups does the bot miss entirely?
3. What day types does the bot misclassify?

Then write IMPROVED Python detection functions that fix these patterns.
Use the same function signature: `(bars: list[Bar], ema: list[float]) -> list[Setup]`

Return ONLY valid Python code with improved detection functions.
"""

ACCURACY_PROMPT = """You are analyzing the accuracy of trading pattern detection functions.

Here is the current algo_engine.py code:
```python
{engine_code}
```

Here are the most recent training corrections (bot vs teacher):
```csv
{corrections}
```

Analyze where the current code fails and write IMPROVED versions of the detection
functions. Focus on:
1. Reducing false positives (bot detected setup that wasn't there)
2. Reducing false negatives (bot missed a setup the teacher identified)
3. Better confidence calibration

Return ONLY valid Python function definitions. No markdown.
"""


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return key


def generate_from_encyclopedia():
    """Read encyclopedia and generate new pattern detection functions."""
    from google import genai
    from google.genai import types

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found")
        sys.exit(1)

    print("📚 Reading encyclopedia...")
    encyclopedia = ENCYCLOPEDIA_PATH.read_text(encoding="utf-8")

    # Take the detailed slide learnings section (most actionable rules)
    # Limit to ~30K chars to stay within context
    if len(encyclopedia) > 30000:
        encyclopedia = encyclopedia[:30000]

    prompt = f"""{CODE_GEN_PROMPT}

## Al Brooks Encyclopedia (pattern rules to implement):

{encyclopedia}

Now write Python detection functions for the patterns described above.
Focus on the most tradeable, highest-probability patterns first:
- Major Trend Reversal (MTR) detection
- Failed Breakout (Bull/Bear Trap) detection
- Buy/Sell Climax detection
- Opening Reversal detection
- Head & Shoulders detection
- Cup and Handle detection
"""

    print("🤖 Asking Gemini to write algo rules...")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.1,  # Low temp for code generation
    )

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=config,
    )

    code = response.text.strip()
    # Clean markdown fences if present
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    code = code.strip()

    # Validate it's actual Python
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        print(f"⚠️  Generated code has syntax error: {e}")
        print("Saving anyway for manual review...")

    GENERATED_RULES_PATH.write_text(code, encoding="utf-8")
    print(f"✅ Generated rules saved to {GENERATED_RULES_PATH}")
    print(f"   ({len(code)} chars, {code.count('def ')} functions)")
    print(f"\nNext steps:")
    print(f"  1. Review the generated code in {GENERATED_RULES_PATH}")
    print(f"  2. Copy approved functions into algo_engine.py")
    print(f"  3. Run: python algo_engine.py  (to test)")


def generate_from_corrections():
    """Analyze training corrections and generate improved rules."""
    import pandas as pd
    from google import genai
    from google.genai import types

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found")
        sys.exit(1)

    if not TRAINING_CSV.exists():
        print("No training_data.csv found. Approve some charts first!")
        sys.exit(1)

    print("📊 Loading training corrections...")
    df = pd.read_csv(TRAINING_CSV)

    # Filter to rows where teacher made corrections
    has_override = df[
        df["override_setup_1"].notna() |
        df["override_day_type"].notna() |
        df["teacher_notes"].notna()
    ]

    if has_override.empty:
        print("No corrections found in training data. The bot is doing great! 🎉")
        sys.exit(0)

    print(f"   Found {len(has_override)} corrected charts")

    # Convert to string for the prompt (limit to recent 50)
    corrections_str = has_override.tail(50).to_csv(index=False)

    prompt = f"""{CODE_GEN_PROMPT}

{CORRECTION_LEARNING_PROMPT}

## Training Corrections:

{corrections_str}

Write improved or new detection functions based on these corrections.
"""

    print("🤖 Asking Gemini to learn from your corrections...")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.1)

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=config,
    )

    code = response.text.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]

    output_path = DATA_DIR / "improved_rules.py"
    output_path.write_text(code.strip(), encoding="utf-8")
    print(f"✅ Improved rules saved to {output_path}")
    print(f"   ({code.count('def ')} functions generated from {len(has_override)} corrections)")


def improve_accuracy():
    """Full accuracy improvement cycle: read engine + corrections, write better code."""
    import pandas as pd
    from google import genai
    from google.genai import types

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found")
        sys.exit(1)

    engine_code = ALGO_ENGINE_PATH.read_text(encoding="utf-8")
    corrections_str = ""
    if TRAINING_CSV.exists():
        df = pd.read_csv(TRAINING_CSV)
        corrections_str = df.tail(50).to_csv(index=False)

    prompt = ACCURACY_PROMPT.format(
        engine_code=engine_code,
        corrections=corrections_str if corrections_str else "(No training data yet)"
    )

    print("🔬 Analyzing accuracy and generating improvements...")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.1)

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=config,
    )

    code = response.text.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]

    output_path = DATA_DIR / "accuracy_improvements.py"
    output_path.write_text(code.strip(), encoding="utf-8")
    print(f"✅ Improvements saved to {output_path}")
    print(f"   Review and merge the best functions into algo_engine.py")


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-powered algo rule generator")
    parser.add_argument("--from-corrections", action="store_true",
                        help="Generate rules by learning from training corrections")
    parser.add_argument("--improve", action="store_true",
                        help="Analyze accuracy and improve existing rules")
    args = parser.parse_args()

    if args.from_corrections:
        generate_from_corrections()
    elif args.improve:
        improve_accuracy()
    else:
        generate_from_encyclopedia()


if __name__ == "__main__":
    main()
