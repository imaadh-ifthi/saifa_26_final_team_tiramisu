# Streamlit Dashboard UI Design Choices & Architectural Rationale

**Project**: TiraMisU — Serendib University Endowment Fund Investment Proposal  
**Affiliation**: University of Peradeniya | Quant Edge 1.0  
**Tagline**: *Thoughtful allocation. Transparent risk.*  
**Target Reference Capital**: USD 10,000,000  
**Target Audience**: Investment Committee, Board of Trustees, and Regulatory Risk Auditors  

---

## 1. Executive Summary & Design Philosophy

The TiraMisU Streamlit dashboard was engineered as an **audit-grade institutional decision support system**, not a marketing showcase or casual retail tracker. When presenting a \$10M university endowment allocation to a fiduciary investment committee, visual aesthetics directly influence credibility, governance rigor, and analytical clarity.

### Core Design Tenets
1. **Intellectual Honesty Over Cosmetic Optimism**:
   - The dashboard makes an uncompromising distinction between **forward-looking model estimates** and **realized historical performance**.
   - If an optimization constraint collapses the efficient frontier to a single point, the UI displays a single honest point with diagnostic commentary rather than interpolating a fabricated curve.
   - If risk models fail statistical validation over multi-year crisis periods, the failure is highlighted prominently alongside the regulatory green traffic light, explaining the mathematical distinction between 250-day regulatory windows and long-term tail clustering.
2. **Strict Separation of Analytical Horizons**:
   - **Current Proposed Portfolio** (fitted across 2007–2026; analyzed via retrospective historical replay).
   - **Frozen Validation Portfolio** (fitted strictly on 2007–2017 training data; evaluated out-of-sample across 2018–2026 with zero look-ahead leakage).
   - The UI enforces this separation via color-coded badges, disabled toggles, and locked controls.
3. **Institutional Elegance Without "AI Clutter"**:
   - All decorative emojis, cartoonish indicators, and synthetic visual noise were systematically excised.
   - Replaced with clean typography, tabular numerals, semantic callouts, and restrained financial charts.
4. **Zero-Latency Offline Execution**:
   - Operates entirely from pre-computed artifacts (`outputs/tables/` and `outputs/models/`).
   - No external API downloads or long-running optimizations occur during runtime, guaranteeing fast, repeatable committee presentations.

---

## 2. Visual Identity & Theme Architecture

The visual layer is governed by `app/theme.py`, `app/design.py`, and `.streamlit/config.toml`.

### 2.1 Dual Espresso/Ivory & Warm Charcoal Palette
Rather than standard stark white or terminal-like pure black, the dashboard utilizes a bespoke palette reflecting university heritage, warm parchment, and institutional espresso:

| Token | Light Theme (`#fffdf9`) | Dark Theme (`#211f1b`) | Institutional Purpose |
|---|---|---|---|
| `navy` | `#302720` (Deep Espresso) | `#f4eee5` (Cream Ivory) | High-contrast primary headings and KPI figures |
| `slate` | `#51483f` (Warm Slate) | `#ddd3c6` (Soft Sand) | Body text and secondary metrics |
| `muted` | `#70665b` (Taupe) | `#b6ab9c` (Muted Khaki) | Captions, footnotes, table metadata, and axis labels |
| `light_bg` | `#f3eee6` (Warm Linen) | `#292722` (Warm Charcoal) | Page canvas, callout backgrounds, metric card fills |
| `card_bg` | `#fffdf9` (Parchment White) | `#211f1b` (Deep Espresso) | Elevated card surfaces and Plotly plot paper |
| `border` | `#d9d0c4` (Pale Sandstone) | `#49443c` (Muted Umber) | Subtle dividers, metric container borders, table grids |
| `teal` | `#08786d` (Deep Sea Teal) | `#70d8c3` (Luminous Mint) | Brand accent, primary buttons, Current portfolio badges |
| `teal_light`| `#e0f0eb` | `#193d36` | Active sidebar item fill, badge highlights |
| `blue` | `#315fb5` (Cobalt) | `#9ebdff` (Soft Azure) | Frozen Out-of-Sample (OOS) validation badge and metrics |
| `blue_light`| `#e6edf9` | `#24344d` | OOS callout backgrounds |
| `amber` | `#8c5b13` (Cognac Amber) | `#edc47c` (Warm Ochre) | Cautionary warnings, Basel yellow/amber zone |
| `amber_light`| `#fff0d2` | `#42331e` | Warning callout box fills |
| `red` | `#b33a3a` (Crimson) | `#ffaaa3` (Coral Red) | VaR breaches, maximum drawdowns, Basel red zone |
| `red_light` | `#fbe8e5` | `#482b2a` | Error/breach alert backgrounds |
| `green` | `#237044` (Forest Green) | `#97d5a6` (Sage Green) | Compliant constraints, Basel green zone, hedging gains |
| `green_light`| `#e4f1e5` | `#253c2d` | Success alert backgrounds |

### 2.2 Rationale Behind Palette Choices
- **Elimination of Pure Black (`#000000`) and Pure White (`#ffffff`)**: High-contrast pure black on stark white causes eye fatigue during extended committee reviews. Off-white linen and deep espresso create an organic, print-like academic report feel.
- **Color-Coded Analysis Basis**:
  - **Teal**: Associated with the primary proposal and current retrospective data.
  - **Blue**: Associated exclusively with out-of-sample frozen validation, training windows, and statistical backtests.
- **Accessible Contrast Compliance (WCAG AA)**:
  - All text colors (`navy`, `slate`, `muted`) against their respective card and panel backgrounds have been verified by automated contrast tests (`tests/test_tiramisu_theme.py`) to exceed the **4.5:1** minimum contrast ratio requirement.

### 2.3 Session-Local Theme Adapter (`ThemePalette`)
```python
# app/theme.py
class ThemePalette(Mapping):
    def __getitem__(self, key):
        return (DARK if theme_name() == "dark" else LIGHT)[key]
```
- **Why**: Traditional Streamlit implementations store styling in global state or mutate shared dictionaries. In a multi-user or tabbed presentation, mutating global dictionaries causes race conditions and visual leakage. `ThemePalette` dynamically queries `st.context.theme.type` on every interaction, ensuring 100% thread-safe isolation.

### 2.4 Removal of Emojis
- **Why**: Modern generative AI interfaces frequently overuse emojis (🚀, 📈, 💡, 🏛️, 🔒, ⚠️) on every header and button. In an endowment board meeting, this aesthetic immediately signals amateurism or unvetted AI generation. Emojis were replaced with formal, institutional text labels, uppercase section numbering, and clean SVG badges.

### 2.5 Typography & Tabular Numerals
- FONT: Inter / `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`.
- All metric displays (`[data-testid="stMetricValue"]`) are formatted with `font-variant-numeric: tabular-nums`.
- **Why**: Monospaced tabular alignment ensures that multi-million dollar portfolio values, percentage basis points, and drawdown ratios align vertically without layout shifting or visual jitter.

---

## 3. Global Information Architecture & Navigation

### 3.1 Structural Navigation
The application organizes complex quantitative workflows into six numbered, sequential views:

```
01  Investment briefing          (app/views/overview.py)
02  Allocation & performance     (app/views/portfolio.py)
03  Risk explorer                (app/views/risk.py)
04  Model reliability            (app/views/validation.py)
05  Crisis & contributions       (app/views/stress.py)
06  Evidence & assumptions       (app/views/assumptions.py)
```

- **Two-Digit Indexing (`01` to `06`)**: Reflects executive briefing packs and regulatory dossiers.
- **Sidebar Integration**: Placed in `st.sidebar.radio` with custom CSS styling to highlight the active view in soft brand teal (`PALETTE['teal_light']`).
- **Overview Decision Routes**: The Overview page provides direct shortcut buttons to key decision branches:
  - *"Explore the allocation →"* (Jumps to 02)
  - *"Examine crisis losses →"* (Jumps to 05)
  - *"Check model reliability →"* (Jumps to 04)

### 3.2 Global Sidebar Controls & Guardrails
- **Analysis Portfolio Basis Selector**:
  - Available on views where comparisons are meaningful (Portfolio and Risk).
  - Explicitly disabled and locked on **Model Validation** (locked to Frozen OOS) and **Stress Testing** (locked to Current Allocation).
  - Includes a helper caption clarifying the exact sample dates of each basis.
- **Confidence Level Slider**:
  - Standardized discrete steps: `90%`, `95%`, `99%`.
  - Default: `95%` on Risk Explorer (standard institutional market risk); `99%` on Model Validation (Basel regulatory standard).
- **Primary Risk Methodology Selector**:
  - Dynamically populates based on models available for the selected portfolio basis (`Historical Simulation`, `EWMA Parametric Normal`, `Bootstrap Monte Carlo`, and `EVT-GPD`).
  - Helper text explicitly notes: *"Changing methodology filters risk estimates; it does not alter portfolio weights."*
- **Audit & Cache Controls**:
  - **"Reload Results Cache"** button: Calls `clear_app_cache()` and re-reads disk artifacts without restarting the server.
  - **Data Cutoff Status Panel**: Displays exact timestamps for raw price availability, complete return series, and evaluated risk models.
  - **Configuration Discrepancy Detector**: Automatically checks if `config.yaml` parameters differ from saved model run metadata.

---

## 4. Component System & Formatting Conventions

Governed by `app/components.py` and `app/formatting.py`:

### 4.1 Reusable UI Primitives
1. **`render_header(title, subtitle)`**:
   - Institutional page title with strict negative letter-spacing (`-0.02em`) and an understated taupe divider line.
2. **`render_basis_badge(basis_label)`**:
   - Persistent badge placed immediately beneath headers.
   - Teal pill for Retrospective Replay; Blue pill for Frozen Out-of-Sample Validation.
   - **Why**: Eliminates ambiguity regarding whether a KPI or chart represents in-sample backtesting or genuine forward evaluation.
3. **`render_callout(text, level)`**:
   - Semantic notices (`info`, `warning`, `error`, `success`) using native Streamlit alerts.
   - Used for fiduciary warnings, data caveats, and reconciliation notes.
4. **`render_traffic_light_badge(traffic_status, exceptions, observations)`**:
   - Renders regulatory Basel Accord traffic light badges:
     - **GREEN** (0–4 exceptions): Normal capital requirements.
     - **YELLOW / AMBER** (5–9 exceptions): Scaled capital penalty.
     - **RED** (10+ exceptions): Presumption of flawed risk model.
     - **INSUFFICIENT DATA** (<250 observations): Regulatory disqualification.
5. **`render_download_button(df, filename, label)`**:
   - Attached to every data table, allowing one-click CSV export for external risk auditing in Excel/R/Python.

### 4.2 Formatting Guardrails (`app/formatting.py`)
- **`format_currency(val, decimals=0, include_sign=False)`**:
  - Formats negative currency as `-$120,000` rather than `$-120,000` or `($120,000)` to ensure consistent mathematical readability.
- **`format_percent(val, decimals=2, include_sign=False)`**:
  - Transforms decimals (`0.065`) to percentage strings (`6.50%`). Handles zero values robustly without floating-point artifacts (`-0.00%`).
- **`format_p_value(val, decimals=4)`**:
  - **Why**: In statistical backtesting, rounding a p-value of `0.00003` to `0.0000` misleadingly implies an exact probability of zero. `format_p_value` displays `<0.0001` whenever values fall below threshold.
- **`parse_boolean(val)`**:
  - **Why**: In Python data pipelines, reading strings like `"False"` from CSV files evaluates to boolean `True` if using standard `bool()`. `parse_boolean` explicitly evaluates `"false"`, `"0"`, `"no"`, and `nan` to `False`.

---

## 5. Charting System & Data Visualizations

All interactive charts in `app/charts.py` are built on Plotly and obey institutional finance standards:

### 5.1 Unified Institutional Layout (`apply_institutional_layout`)
- **Background**: Dynamic theme matching (`card_bg` for paper, `light_bg` for plot area).
- **Gridlines & Zero-Lines**: Subtle borders (`PALETTE['border']`), with strong zero-lines (`PALETTE['muted']`) to ground positive and negative returns.
- **Legend**: Horizontal orientation above the plot (`y=1.02, x=1.0`), preventing chart plot compression on narrower laptop screens.
- **Hovermode**: `x unified` crosshair hover with clean, non-overlapping formatting.

### 5.2 Deterministic Asset Color Registry (`ASSET_REGISTRY`)
Every asset ETF has a permanent, semantically meaningful color across all views:
- **SPY** (US Large Cap): Institutional Blue (`#5585bd`)
- **EFA** (Developed Intl): Cornflower Blue (`#6987d5`)
- **EEM** (Emerging Mkts): Cyan Slate (`#56a9bf`)
- **AGG** (US Aggregate Bonds): Emerald / Teal (`#0d9488`) — conveys stability/defense.
- **GLD** (Gold Bullion): Warm Amber Gold (`#d97706`) — conveys precious metal/real asset.
- **XLE** (Energy Sector): Crimson Red (`#dc2626`) — conveys commodity cyclicality.
- **XLK** (Technology Sector): Amethyst Purple (`#a581cc`) — conveys growth/tech.

*Why*: Changing asset colors between pages forces cognitive re-mapping. A single global registry guarantees SPY is always blue and GLD is always gold everywhere in the application.

### 5.3 Specialized Chart Innovations

#### A. Allocation Donut Chart (`plot_allocation_donut`)
- 55% hole diameter with clean slice borders matching theme card background.
- Slice labels display ticker and percentage; unified tooltip provides full asset name, asset class category, weight percentage, and exact dollar allocation on \$10M reference capital.

#### B. Wealth Trajectory & Underwater Drawdown Plots (`plot_cumulative_wealth`, `plot_drawdowns`)
- Cumulative wealth trajectory tracks growth from \$10,000,000 initial capital against the official **Policy Benchmark** (35% SPY, 10% EFA, 5% EEM, 40% AGG, 10% GLD).
- Underwater drawdown plot shades the deficit area in transparent crimson (`rgba(220, 38, 38, 0.15)`), highlighting depth and duration of recovery.

#### C. The "Honest Point" Efficient Frontier (`plot_efficient_frontier`)
- **The Problem**: In portfolio optimization under endowment mandate constraints (45% single-asset cap, 15% GLD cap, 35% minimum defensive buffer), all return targets between 5.0% and 7.5% collapse to the identical minimum-variance solution (7.66% return, 8.57% volatility).
- **The Design Choice**: Rather than drawing a synthetic curved line to "look like a textbook frontier", the chart renders a **single diamond marker** labeled *"Optimal Constrained Minimum-Variance Portfolio"* paired with an explicit explanatory callout.
- **Why**: Fabricating an efficient frontier curve when constraints make other points mathematically infeasible is unethical in financial presentation.

#### D. Positive Loss Convention in Model Validation (`plot_validation_breaches`)
- Standard academic risk management plots losses as positive numbers.
- Y-axis: Realized Daily Loss / (Gain) in USD.
- VaR Line: Positive threshold line in teal.
- Breaches: Rendered as **hollow red diamonds** (`diamond-open`) above the threshold.
- Gains: Plotted naturally as negative losses below the \$0 horizontal baseline (not artificially truncated).
- **Why**: Allows trustees to see both the magnitude of breaches and the distribution of winning trading days on the same unclipped axis.

#### E. Diverging Horizontal Bar Chart for Risk Contributions (`plot_risk_contributions_diverging`)
- **The Problem**: Traditional risk reports frequently force Euler risk contributions into a pie chart. However, hedging assets (like AGG or GLD during equity shocks) have **negative marginal risk contributions**. A pie chart cannot display negative slices and forces misleading percentages.
- **The Design Choice**: A diverging horizontal bar chart:
  - Positive risk contributors extend to the right in **Red**.
  - Negative risk contributors (hedges) extend to the left in **Green**.
- **Why**: Respects the mathematical identity of Euler risk decomposition while highlighting which assets actively mitigate portfolio tail risk.

---

## 6. Page-by-Page Detailed Walkthrough & Rationale

### 6.1 View 01: Executive Overview (`app/views/overview.py`)
- **Hero & Decision Routes**: Clean brand introduction with quick-action navigation buttons.
- **Four-Metric Executive Banner**:
  1. *Fund Reference Notional*: \$10,000,000
  2. *Optimized Expected Return*: 7.66% (with delta badge: `+1.16% vs Target (6.5%)`)
  3. *Model Annual Volatility*: 8.57%
  4. *Latest Observation Date*: 2026-10-20
- **Model Estimates vs. Historical Reality Expander**:
  - Collapsible side-by-side comparison tables explicitly contrasting:
    - Target Return (6.50%) vs Model Expected (7.66%) vs Realized Compound Return (7.58%).
    - Model Volatility (8.57%) vs Realized Historical Volatility (9.19%).
    - Realized Max Drawdown (-24.84%) and Realized Sharpe Ratio (0.37).
- **Proposed Holdings Cards**: Compact horizontal cards for each active asset (AGG 45%, SPY 22.8%, GLD 15%, EFA 9.6%, XLK 6.3%, XLE 1.3%) with color-coded top borders.
- **Endowment Takeaway Synthesis**: Dynamically generated bulleted paragraphs summarizing capital preservation, defensive buffer (60% AGG+GLD), equity constraint (40%), and model validation status.
- **Mandatory Governance Notice**: Bold notice stating *"Team investment recommendation pending formal presentation and human review. The quantitative figures represent analytical outputs, not an automated buy/sell recommendation."*

### 6.2 View 02: Portfolio & Performance (`app/views/portfolio.py`)
Organized into 4 specialized tabs:
1. **Asset Allocation & Mandate**:
   - Donut chart alongside holdings table (ticker, description, category, weight, dollar amount).
   - **Mandate Compliance Verification Table**: Evaluates 7 formal rules (Total Equity $\le 60\%$, Defensive Minimum $\ge 35\%$, Single Position $\le 45\%$, Gold 5–15%, EEM $\le 10\%$, XLE $\le 10\%$, XLK $\le 20\%$) with `COMPLIANT` status flags and explicit floating-point tolerance ($1\times 10^{-5}$).
2. **Historical Performance & KPIs**:
   - Cumulative wealth trajectory vs Policy Benchmark.
   - Underwater drawdown comparison chart.
   - Core KPI Cards: Compound Annual Return (7.58%), Annual Volatility (9.19%), Max Drawdown (-24.84%), Sharpe Ratio (0.37).
   - Detailed Conventions Expander: Sortino Ratio (0.50), Calmar Ratio (0.31), Beta (0.47), Jensen's Alpha (0.83%), Information Ratio (-0.01), Treynor Ratio (7.21%).
   - Risk-free rate convention disclosure (fixed 4.22% quote for retrospective replay vs time-varying daily FRED series for OOS validation).
3. **Efficient Frontier & Optimization**:
   - Diagnostic plot displaying the constrained minimum-variance solution.
   - Detailed explanation of why return targets from 5.0% to 7.5% produce identical weights.
   - Configured target return grid results table.
4. **Supporting Estimates (Returns & Covariance)**:
   - Complete transparency into input parameters: Blended expected return table (50% historical simple, 50% CAPM prior), annualized log-return covariance matrix (shrunk 25% to diagonal), and model-implied correlation matrix.

### 6.3 View 03: Risk Explorer (`app/views/risk.py`)
- **VaR vs. Expected Shortfall Concept Banner**: Clear definition distinguishing loss threshold (VaR) from tail conditional expectation (ES).
- **Latest 1-Day Loss Metrics Banner**: Displays dollar and percentage figures at the selected confidence level ($96,938 / 0.97% VaR; $133,231 / 1.33% ES at 95% Historical).
- **Three Core Tabs**:
  1. *Risk Forecasts Over Time*: Daily time series tracking VaR and ES evolution over 20+ years.
  2. *Cross-Method Bar Comparison*: Side-by-side bar chart comparing Historical, EWMA Normal, Bootstrap Monte Carlo, and EVT-GPD at the common latest date.
  3. *EVT Threshold Diagnostics*: Threshold selection diagnostics, exceedance counts, shape parameter ($\xi$), scale parameter ($\sigma$), and GPD fit stability heuristics.
- **Methodology Expander**: In-depth theoretical descriptions detailing mathematical formulation, strengths, weaknesses, and assumptions for each forecasting engine.

### 6.4 View 04: Model Reliability & Validation (`app/views/validation.py`)
- **Strictly Locked to Frozen OOS Allocation**: Prevents misleading in-sample validation.
- **Interactive Evaluation Window Filter**: Date form allowing trustees to re-evaluate statistical tests dynamically across custom crisis windows (e.g., 2020 COVID crash or 2022 rate hike cycle) using live backend likelihood-ratio formulas.
- **Realized Losses vs. VaR Breaches Chart**: Positive loss convention with red breach markers.
- **Statistical Test Suite Cards**:
  - *Basel Regulatory Traffic Light*: 99% / 250-Day window.
  - *Kupiec POF Test (Coverage)*: Likelihood-ratio test on exception frequency ($p$-value and reject/accept status).
  - *Christoffersen Independence Test*: First-order Markov chain likelihood-ratio test evaluating breach clustering ($p$-value and reject/accept status).
  - *Joint Conditional Coverage Test*: Combined test ($\chi^2_2$) testing simultaneous coverage and independence.
- **Baseline Calibration Master Table**: Full out-of-sample summary table across all methods and confidence levels (90%, 95%, 99%).
- **Crucial Governance Insight Expander**:
  - Explicitly addresses the paradox: *"Why does the Basel Traffic Light show 'GREEN' while formal tests REJECT at 99%?"*
  - Explains that the trailing 250 days were calm (1–4 exceptions), putting models in the Green Zone, but multi-year history contains severe breach clusters (March 2020, 2022) causing statistical rejection.

### 6.5 View 05: Crisis & Contributions (`app/views/stress.py`)
- **Tab 1: Historical Crisis Replays**:
  - Replays today's proposed weights through three major dislocations:
    1. *Global Financial Crisis (2008)*: -21.03% (-\$2,102,601)
    2. *COVID-19 Market Crash (2020)*: -15.28% (-\$1,528,265)
    3. *2022 Inflation & Rate Shock*: -12.18% (-\$1,218,238)
  - Interactive scenario trajectory visualizer (portfolio wealth and underwater drawdown).
  - Retrospective replay warning: Clarifies that weights were not chosen prior to 2008.
- **Tab 2: Hypothetical Stagflation Shock**:
  - Models simultaneous team-assumed shocks (SPY -25%, EFA -28%, EEM -35%, AGG -12%, GLD +8%, XLE -15%, XLK -35%), producing -\$1,499,218 net impact (-14.99%).
  - Demonstrates gold's +\$120,000 cushioning contribution.
  - Position-level P&L horizontal bar chart (green for gold, red for losses).
  - **Mathematical Reconciliation Check**: Confirms that the sum of position dollar contributions matches scenario net loss to within \$1.00.
- **Tab 3: Position Risk Contributions**:
  - Euler risk decomposition table: Marginal VaR/ES, Component VaR/ES, and Risk Share %.
  - Diverging horizontal bar chart preserving negative signs.
  - Informational expander explaining why capital weight does not equal tail risk share (e.g., AGG is 45% of capital but has minimal or negative tail risk contribution).

### 6.6 View 06: Evidence & Assumptions (`app/views/assumptions.py`)
- **Three Explicit Date Cutoffs**:
  1. *Latest Raw Price Date*: 2026-10-22
  2. *Latest Complete Return Date*: 2026-10-20
  3. *Latest Evaluated Risk Date*: 2026-10-20
- **Automated Discrepancy Detector**: Automatically detects missing prices on 2026-10-21 (EFA and AGG), explaining that returns terminate cleanly on 2026-10-20 without forward-fill imputation.
- **Four Detailed Tabs**:
  1. *Data Quality Report*: Missing price days, 8-MAD robust outlier flags, and return bounds per asset.
  2. *Stationarity Tests (ADF)*: Augmented Dickey-Fuller test table confirming all 7 ETFs reject unit-root null hypothesis at 1% ($p < 0.01$).
  3. *Core Modeling Assumptions*: Detailed disclosure of portfolio optimization, covariance shrinkage, timing split, and execution assumptions.
  4. *Limitations & AI Disclosure*:
     - Explicit disclosure of omitted frictions (trading commissions, bid-ask spreads, custodial fees).
     - Static notional assumption.
     - Thin-tail limits of normal models and sample limits of empirical models.
     - **AI Disclosure**: Transparent acknowledgment that quantitative algorithms and UI were implemented with AI pair programming (Codex / Antigravity IDE), with team fiduciary sign-off required prior to submission.
     - **Experimental Candidate Models Disclosure**: Discloses research into CUSUM alert models (`experiments/`) and explains why they were not substituted into baseline pipeline outputs.

---

## 7. Data Architecture & Integrity Verification

### 7.1 Separation of Data Layers
- **`app/data_access.py`**: Serves as the single source of truth for reading data. View scripts never open raw files directly.
- **Strict Key Validation**: Every loader function validates expected schema columns, JSON keys, and numeric types. Missing keys raise immediate, descriptive exceptions rather than silently propagating `NaN`s.
- **Fine-Grained Streamlit Caching**: Read-only operations use `@st.cache_data(show_spinner=False)`. Disk re-reads occur only on startup or when the user explicitly triggers the "Reload Results Cache" button.

### 7.2 Testing & Verification Suite
The entire Streamlit application and its design layer are continuously verified by **91 automated tests**:
1. **`tests/test_app_views_apptest.py`**:
   - Uses Streamlit's native `AppTest` framework to simulate headless user navigation across all 6 views.
   - Tests basis switching, confidence slider adjustments, risk methodology changes, and reload button execution.
2. **`tests/test_tiramisu_theme.py`**:
   - Exercises all pages in both `light` and `dark` mode.
   - Verifies that palette tokens do not leak between sessions.
   - Performs automated WCAG AA mathematical luminance contrast calculations on all text-background combinations.
3. **Regression & Analytical Tests**:
   - Verifies mathematical identity reconciliation (Euler risk decomposition sums to total risk; scenario contributions sum to scenario P&L).

---

## 8. Summary Checklist of Design Decisions

| Category | Design Decision | Rationale |
|---|---|---|
| **Aesthetics** | Dual Espresso/Ivory & Warm Charcoal Palette | Evokes academic endowment heritage; reduces visual fatigue. |
| **Aesthetics** | Complete elimination of emojis | Preserves institutional gravity; avoids "cheap AI" appearance. |
| **Aesthetics** | WCAG AA contrast compliance ($\ge 4.5:1$) | Guarantees accessibility for trustees and auditors across all screens. |
| **Aesthetics** | Session-local theme tokens | Prevents shared-state leakage and thread collisions in Streamlit. |
| **Navigation** | Two-digit numbered views (`01`–`06`) | Follows executive briefing standards; clear progressive disclosure. |
| **Navigation** | Overview decision route buttons | Provides direct jumping points to critical trustee concerns. |
| **Governance** | Persistent color-coded basis badges | Prevents confusion between retrospective replays and forward OOS tests. |
| **Governance** | Locked basis on Validation & Stress views | Enforces zero look-ahead bias by locking out invalid combinations. |
| **Governance** | Explicit triple date timestamps | Discloses exact data cutoffs; prevents deceptive freshness claims. |
| **Visualizations**| Deterministic asset color registry | SPY is always blue, GLD is always gold; zero cognitive re-learning. |
| **Visualizations**| Single-point efficient frontier display | Intellectually honest; refuses to fabricate a curve under binding constraints. |
| **Visualizations**| Positive loss convention for VaR backtests | Follows Basel standards; shows gains below \$0 line without clipping. |
| **Visualizations**| Diverging horizontal bar chart for Euler risk | Preserves negative signs for hedging assets; eliminates misleading pie charts. |
| **Validation** | Side-by-side Basel vs Formal Test disclosure | Explains why calm 250-day windows show green while multi-year tests reject. |
| **Validation** | Dynamic date filter with backend recomputation | Allows trustees to stress test custom sub-horizons on the fly. |
| **Compliance** | Full mandate compliance verification table | Shows explicit $\le$ and $\ge$ bounds, actual weights, and COMPLIANT status. |
| **Ethics** | Transparent AI pair programming disclosure | Fulfills competition rules and academic integrity requirements. |
| **Data** | 100% offline pre-computed data contracts | Eliminates API failures, rate limits, and latency during board presentations. |

