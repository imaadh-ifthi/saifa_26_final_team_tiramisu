# TiraMisU presentation patch

Identity: TiraMisU — University of Peradeniya.
Tagline: Thoughtful allocation. Transparent risk.

This patch targets the uploaded rash (2) archive. It preserves financial models and saved outputs.

## Included
- Espresso/ivory light theme and warm charcoal dark theme using native Streamlit settings.
- Session-local theme tokens for custom cards and Plotly figures; no shared palette mutation.
- Non-blocking, once-per-session brand animation respecting reduced-motion preference.
- Compact navigation and overview decision shortcuts.
- Collapsed model-versus-history details; redundant overview navigation cards removed.
- Native readable Markdown notices and consistent card surfaces.
- Page routing tests in both theme configurations, contrast checks, and absolute AppTest paths.

## Runtime
Requires Streamlit >=1.66.1,<2. Run from the repository root so .streamlit/config.toml is read.
Choose Appearance using the top-right menu: Settings > Theme > Light / Dark / system.
The native setting controls widgets and tables as well as the custom dashboard.

## Verification
91 tests passed with 7 existing statsmodels future warnings.
Both theme configurations and all six pages exercised via AppTest.
Text colours checked for at least 4.5:1 contrast on card and panel backgrounds.
A headless server launched successfully. Browser installation failed in the review environment,
so visual screenshot validation was not performed. Check desktop/mobile layouts, Plotly hover
labels, tables, and native theme switching locally before presentation.
The update is a distinctive team treatment, not a claim of exclusive worldwide design originality.
No backend calculations, results, or investment conclusions changed.
