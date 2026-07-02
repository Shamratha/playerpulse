"""PlayerPulse — churn & retention dashboard.

Run:  .venv/Scripts/streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_clean  # noqa: E402
from simulate import simulate  # noqa: E402
from features import (  # noqa: E402
    build_enriched_features, ENRICHED_NUMERIC, ENRICHED_CATEGORICAL,
)
from modeling import NUMERIC, CATEGORICAL, ALL_FEATURES  # noqa: E402
from economics import Campaign, plan, roi_curve  # noqa: E402
import shap  # noqa: E402

st.set_page_config(page_title="PlayerPulse", layout="wide")

ENRICHED_COLS = ENRICHED_NUMERIC + ENRICHED_CATEGORICAL
MODEL_PATH = ROOT / "models" / "churn_model_v2.joblib"
METRICS_PATH = ROOT / "reports" / "metrics_v2.json"

# --- design tokens ---------------------------------------------------------
INK, MUTED = "#1a1d29", "#6b7280"
ACCENT, ACCENT_SOFT = "#5145cd", "#c7c2f0"
HIGH, MED, LOW = "#e5484d", "#f0a020", "#30a46c"
GRID = "#eef0f3"
PLOT_FONT = dict(family="Inter, sans-serif", size=13, color=INK)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Inter',-apple-system,'Segoe UI',sans-serif; }

#MainMenu, footer {visibility:hidden;}
[data-testid="stToolbar"], [data-testid="stDecoration"] {display:none;}
header[data-testid="stHeader"] {background:transparent; height:0;}
.block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1180px;}

.pp-logo {font-size:1.7rem; font-weight:700; letter-spacing:-.02em; color:#1a1d29; line-height:1;}
.pp-logo span {color:#5145cd;}
.pp-tag {color:#6b7280; font-size:.92rem; margin-top:.35rem;}
.pp-sub {color:#9aa0aa; font-size:.78rem; margin-top:.15rem;}

[data-testid="stMetric"] {
  background:#fff; border:1px solid #e6e8ec; border-radius:12px;
  padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,.04);
}
[data-testid="stMetricLabel"] p {font-size:.72rem; text-transform:uppercase;
  letter-spacing:.04em; color:#6b7280; font-weight:600;}
[data-testid="stMetricValue"] {font-size:1.45rem; font-weight:700; color:#1a1d29;}

[data-baseweb="tab-list"] {gap:1.8rem; border-bottom:1px solid #e6e8ec;}
[data-baseweb="tab"] {padding:.45rem 0; font-weight:500; color:#6b7280;}
[aria-selected="true"][data-baseweb="tab"] {color:#5145cd;}
[data-baseweb="tab-highlight"] {background:#5145cd;}

h2, h3 {color:#1a1d29; font-weight:600; letter-spacing:-.01em;}

.pp-note {background:#f7f8fa; border:1px solid #e6e8ec; border-left:3px solid #5145cd;
  border-radius:8px; padding:.75rem .95rem; color:#4b5163; font-size:.86rem; line-height:1.5;}
.pp-note b {color:#1a1d29;}
.pp-badge {display:inline-block; font-size:1.05rem; font-weight:600; padding:.1rem 0;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# --- loaders ---------------------------------------------------------------
@st.cache_data
def get_data() -> pd.DataFrame:
    return build_enriched_features(simulate(load_clean()))


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    # Fallback for a fresh deploy where the artifact wasn't committed: train a
    # quick calibrated model on the spot (no hyperparameter search).
    from sklearn.calibration import CalibratedClassifierCV
    from modeling import base_pipeline
    from features import make_xy_enriched
    X_tr, y_tr = make_xy_enriched(get_data())
    m = CalibratedClassifierCV(
        base_pipeline(max_depth=4, learning_rate=0.08, max_iter=300),
        method="isotonic", cv=3)
    m.fit(X_tr, y_tr)
    return m


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def get_json(name: str):
    p = ROOT / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def defaults() -> dict:
    d = get_data()
    out = {c: float(d[c].median()) for c in ENRICHED_NUMERIC}
    for c in ENRICHED_CATEGORICAL:
        out[c] = d[c].mode().iloc[0]
    return out


@st.cache_data
def scored_population() -> pd.DataFrame:
    df = get_data().copy()
    df["churn_proba"] = get_model().predict_proba(df[ENRICHED_COLS])[:, 1]
    df["risk_tier"] = pd.cut(df["churn_proba"], bins=[-0.01, 0.4, 0.7, 1.01],
                             labels=["Low", "Medium", "High"])
    return df


@st.cache_data
def shap_background(n=60):
    return get_data()[NUMERIC].sample(n, random_state=0).astype(float).reset_index(drop=True)


def explain_row(row: dict):
    """SHAP contributions for one player. Numerics are explained; categoricals
    (near-zero importance) are held fixed at the player's values."""
    model = get_model()
    cat_vals = {c: row[c] for c in CATEGORICAL}

    def f(xn):
        d = pd.DataFrame(xn, columns=NUMERIC).astype(float)
        for c, v in cat_vals.items():
            d[c] = v
        return model.predict_proba(d[ALL_FEATURES])[:, 1]

    expl = shap.Explainer(f, shap_background(), algorithm="permutation")
    inst = pd.DataFrame([{c: row[c] for c in NUMERIC}]).astype(float)
    sv = expl(inst, max_evals=300, silent=True)
    return pd.Series(sv.values[0], index=NUMERIC), float(sv.base_values[0])


def style(fig, height=360, legend=False):
    fig.update_layout(
        template="plotly_white", font=PLOT_FONT, height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", y=1.06, x=0, title=None,
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e6e8ec", title_font_size=12)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title_font_size=12)
    return fig


df = get_data()
metrics = get_metrics()
test = metrics["test"]
leak = get_json("metrics_leakage.json")

# --- header ----------------------------------------------------------------
_headline = (f'predictive ROC-AUC {leak["safe"]["auc"]:.2f} (leakage-safe) · '
             f'{leak["full"]["auc"]:.2f} with 14-day engagement'
             if leak else f'calibrated HGB · ROC-AUC {test["roc_auc"]:.2f}')
st.markdown(
    '<div class="pp-logo">Player<span>Pulse</span></div>'
    '<div class="pp-tag">Churn &amp; retention intelligence for mobile games</div>'
    f'<div class="pp-sub">Cookie Cats · {len(df):,} players · {_headline}</div>',
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Players", f"{len(df):,}")
c2.metric("Churn rate (D7)", f"{df['churned'].mean():.1%}")
c3.metric("Day-1 retention", f"{df['retention_1'].mean():.1%}")
c4.metric("Day-7 retention", f"{(1 - df['churned'].mean()):.1%}")
c5.metric("Median rounds", f"{int(df['sum_gamerounds'].median())}")
st.write("")

tab_overview, tab_model, tab_predict, tab_atrisk, tab_campaign = st.tabs(
    ["Retention", "Model", "Risk predictor", "At-risk players", "Campaign planner"]
)

# ================================================================== OVERVIEW
with tab_overview:
    st.subheader("Engagement vs. churn")
    bins = [-1, 0, 1, 5, 10, 30, 50, 100, 1e9]
    labels = ["0", "1", "2-5", "6-10", "11-30", "31-50", "51-100", "100+"]
    d = df.copy()
    d["bucket"] = pd.cut(d["sum_gamerounds"], bins=bins, labels=labels)
    agg = d.groupby("bucket", observed=True).agg(
        players=("userid", "size"), churn_rate=("churned", "mean")).reset_index()
    fig = px.bar(agg, x="bucket", y="churn_rate",
                 text=agg["players"].map("{:,}".format))
    fig.update_traces(marker_color=ACCENT, textposition="outside", cliponaxis=False,
                      textfont=dict(color=MUTED, size=11))
    fig.add_hline(y=df["churned"].mean(), line_dash="dot", line_color=MUTED,
                  annotation_text="overall", annotation_font_color=MUTED)
    fig.update_yaxes(tickformat=".0%", title="Churn rate")
    fig.update_xaxes(title="Total game rounds (first 14 days)")
    st.plotly_chart(style(fig), use_container_width=True)
    st.markdown('<div class="pp-note">Players with <b>100+ rounds</b> churn at ~29%, '
                'versus ~99% for those who barely play. Early engagement is the '
                'strongest retention lever.</div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("A/B test — first gate at level 30 vs 40")
    ab = df.groupby("version").agg(
        day1=("retention_1", "mean"),
        day7=("churned", lambda s: 1 - s.mean())).reset_index()
    ab_m = ab.melt(id_vars="version", value_vars=["day1", "day7"],
                   var_name="metric", value_name="rate")
    ab_m["metric"] = ab_m["metric"].map({"day1": "Day 1", "day7": "Day 7"})
    fig2 = px.bar(ab_m, x="metric", y="rate", color="version", barmode="group",
                  color_discrete_map={"gate_30": ACCENT, "gate_40": ACCENT_SOFT})
    fig2.update_yaxes(tickformat=".1%", title="Retention rate")
    fig2.update_xaxes(title=None)
    st.plotly_chart(style(fig2, legend=True), use_container_width=True)

    exp = get_json("metrics_experiment.json")
    if exp:
        a7, a1 = exp["retention_7"], exp["retention_1"]
        e1, e2, e3 = st.columns(3)
        e1.metric("Day-7 effect of gate_40", f"{a7['effect_pp']:+.2f} pp",
                  f"95% CI [{a7['ci95_pp'][0]:+.2f}, {a7['ci95_pp'][1]:+.2f}]",
                  delta_color="off")
        e2.metric("Significance (day 7)", f"p = {a7['p_value']:.4f}",
                  "significant" if a7["significant_0.05"] else "not significant",
                  delta_color="off")
        e3.metric("Day-1 effect", f"{a1['effect_pp']:+.2f} pp",
                  f"p = {a1['p_value']:.3f} (n.s.)", delta_color="off")
        st.markdown(
            f'<div class="pp-note"><b>Causal reading (valid — this is a randomized test).</b> '
            f'Moving the first gate to level 40 <b>significantly lowers day-7 retention</b> by '
            f'{abs(a7["effect_pp"]):.2f} pp (p={a7["p_value"]:.4f}); the day-1 effect is not '
            f'significant. <b>Why no per-player uplift model?</b> A personalized CATE needs '
            f'<i>pre-treatment</i> covariates to segment on — here players are randomized at '
            f'install with none, so only the <i>average</i> effect is identified. The split '
            f'below is descriptive, not causal.</div>', unsafe_allow_html=True)
        het = pd.DataFrame(exp["heterogeneity_day7"]).rename(columns={
            "bucket": "Rounds", "players": "Players", "control_rate": "gate_30",
            "treatment_rate": "gate_40", "effect_pp": "Effect (pp)"})
        st.dataframe(het, hide_index=True, use_container_width=True)
        st.caption("Descriptive heterogeneity — the gate_40 penalty concentrates in "
                   "mid-engagement players (31–100 rounds).")

# ================================================================== MODEL
with tab_model:
    st.subheader("Predictive performance — leakage-safe headline")
    if leak:
        s = leak["safe"]
        m1, m2, m3 = st.columns(3)
        m1.metric("ROC-AUC (leakage-safe)", f"{s['auc']:.3f}",
                  f"95% CI {s['ci95'][0]:.3f}–{s['ci95'][1]:.3f}", delta_color="off")
        m2.metric("CV ROC-AUC", f"{s['cv_auc_mean']:.3f}",
                  f"± {s['cv_auc_std']:.3f}", delta_color="off")
        m3.metric("Brier (calibrated)", f"{s['brier']:.3f}")

        comp = pd.DataFrame([
            {"Model": "Leakage-safe (reported)", "Features": "retention_1 + version",
             "ROC-AUC": leak["safe"]["auc"],
             "95% CI": f"{leak['safe']['ci95'][0]:.3f}–{leak['safe']['ci95'][1]:.3f}"},
            {"Model": "Engagement-inclusive", "Features": "+ 14-day rounds",
             "ROC-AUC": leak["engagement"]["auc"],
             "95% CI": f"{leak['engagement']['ci95'][0]:.3f}–{leak['engagement']['ci95'][1]:.3f}"},
            {"Model": "Full enriched", "Features": "+ 11 simulated",
             "ROC-AUC": leak["full"]["auc"],
             "95% CI": f"{leak['full']['ci95'][0]:.3f}–{leak['full']['ci95'][1]:.3f}"},
        ])
        st.dataframe(comp, hide_index=True, use_container_width=True)
        st.markdown(
            f'<div class="pp-note"><b>Why {leak["safe"]["auc"]:.3f} is the headline.</b> '
            f'The label is 7-day retention, but <code>sum_gamerounds</code> spans 14 days — '
            f'it partly postdates the label, so using it is peeking. This dataset ships only '
            f'the 14-day total (no daily breakdown to truncate), so the honest predictive '
            f'number comes from features that provably predate the label. Dropping the 14-day '
            f'feature moves AUC {leak["full"]["auc"]:.2f} → {leak["safe"]["auc"]:.2f}; that '
            f'gap is part legitimate early play, part day-8–14 leakage. The engagement '
            f'({leak["engagement"]["auc"]:.3f}) and full ({leak["full"]["auc"]:.3f}) CIs '
            f'overlap → the simulated features add no real lift.<br>'
            f'<span style="color:#6b7280">{leak["methodology"]}</span></div>',
            unsafe_allow_html=True)

    st.write("")
    st.subheader("Working-model diagnostics — engagement-inclusive")
    st.caption("The predictor, at-risk and campaign tabs run this full model as a day-14 "
               "monitoring view: scoring happens after the 14-day window has closed, so using "
               "sum_gamerounds here is characterization, not leakage. The leakage-safe 0.72 "
               "above is the number for genuine early prediction.")
    ci = metrics["auc_ci95"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "HGB + isotonic")
    m2.metric("ROC-AUC (test)", f"{test['roc_auc']:.3f}",
              f"95% CI {ci[0]:.3f}–{ci[1]:.3f}", delta_color="off")
    m3.metric("CV ROC-AUC", f"{metrics['cv_auc_mean']:.3f}",
              f"± {metrics['cv_auc_std']:.3f}", delta_color="off")
    m4.metric("Brier (calibrated)", f"{test['brier_calibrated']:.3f}")

    st.write("")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**AUC by segment** — robustness / fairness check")
        rows = [{"Segment": g, "Value": k, "ROC-AUC": v}
                for g, d in metrics["slice_auc"].items() for k, v in d.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     height=280)
        st.caption("Consistent across version / payer / platform. Near-random (~0.60) for "
                   "zero-engagement players — with no engagement signal there is nothing left "
                   "to discriminate on (an inherent limit, not a modelling bug).")
    with cB:
        st.markdown("**Feature importance** — permutation, calibrated model")
        imp = pd.Series(metrics["importance"]).sort_values()
        figc = px.bar(imp, orientation="h")
        figc.update_traces(marker_color=ACCENT)
        figc.update_xaxes(title="ROC-AUC drop when shuffled")
        figc.update_yaxes(title=None)
        st.plotly_chart(style(figc, height=300), use_container_width=True)

    img = ROOT / "reports" / "figures" / "06_v2_diagnostics.png"
    if img.exists():
        st.image(str(img), caption="Calibration curve & permutation importance (held-out test)")

# ================================================================== PREDICT
with tab_predict:
    st.subheader("Individual churn risk")
    model = get_model()
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        a, b = st.columns(2)
        with a:
            rounds = st.slider("Game rounds (14 days)", 0, 500, 20)
            friends = st.slider("Friends invited", 0, 8, 0)
            spend = st.number_input("Total spend (USD)", 0.0, 700.0, 0.0, step=5.0)
        with b:
            crashes = st.slider("App crashes", 0, 12, 1)
            ret1 = st.checkbox("Returned on day 1", value=False)
            version = st.radio("A/B group", ["gate_30", "gate_40"])
        st.caption("Remaining telemetry is held at population medians; sessions and "
                   "level are derived from rounds to stay coherent.")

    row = defaults().copy()
    row.update({
        "log_gamerounds": np.log1p(rounds),
        "retention_1": int(ret1), "is_gate_40": int(version == "gate_40"),
        "friends_invited": friends, "crashes": crashes,
        "log_spend": np.log1p(spend),
        "n_purchases": 0 if spend == 0 else max(1, round(spend / 5)),
        "log_sessions": np.log1p(max(0, round(rounds / 5))),
        "log_level": np.log1p(round(rounds / 1.4)),
    })
    x = pd.DataFrame([row])[ENRICHED_COLS]
    proba = float(model.predict_proba(x)[:, 1][0])
    tier = "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
    tcol = {"High": HIGH, "Medium": MED, "Low": LOW}[tier]

    with right:
        st.markdown(f'<div class="pp-badge" style="color:{tcol}">{tier} churn risk</div>',
                    unsafe_allow_html=True)
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number={"suffix": "%", "font": {"size": 38, "color": INK}},
            gauge={"axis": {"range": [0, 100], "tickcolor": MUTED},
                   "bar": {"color": tcol, "thickness": 0.3},
                   "bgcolor": "#f7f8fa", "borderwidth": 0,
                   "steps": [{"range": [0, 40], "color": "#eaf6ef"},
                             {"range": [40, 70], "color": "#fdf3e2"},
                             {"range": [70, 100], "color": "#fbe9ea"}]}))
        g.update_layout(height=280, margin=dict(t=10, b=10, l=30, r=30),
                        font=PLOT_FONT, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g, use_container_width=True)

    st.write("")
    if st.button("Explain this prediction", type="secondary"):
        with st.spinner("Computing SHAP contributions…"):
            contrib, base = explain_row(row)
        top = contrib.sort_values(key=abs, ascending=False).head(8).sort_values()
        colors = [HIGH if v > 0 else LOW for v in top.values]
        figs = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                marker_color=colors))
        figs.update_xaxes(title="←  lowers risk        raises risk  →",
                          zeroline=True, zerolinecolor="#c9ccd2")
        figs.update_yaxes(title=None)
        st.plotly_chart(style(figs, height=340), use_container_width=True)
        st.markdown(
            f'<div class="pp-note">The average player starts at <b>{base:.0%}</b> churn '
            f'risk. The factors above move <i>this</i> player to <b>{proba:.0%}</b> — '
            f'<span style="color:{HIGH}"><b>red raises</b></span> risk, '
            f'<span style="color:{LOW}"><b>green lowers</b></span> it. '
            f'(SHAP values; platform/region held fixed.)</div>',
            unsafe_allow_html=True)

# ================================================================== AT-RISK
with tab_atrisk:
    st.subheader("At-risk segment")
    scored = scored_population()
    thr = st.slider("Churn probability threshold", 0.0, 1.0, 0.7, 0.05)
    seg = scored[scored["churn_proba"] >= thr]

    a1, a2, a3 = st.columns(3)
    a1.metric("Players in segment", f"{len(seg):,}")
    a2.metric("Share of base", f"{len(seg)/len(scored):.1%}")
    a3.metric("Avg rounds", f"{seg['sum_gamerounds'].mean():.1f}")
    st.write("")

    dist = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
    figd = px.bar(dist, color=dist.index,
                  color_discrete_map={"Low": LOW, "Medium": MED, "High": HIGH})
    figd.update_xaxes(title="Risk tier")
    figd.update_yaxes(title="Players")
    st.plotly_chart(style(figd), use_container_width=True)

    disp = seg.sort_values("churn_proba", ascending=False).head(200).copy()
    disp["risk_pct"] = disp["churn_proba"] * 100
    disp["d1"] = disp["retention_1"].astype(bool)
    st.dataframe(
        disp[["userid", "version", "sum_gamerounds", "d1", "risk_pct"]],
        column_config={
            "userid": st.column_config.NumberColumn("Player ID", format="%d"),
            "version": "A/B group",
            "sum_gamerounds": st.column_config.NumberColumn("Rounds"),
            "d1": st.column_config.CheckboxColumn("D1 return"),
            "risk_pct": st.column_config.ProgressColumn(
                "Churn risk", min_value=0, max_value=100, format="%.0f%%"),
        },
        hide_index=True, use_container_width=True,
    )

# ================================================================== CAMPAIGN
with tab_campaign:
    st.subheader("Retention campaign planner")
    st.caption("Target a player only when  churn probability × effectiveness × LTV > cost.")
    proba = scored_population()["churn_proba"].to_numpy()

    s1, s2, s3 = st.columns(3)
    ltv = s1.slider("Retained player LTV ($)", 5, 200, 25, 5)
    cost = s2.slider("Cost per player ($)", 0.1, 10.0, 1.0, 0.1)
    eff = s3.slider("Win-back effectiveness", 0.02, 0.60, 0.15, 0.01)

    camp = Campaign(ltv=ltv, cost=cost, effectiveness=eff)
    res = plan(proba, camp)
    st.write("")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Break-even risk", f"{res['threshold']:.0%}")
    k2.metric("Players to target", f"{res['targeted']:,}", f"{res['targeted_share']:.0%} of base")
    k3.metric("Expected saved", f"{res['expected_players_saved']:,.0f}")
    k4.metric("Net value", f"${res['net_value']:,.0f}", f"{res['roi']:.1f}× ROI")
    st.write("")

    curve = roi_curve(proba, camp)
    peak = curve.loc[curve["cumulative_net_value"].idxmax()]
    fig = px.area(curve, x="share_targeted", y="cumulative_net_value")
    fig.update_traces(line_color=ACCENT, fillcolor="rgba(81,69,205,0.12)")
    fig.add_vline(x=peak["share_targeted"], line_dash="dash", line_color=INK,
                  annotation_text=f"optimum · top {peak['share_targeted']:.0%}",
                  annotation_font_color=INK)
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_xaxes(tickformat=".0%", title="Share of players targeted (highest risk first)")
    fig.update_yaxes(title="Cumulative net value ($)")
    st.plotly_chart(style(fig), use_container_width=True)

    naive = float((proba * eff * ltv - cost).sum())
    st.markdown(
        f'<div class="pp-note">Optimal targeting nets <b>${res["net_value"]:,.0f}</b> '
        f'versus <b>${naive:,.0f}</b> for targeting everyone — '
        f'<b>${res["net_value"]-naive:,.0f}</b> saved by not spending on players who '
        f'would stay anyway.</div>', unsafe_allow_html=True)
    st.caption("Assumes the campaign only affects would-be churners and does not fatigue "
               "loyal players.")
