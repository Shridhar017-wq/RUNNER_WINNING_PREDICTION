import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Runner Winning Prediction",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #0e1117;
    color: #e0e0e0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1c2333;
    border: 1px solid #2e3650;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] p {
    color: #8b9ab5 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #8b9ab5 !important;
    font-weight: 600;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4f8ef7 !important;
    border-bottom: 2px solid #4f8ef7 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c5cbf);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 20px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Sliders ── */
[data-testid="stSlider"] label {
    color: #c0cce0 !important;
    font-size: 13px !important;
    font-weight: 500;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2e3650;
    border-radius: 10px;
}

/* ── Expander ── */
details {
    background: #1c2333 !important;
    border: 1px solid #2e3650 !important;
    border-radius: 10px !important;
}
details summary {
    color: #c0cce0 !important;
    font-weight: 600;
}

/* ── Section card ── */
.section-card {
    background: #1c2333;
    border: 1px solid #2e3650;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

/* ── Rank badge ── */
.rank-badge {
    display: inline-block;
    width: 36px; height: 36px;
    border-radius: 50%;
    text-align: center;
    line-height: 36px;
    font-weight: 700;
    font-size: 16px;
    margin-right: 10px;
}
.rank-1 { background: #FFD700; color: #000; }
.rank-2 { background: #C0C0C0; color: #000; }
.rank-3 { background: #CD7F32; color: #fff; }
.rank-other { background: #2e3650; color: #e0e0e0; }

/* ── Progress bar ── */
.prog-wrap { background: #2e3650; border-radius: 8px; height: 10px; margin: 6px 0 14px; }
.prog-fill  { height: 10px; border-radius: 8px; }

/* ── Divider ── */
hr { border-color: #2e3650 !important; }

/* ── Input number ── */
input[type="number"] {
    background: #1c2333 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2e3650 !important;
    border-radius: 6px !important;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div > div {
    background: #1c2333 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2e3650 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1c2333 !important;
    border: 1px dashed #4f8ef7 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────

def chance_meta(pred, conf):
    if pred == 1:
        if conf > 0.80: return "High Winning Chance",   "#22c55e", "🟢"
        if conf > 0.55: return "Medium Winning Chance", "#eab308", "🟡"
        return               "Low Winning Chance",      "#f97316", "🟠"
    else:
        if conf > 0.80: return "Very Low Chance",       "#ef4444", "🔴"
        if conf > 0.55: return "Low Winning Chance",    "#f97316", "🟠"
        return               "Borderline – Some Chance","#eab308", "🟡"

def dark_plotly(fig):
    fig.update_layout(
        paper_bgcolor="#1c2333",
        plot_bgcolor="#1c2333",
        font_color="#e0e0e0",
        title_font_color="#ffffff",
        legend_font_color="#e0e0e0",
        xaxis=dict(gridcolor="#2e3650", zerolinecolor="#2e3650", color="#c0cce0"),
        yaxis=dict(gridcolor="#2e3650", zerolinecolor="#2e3650", color="#c0cce0"),
    )
    return fig

def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor":  "#1c2333",
        "axes.facecolor":    "#1c2333",
        "axes.edgecolor":    "#2e3650",
        "axes.labelcolor":   "#c0cce0",
        "xtick.color":       "#c0cce0",
        "ytick.color":       "#c0cce0",
        "text.color":        "#e0e0e0",
        "grid.color":        "#2e3650",
    })

@st.cache_data
def load_and_train(file_bytes):
    import io
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.dropna(inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=200)
    model.fit(Xtr, y_train)

    acc = (model.predict(Xte) == y_test).mean()
    return model, scaler, X.columns.tolist(), acc, df, X_train.values

def predict_one(model, scaler, feat_dict, feature_names):
    arr = pd.DataFrame([feat_dict], columns=feature_names)
    scaled = scaler.transform(arr)
    pred  = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    return int(pred), float(proba[pred]), proba, scaled, arr

# ─── Session state ───────────────────────────────────────────────────────────
for k in ("model","scaler","features","accuracy","df","X_train_raw"):
    if k not in st.session_state:
        st.session_state[k] = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏃 Runner AI")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Runner Data (.xlsx)", type=["xlsx"])

    if uploaded:
        raw = uploaded.read()
        with st.spinner("Training XGBoost model…"):
            model, scaler, features, acc, df, X_train_raw = load_and_train(raw)
        st.session_state.model       = model
        st.session_state.scaler      = scaler
        st.session_state.features    = features
        st.session_state.accuracy    = acc
        st.session_state.df          = df
        st.session_state.X_train_raw = X_train_raw
        st.success("Model ready!")

    if st.session_state.model:
        st.markdown("---")
        st.markdown("### Model Stats")
        st.metric("Accuracy",  f"{st.session_state.accuracy*100:.1f}%")
        st.metric("Features",  len(st.session_state.features))
        st.metric("Samples",   len(st.session_state.df))

    st.markdown("---")
    st.caption("Powered by XGBoost · SHAP · LIME")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a2a4a,#2d1b4e);
            border:1px solid #2e3650; border-radius:16px;
            padding:28px 36px; margin-bottom:24px;'>
  <h1 style='color:#ffffff;margin:0;font-size:2rem;'>🏃 Runner Winning Prediction Dashboard</h1>
  <p style='color:#8b9ab5;margin:6px 0 0;font-size:15px;'>
      AI-powered performance analysis · XGBoost · SHAP · LIME
  </p>
</div>
""", unsafe_allow_html=True)

# ─── No data state ────────────────────────────────────────────────────────────
if st.session_state.model is None:
    st.markdown("""
    <div class='section-card' style='text-align:center;padding:60px;'>
      <h2 style='color:#4f8ef7;'>📂 Upload your Excel file to begin</h2>
      <p style='color:#8b9ab5;font-size:16px;'>
          The file must contain runner performance columns and an <code>Outcome</code> column (0 = lost, 1 = won).
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Single Prediction",
    "🏆  Top 5 Rankings",
    "🔬  Explainability (SHAP & LIME)",
    "📊  Data & Model Insights",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### ⚙️ Runner Parameters")

    # ── Preset buttons rendered BEFORE sliders so session state is set first ──
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🌟 Elite Preset"):
            for f in st.session_state.features:
                st.session_state[f"s1_{f}"] = float(st.session_state.df[f].quantile(0.9))
            st.rerun()
    with c2:
        if st.button("📊 Average Preset"):
            for f in st.session_state.features:
                st.session_state[f"s1_{f}"] = float(st.session_state.df[f].mean())
            st.rerun()
    with c3:
        if st.button("🔄 Reset"):
            for f in st.session_state.features:
                st.session_state[f"s1_{f}"] = float(st.session_state.df[f].mean())
            st.rerun()

    st.markdown("---")

    # ── Sliders rendered AFTER presets so they pick up session state values ──
    feat_vals = {}
    n_cols = 3
    rows = [st.session_state.features[i:i+n_cols] for i in range(0, len(st.session_state.features), n_cols)]

    for row in rows:
        cols = st.columns(n_cols)
        for col, feat in zip(cols, row):
            mn = float(st.session_state.df[feat].min())
            mx = float(st.session_state.df[feat].max())
            mu = float(st.session_state.df[feat].mean())
            # Use session state value if set by a preset, else default to mean
            default = float(st.session_state.get(f"s1_{feat}", mu))
            default = max(mn, min(mx, default))  # clamp to valid range
            with col:
                feat_vals[feat] = st.slider(feat, mn, mx, default, key=f"s1_{feat}")

    st.markdown("---")
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        pred, conf, proba, scaled, arr = predict_one(
            st.session_state.model, st.session_state.scaler,
            feat_vals, st.session_state.features
        )
        label, color, icon = chance_meta(pred, conf)

        # ── Result cards ──
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Prediction", "✅ Winner" if pred == 1 else "❌ Not Winner")
        with r2:
            st.metric("Win Probability", f"{proba[1]*100:.1f}%")
        with r3:
            st.metric("Confidence", f"{conf*100:.1f}%")

        # ── Chance banner ──
        st.markdown(f"""
        <div style='background:#1c2333;border:2px solid {color};border-radius:12px;
                    padding:16px 24px;margin:16px 0;display:flex;align-items:center;gap:12px;'>
          <span style='font-size:28px;'>{icon}</span>
          <span style='font-size:20px;font-weight:700;color:{color};'>{label}</span>
          <span style='margin-left:auto;color:#8b9ab5;font-size:14px;'>
              Confidence: {conf*100:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge ──
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1]*100,
            title={"text": "Win Probability %", "font": {"color": "#ffffff", "size": 18}},
            number={"font": {"color": "#ffffff", "size": 36}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b9ab5", "tickfont": {"color": "#8b9ab5"}},
                "bar":  {"color": color},
                "bgcolor": "#2e3650",
                "steps": [
                    {"range": [0,  40], "color": "#2a1a1a"},
                    {"range": [40, 65], "color": "#2a2a1a"},
                    {"range": [65,100], "color": "#1a2a1a"},
                ],
                "bordercolor": "#2e3650",
            }
        ))
        fig_g.update_layout(height=280, paper_bgcolor="#1c2333", font_color="#e0e0e0",
                            margin=dict(l=30,r=30,t=50,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        # ── Probability bar ──
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=["Not Winner", "Winner"],
            y=[proba[0]*100, proba[1]*100],
            marker_color=["#ef4444", "#22c55e"],
            text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
            textposition="outside",
            textfont=dict(color="#ffffff", size=14),
        ))
        fig_p.update_layout(
            title="Class Probability Breakdown",
            yaxis_range=[0, 115],
            height=320,
        )
        st.plotly_chart(dark_plotly(fig_p), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Top 5 Rankings
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 👥 Enter Multiple Runners")
    num = st.number_input("Number of runners", 2, 20, 5, key="num_runners")

    # Each runner gets unique random defaults (seeded per runner index so stable on rerun)
    runner_inputs = []
    for i in range(int(num)):
        rng = np.random.default_rng(seed=i * 7 + 13)
        with st.expander(f"Runner {i+1}", expanded=(i < 2)):
            cols = st.columns(min(len(st.session_state.features), 4))
            rd = {}
            for j, feat in enumerate(st.session_state.features):
                mn  = float(st.session_state.df[feat].min())
                mx  = float(st.session_state.df[feat].max())
                mu  = float(st.session_state.df[feat].mean())
                std = float(st.session_state.df[feat].std())
                # Unique default: mean ± small random offset per runner
                default = float(np.clip(mu + rng.uniform(-0.6, 0.6) * std, mn, mx))
                with cols[j % len(cols)]:
                    rd[feat] = st.slider(
                        feat, mn, mx, default,
                        key=f"r{i}_{feat}",
                        help=f"Mean: {mu:.2f}"
                    )
            runner_inputs.append(rd)

    if st.button("🚀 Rank All Runners", type="primary", use_container_width=True):
        results = []
        for i, rd in enumerate(runner_inputs):
            pred, conf, proba, _, _ = predict_one(
                st.session_state.model, st.session_state.scaler,
                rd, st.session_state.features
            )
            label, color, icon = chance_meta(pred, conf)
            results.append({
                "rank": 0,
                "name": f"Runner {i+1}",
                "pred": pred,
                "conf": conf,
                "win_prob": proba[1]*100,
                "label": label,
                "color": color,
                "icon": icon,
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        top5 = results[:5]

        st.markdown("---")
        st.markdown("### 🏆 Top 5 Leaderboard")

        rank_colors = ["#FFD700","#C0C0C0","#CD7F32","#4f8ef7","#4f8ef7"]
        for r in top5:
            rk = r["rank"]
            rc = rank_colors[rk-1]
            pct = r["win_prob"]
            st.markdown(f"""
            <div class='section-card' style='border-left:4px solid {rc};padding:18px 24px;margin-bottom:12px;'>
              <div style='display:flex;align-items:center;gap:16px;'>
                <div style='width:42px;height:42px;border-radius:50%;background:{rc};
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;font-size:18px;color:#000;flex-shrink:0;'>
                  {rk}
                </div>
                <div style='flex:1;'>
                  <div style='font-size:18px;font-weight:700;color:#ffffff;'>{r["name"]}</div>
                  <div style='color:#8b9ab5;font-size:13px;margin-top:2px;'>
                      {r["icon"]} {r["label"]}
                  </div>
                </div>
                <div style='text-align:right;'>
                  <div style='font-size:26px;font-weight:800;color:{r["color"]};'>{pct:.1f}%</div>
                  <div style='color:#8b9ab5;font-size:12px;'>Win Probability</div>
                </div>
              </div>
              <div style='margin-top:12px;background:#2e3650;border-radius:8px;height:8px;'>
                <div style='width:{pct}%;background:{r["color"]};height:8px;border-radius:8px;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Comparison chart ──
        st.markdown("### 📊 Win Probability Comparison")
        names  = [r["name"]     for r in results]
        probs  = [r["win_prob"] for r in results]
        colors = [r["color"]    for r in results]

        fig_b = go.Figure(go.Bar(
            x=names, y=probs,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color="#ffffff"),
        ))
        fig_b.update_layout(
            title="All Runners – Win Probability",
            yaxis_range=[0, 115],
            height=400,
        )
        st.plotly_chart(dark_plotly(fig_b), use_container_width=True)

        # ── Full table ──
        st.markdown("### 📋 Full Results Table")
        tbl = pd.DataFrame([{
            "Rank": r["rank"],
            "Runner": r["name"],
            "Win Probability": f"{r['win_prob']:.1f}%",
            "Confidence": f"{r['conf']*100:.1f}%",
            "Prediction": "Winner" if r["pred"] == 1 else "Not Winner",
            "Chance Level": r["label"],
        } for r in results])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Explainability: SHAP + LIME
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 AI Explainability – SHAP & LIME")
    st.markdown("""
    <div class='section-card'>
      <p style='color:#c0cce0;margin:0;'>
        Adjust the sliders below and click <strong>Explain</strong> to see why the model
        made its decision using both <strong>SHAP</strong> (global tree-based) and
        <strong>LIME</strong> (local linear approximation).
      </p>
    </div>
    """, unsafe_allow_html=True)

    xfeat = {}
    xcols = st.columns(3)
    for idx, feat in enumerate(st.session_state.features):
        mn = float(st.session_state.df[feat].min())
        mx = float(st.session_state.df[feat].max())
        mu = float(st.session_state.df[feat].mean())
        with xcols[idx % 3]:
            xfeat[feat] = st.slider(feat, mn, mx, mu, key=f"x_{feat}")

    if st.button("🔍 Explain Prediction", type="primary", use_container_width=True):
        pred, conf, proba, scaled, arr = predict_one(
            st.session_state.model, st.session_state.scaler,
            xfeat, st.session_state.features
        )
        label, color, icon = chance_meta(pred, conf)

        st.markdown(f"""
        <div style='background:#1c2333;border:2px solid {color};border-radius:12px;
                    padding:14px 22px;margin:14px 0;display:flex;align-items:center;gap:12px;'>
          <span style='font-size:24px;'>{icon}</span>
          <span style='font-size:18px;font-weight:700;color:{color};'>{label}</span>
          <span style='margin-left:auto;color:#8b9ab5;'>
              Win Prob: {proba[1]*100:.1f}% &nbsp;|&nbsp; Confidence: {conf*100:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP ──────────────────────────────────────────────────────────
        st.markdown("#### 🌳 SHAP – Feature Contribution")
        st.markdown("""
        <p style='color:#8b9ab5;font-size:13px;'>
        SHAP values show how much each feature pushed the prediction toward or away from "Winner".
        <span style='color:#ef4444;'>Red = pushes toward Not Winner</span> &nbsp;
        <span style='color:#22c55e;'>Green = pushes toward Winner</span>
        </p>
        """, unsafe_allow_html=True)

        explainer = shap.TreeExplainer(st.session_state.model)
        shap_vals  = explainer.shap_values(scaled)

        mpl_dark()
        fig_shap, ax = plt.subplots(figsize=(10, 5))
        shap_series = pd.Series(shap_vals[0], index=st.session_state.features).sort_values()
        bar_colors  = ["#ef4444" if v < 0 else "#22c55e" for v in shap_series]
        shap_series.plot(kind="barh", ax=ax, color=bar_colors)
        ax.set_xlabel("SHAP Value (impact on prediction)", color="#c0cce0")
        ax.set_title("SHAP Feature Contributions", color="#ffffff", fontsize=14, pad=12)
        ax.axvline(0, color="#8b9ab5", linewidth=0.8, linestyle="--")
        ax.tick_params(colors="#c0cce0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3650")
        fig_shap.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

        # SHAP waterfall table
        shap_df = pd.DataFrame({
            "Feature":    st.session_state.features,
            "Value":      [xfeat[f] for f in st.session_state.features],
            "SHAP Impact": shap_vals[0],
        }).sort_values("SHAP Impact", key=abs, ascending=False)

        st.dataframe(
            shap_df.style
                .background_gradient(subset=["SHAP Impact"], cmap="RdYlGn")
                .format({"Value": "{:.3f}", "SHAP Impact": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # ── LIME ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🍋 LIME – Local Interpretable Explanation")
        st.markdown("""
        <p style='color:#8b9ab5;font-size:13px;'>
        LIME fits a simple linear model around this specific prediction to explain it locally.
        <span style='color:#22c55e;'>Green bars = support winning</span> &nbsp;
        <span style='color:#ef4444;'>Red bars = oppose winning</span>
        </p>
        """, unsafe_allow_html=True)

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=st.session_state.X_train_raw,
            feature_names=st.session_state.features,
            class_names=["Not Winner", "Winner"],
            mode="classification",
            random_state=42,
        )

        lime_exp = lime_explainer.explain_instance(
            data_row=scaled[0],
            predict_fn=st.session_state.model.predict_proba,
            num_features=len(st.session_state.features),
        )

        lime_list = lime_exp.as_list()
        lime_df   = pd.DataFrame(lime_list, columns=["Condition", "Weight"])
        lime_df   = lime_df.sort_values("Weight", key=abs, ascending=False)

        mpl_dark()
        fig_lime, ax2 = plt.subplots(figsize=(10, 5))
        lime_colors = ["#22c55e" if w > 0 else "#ef4444" for w in lime_df["Weight"]]
        ax2.barh(lime_df["Condition"], lime_df["Weight"], color=lime_colors)
        ax2.set_xlabel("LIME Weight", color="#c0cce0")
        ax2.set_title("LIME Local Explanation", color="#ffffff", fontsize=14, pad=12)
        ax2.axvline(0, color="#8b9ab5", linewidth=0.8, linestyle="--")
        ax2.tick_params(colors="#c0cce0")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2e3650")
        fig_lime.tight_layout()
        st.pyplot(fig_lime)
        plt.close()

        st.dataframe(
            lime_df.style
                .background_gradient(subset=["Weight"], cmap="RdYlGn")
                .format({"Weight": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Data & Model Insights
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{st.session_state.accuracy*100:.1f}%")
    m2.metric("Features",  len(st.session_state.features))
    m3.metric("Samples",   len(st.session_state.df))
    winners = int(st.session_state.df["Outcome"].sum())
    m4.metric("Winners in Data", f"{winners} / {len(st.session_state.df)}")

    st.markdown("---")

    # ── Feature Importance ──
    st.markdown("#### 🎯 Feature Importance (XGBoost)")
    fi = pd.DataFrame({
        "Feature":    st.session_state.features,
        "Importance": st.session_state.model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"],
        orientation="h",
        marker=dict(
            color=fi["Importance"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#e0e0e0"), title=dict(text="Importance", font=dict(color="#e0e0e0")))
        ),
        text=[f"{v:.3f}" for v in fi["Importance"]],
        textposition="outside",
        textfont=dict(color="#ffffff"),
    ))
    fig_fi.update_layout(height=max(300, len(st.session_state.features)*40), title="Feature Importance Ranking")
    st.plotly_chart(dark_plotly(fig_fi), use_container_width=True)

    # ── Distribution ──
    st.markdown("---")
    st.markdown("#### 📈 Feature Distribution by Outcome")
    sel = st.selectbox("Select feature", st.session_state.features)

    fig_d = px.histogram(
        st.session_state.df, x=sel, color="Outcome",
        marginal="box",
        barmode="overlay",
        color_discrete_map={0: "#ef4444", 1: "#22c55e"},
        labels={"Outcome": "Outcome (0=Lost, 1=Won)"},
        title=f"Distribution of {sel}",
    )
    fig_d.update_layout(legend_font_color="#e0e0e0")
    st.plotly_chart(dark_plotly(fig_d), use_container_width=True)

    # ── Correlation heatmap ──
    st.markdown("---")
    st.markdown("#### 🔥 Correlation Heatmap")
    corr = st.session_state.df.corr()
    fig_c = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
    )
    fig_c.update_layout(
        paper_bgcolor="#1c2333", plot_bgcolor="#1c2333",
        font_color="#e0e0e0", title_font_color="#ffffff",
        coloraxis_colorbar=dict(tickfont=dict(color="#e0e0e0"), title=dict(font=dict(color="#e0e0e0")))
    )
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Raw data ──
    st.markdown("---")
    st.markdown("#### 🗃️ Raw Dataset")
    st.dataframe(st.session_state.df, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;color:#4a5568;font-size:12px;margin-top:40px;padding:16px;
            border-top:1px solid #2e3650;'>
  Runner Winning Prediction Dashboard &nbsp;·&nbsp; XGBoost · SHAP · LIME &nbsp;·&nbsp;
  {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div>
""", unsafe_allow_html=True)
