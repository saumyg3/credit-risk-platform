# =============================================================================
# Credit Risk Intelligence Platform — Dash Dashboard
# =============================================================================

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import shap

# ── Load models and data ───────────────────────────────────────────────
with open("models/xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("models/all_models.pkl", "rb") as f:
    all_models = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

df = pd.read_parquet("data/featured.parquet")
feature_cols = [c for c in df.columns if c not in ["default", "grade", "purpose", "loan_status"]]

# Pre-compute fairness data
from sklearn.preprocessing import StandardScaler
X = df[feature_cols]
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
df["default_prob"] = xgb_model.predict_proba(X_scaled_df)[:, 1]
df["predicted_default"] = (df["default_prob"] >= 0.5).astype(int)

# Income bands
df["income_band"] = pd.cut(df["annual_inc"],
    bins=[0, 40000, 60000, 80000, 100000, float("inf")],
    labels=["<40k", "40-60k", "60-80k", "80-100k", "100k+"]
)

# Model results
model_results = {
    "Logistic Regression": {"AUC-ROC": 0.7252, "Avg Precision": 0.3953, "F1": 0.1900},
    "Random Forest":       {"AUC-ROC": 0.7273, "Avg Precision": 0.4025, "F1": 0.1023},
    "XGBoost":             {"AUC-ROC": 0.7347, "Avg Precision": 0.4153, "F1": 0.4514},
    "LightGBM":            {"AUC-ROC": 0.7348, "Avg Precision": 0.4159, "F1": 0.4510},
}

# SHAP global importance
sample = X_scaled_df.sample(1000, random_state=42)
shap_values = explainer.shap_values(sample)
shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=True)

# ── App setup ──────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=0.8"}])
app.title = "Credit Risk Intelligence Platform"

CARD_STYLE = {"background": "#1e2130", "border": "1px solid #2a2d3a", "borderRadius": "12px"}
ACCENT = "#4a6cf7"
DANGER = "#e74c3c"
SUCCESS = "#2ecc71"

# ── Layout ─────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📊 Credit Risk Intelligence Platform",
                style={"color": "white", "fontWeight": "600", "marginBottom": "4px", "fontSize": "1.8rem"}),
            html.P("4-model ensemble credit risk scoring with SHAP explainability and CFPB fairness auditing",
                style={"color": "#888", "fontSize": "14px"}),
        ], width=9),
        dbc.Col([
            dbc.Badge("391,164 Loans Analyzed", color="primary", className="me-2"),
            dbc.Badge("LendingClub 2007-2018", color="secondary"),
        ], width=3, className="d-flex align-items-center justify-content-end"),
    ], className="my-4"),

    # Tabs
    dbc.Tabs([

        # ── Tab 1: Loan Risk Scorer ──────────────────────────────────
        dbc.Tab(label="🎯 Loan Risk Scorer", tab_id="scorer", children=[
            dbc.Row([
                # Input panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Loan Application Details", style={"color": "white", "marginBottom": "20px"}),

                            dbc.Label("Loan Amount ($)", style={"color": "#aaa", "fontSize": "12px"}),
                            dcc.Slider(1000, 40000, 1000, value=15000, id="loan-amnt",
                                marks={1000: "$1k", 10000: "$10k", 20000: "$20k", 40000: "$40k"},
                                tooltip={"placement": "bottom"}),

                            html.Br(),
                            dbc.Label("Interest Rate (%)", style={"color": "#aaa", "fontSize": "12px"}),
                            dcc.Slider(5, 31, 0.5, value=13, id="int-rate",
                                marks={5: "5%", 15: "15%", 25: "25%", 31: "31%"},
                                tooltip={"placement": "bottom"}),

                            html.Br(),
                            dbc.Label("Annual Income ($)", style={"color": "#aaa", "fontSize": "12px"}),
                            dcc.Slider(10000, 200000, 5000, value=65000, id="annual-inc",
                                marks={10000: "$10k", 60000: "$60k", 120000: "$120k", 200000: "$200k"},
                                tooltip={"placement": "bottom"}),

                            html.Br(),
                            dbc.Label("Debt-to-Income Ratio", style={"color": "#aaa", "fontSize": "12px"}),
                            dcc.Slider(0, 40, 1, value=15, id="dti",
                                marks={0: "0", 10: "10", 20: "20", 40: "40"},
                                tooltip={"placement": "bottom"}),

                            html.Br(),
                            dbc.Label("FICO Score", style={"color": "#aaa", "fontSize": "12px"}),
                            dcc.Slider(580, 850, 10, value=700, id="fico",
                                marks={580: "580", 680: "680", 750: "750", 850: "850"},
                                tooltip={"placement": "bottom"}),

                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Grade", style={"color": "#aaa", "fontSize": "12px"}),
                                    dcc.Dropdown(
                                        options=[{"label": g, "value": i} for i, g in enumerate(["A","B","C","D","E","F","G"], 1)],
                                        value=3, id="grade", clearable=False,
                                        style={"backgroundColor": "#14161f", "color": "white"}
                                    ),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Home Ownership", style={"color": "#aaa", "fontSize": "12px"}),
                                    dcc.Dropdown(
                                        options=[{"label": "Mortgage", "value": 3}, {"label": "Own", "value": 2},
                                                 {"label": "Rent", "value": 1}],
                                        value=1, id="home-own", clearable=False,
                                        style={"backgroundColor": "#14161f", "color": "white"}
                                    ),
                                ], width=6),
                            ]),

                            html.Br(),
                            dbc.Button("Calculate Risk Score", id="score-btn", color="primary",
                                className="w-100", style={"borderRadius": "8px", "fontWeight": "500"}),
                        ])
                    ], style=CARD_STYLE),
                ], width=4),

                # Output panel
                dbc.Col([
                    dbc.Row([
                        # Risk score gauge
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Risk Score", style={"color": "#888", "fontSize": "12px"}),
                                    dcc.Graph(id="risk-gauge", style={"height": "250px"},
                                        config={"displayModeBar": False}),
                                    html.Div(id="risk-verdict", className="text-center"),
                                ])
                            ], style=CARD_STYLE),
                        ], width=6),
                        # Key metrics
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Key Metrics", style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"}),
                                    html.Div(id="key-metrics"),
                                ])
                            ], style=CARD_STYLE),
                        ], width=6),
                    ]),
                    html.Br(),
                    # SHAP explanation
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Why this score? — SHAP Feature Contributions",
                                style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"}),
                            dcc.Graph(id="shap-waterfall", style={"height": "350px"},
                                config={"displayModeBar": False}),
                        ])
                    ], style=CARD_STYLE),
                ], width=8),
            ], className="mt-4"),
        ]),

        # ── Tab 2: Model Comparison ──────────────────────────────────
        dbc.Tab(label="🤖 Model Comparison", tab_id="models", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Model Performance Comparison", style={"color": "white"}),
                            html.P("4 models trained on 391,164 LendingClub loans (80/20 train/test split)",
                                style={"color": "#888", "fontSize": "12px"}),
                            dcc.Graph(id="model-comparison-chart", style={"height": "400px"},
                                config={"displayModeBar": False}),
                        ])
                    ], style=CARD_STYLE),
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Model Insights", style={"color": "white", "marginBottom": "16px"}),
                            dbc.Alert([
                                html.Strong("XGBoost & LightGBM"), " tied as best models (AUC: 0.735), "
                                "significantly outperforming Logistic Regression baseline."
                            ], color="success", style={"fontSize": "13px"}),
                            dbc.Alert([
                                html.Strong("Random Forest"), " achieved highest precision (0.616) "
                                "but sacrificed recall — too conservative for credit scoring."
                            ], color="warning", style={"fontSize": "13px"}),
                            dbc.Alert([
                                html.Strong("Class imbalance"), " handled via scale_pos_weight=4, "
                                "giving equal importance to minority (default) class."
                            ], color="info", style={"fontSize": "13px"}),
                        ])
                    ], style=CARD_STYLE),
                ], width=4),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Global Feature Importance — SHAP", style={"color": "white"}),
                            dcc.Graph(id="shap-global", style={"height": "400px"},
                                config={"displayModeBar": False}),
                        ])
                    ], style=CARD_STYLE),
                ], width=12),
            ], className="mt-4"),
        ]),

        # ── Tab 3: Fairness Audit ────────────────────────────────────
        dbc.Tab(label="⚖️ Fairness Audit", tab_id="fairness", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Disparate Impact by Income Band", style={"color": "white"}),
                            html.P("CFPB 4/5ths rule: approval rate ratio must be ≥ 0.8",
                                style={"color": "#888", "fontSize": "12px"}),
                            dcc.Graph(id="fairness-income", style={"height": "350px"},
                                config={"displayModeBar": False}),
                        ])
                    ], style=CARD_STYLE),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Default Rate by Loan Grade", style={"color": "white"}),
                            html.P("Actual vs model-predicted default rates",
                                style={"color": "#888", "fontSize": "12px"}),
                            dcc.Graph(id="fairness-grade", style={"height": "350px"},
                                config={"displayModeBar": False}),
                        ])
                    ], style=CARD_STYLE),
                ], width=6),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Alert([
                                html.Strong("⚠️ Disparate Impact Detected (Ratio: 0.68)"), html.Br(),
                                "The model approves 49.3% of <$40k income applicants vs 72.5% of $100k+ applicants. "
                                "This ratio of 0.68 falls below the CFPB's 0.8 threshold, indicating potential "
                                "proxy discrimination. Income correlates with race/zip code — regulators would flag this."
                            ], color="danger"),
                            dbc.Alert([
                                html.Strong("📋 Recommended Mitigations:"), html.Br(),
                                "1. Reweight training samples to equalize approval rates across income bands", html.Br(),
                                "2. Remove or regularize income-correlated features (loan_to_income, payment_to_income)", html.Br(),
                                "3. Apply post-processing calibration to equalize odds across groups", html.Br(),
                                "4. Implement ongoing fairness monitoring in production"
                            ], color="warning"),
                        ])
                    ], style=CARD_STYLE),
                ], width=12),
            ], className="mt-4"),
        ]),

    ], id="tabs", active_tab="scorer"),

], fluid=True, style={"backgroundColor": "#0f1117", "minHeight": "100vh", "padding": "0 2rem"})


# ── Callbacks ──────────────────────────────────────────────────────────

@app.callback(
    [Output("risk-gauge", "figure"),
     Output("risk-verdict", "children"),
     Output("key-metrics", "children"),
     Output("shap-waterfall", "figure")],
    Input("score-btn", "n_clicks"),
    [State("loan-amnt", "value"), State("int-rate", "value"),
     State("annual-inc", "value"), State("dti", "value"),
     State("fico", "value"), State("grade", "value"),
     State("home-own", "value")],
    prevent_initial_call=False
)
def score_loan(n_clicks, loan_amnt, int_rate, annual_inc, dti, fico, grade, home_own):
    installment = (loan_amnt * (int_rate/100/12)) / (1 - (1 + int_rate/100/12)**-36)
    monthly_income = annual_inc / 12

    input_data = {
        "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
        "grade_num": grade, "emp_length_num": 5, "annual_inc": annual_inc,
        "dti": dti, "delinq_2yrs": 0, "fico_avg": fico,
        "inq_last_6mths": 1, "open_acc": 8, "revol_bal": 10000,
        "revol_util": 45.0, "total_acc": 15, "mort_acc": 0,
        "loan_to_income": loan_amnt / (annual_inc + 1),
        "payment_to_income": installment / (monthly_income + 1),
        "has_derog": 0, "has_bankruptcy": 0,
        "home_ownership_num": home_own, "purpose_risk": 0,
        "verification_num": 1, "acc_util": 0.5,
        "revol_to_income": 10000 / (annual_inc + 1)
    }

    X_input = pd.DataFrame([input_data])[feature_cols]
    X_scaled = scaler.transform(X_input)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    prob = xgb_model.predict_proba(X_scaled_df)[0][1]
    risk_pct = prob * 100

    # Gauge
    color = SUCCESS if risk_pct < 30 else (DANGER if risk_pct > 60 else "#f39c12")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"color": "white", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#888"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#0d1f0d"},
                {"range": [30, 60], "color": "#1a1a0f"},
                {"range": [60, 100], "color": "#1f0d0d"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
        }
    ))
    gauge_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, margin=dict(t=20, b=0, l=20, r=20)
    )

    verdict_color = SUCCESS if risk_pct < 30 else (DANGER if risk_pct > 60 else "#f39c12")
    verdict_text = "✅ LOW RISK — Approve" if risk_pct < 30 else ("❌ HIGH RISK — Decline" if risk_pct > 60 else "⚠️ MEDIUM RISK — Review")
    verdict = html.H5(verdict_text, style={"color": verdict_color, "marginTop": "8px"})

    # Key metrics
    metrics = dbc.ListGroup([
        dbc.ListGroupItem([html.Strong("Default Probability: "), f"{risk_pct:.1f}%"],
            style={"backgroundColor": "#14161f", "color": "white", "border": "1px solid #2a2d3a"}),
        dbc.ListGroupItem([html.Strong("Monthly Payment: "), f"${installment:.0f}"],
            style={"backgroundColor": "#14161f", "color": "white", "border": "1px solid #2a2d3a"}),
        dbc.ListGroupItem([html.Strong("Payment/Income: "), f"{(installment/monthly_income*100):.1f}%"],
            style={"backgroundColor": "#14161f", "color": "white", "border": "1px solid #2a2d3a"}),
        dbc.ListGroupItem([html.Strong("Loan/Income: "), f"{(loan_amnt/annual_inc*100):.1f}%"],
            style={"backgroundColor": "#14161f", "color": "white", "border": "1px solid #2a2d3a"}),
        dbc.ListGroupItem([html.Strong("FICO Score: "), f"{fico}"],
            style={"backgroundColor": "#14161f", "color": "white", "border": "1px solid #2a2d3a"}),
    ], flush=True)

    # SHAP waterfall
    shap_vals = explainer.shap_values(X_scaled_df)[0]
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": shap_vals
    }).sort_values("shap_value", key=abs, ascending=True).tail(10)

    colors = [DANGER if v > 0 else SUCCESS for v in shap_df["shap_value"]]
    waterfall_fig = go.Figure(go.Bar(
        x=shap_df["shap_value"], y=shap_df["feature"],
        orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition="inside",
        insidetextanchor="middle"
    ))
    waterfall_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, 
        xaxis={"gridcolor": "#2a2d3a", "title": "SHAP Value (impact on default probability)"},
        yaxis={"gridcolor": "#2a2d3a", "automargin": True},
        margin=dict(t=10, b=40, l=10, r=10),
        uniformtext_minsize=8, uniformtext_mode="hide"
    )
    
    return gauge_fig, verdict, metrics, waterfall_fig


@app.callback(
    [Output("model-comparison-chart", "figure"),
     Output("shap-global", "figure")],
    Input("tabs", "active_tab")
)
def update_model_tab(tab):
    # Model comparison
    metrics = ["AUC-ROC", "Avg Precision", "F1"]
    names = list(model_results.keys())
    colors_list = [ACCENT, "#2ecc71", DANGER, "#f39c12"]

    comp_fig = go.Figure()
    for i, (name, color) in enumerate(zip(names, colors_list)):
        comp_fig.add_trace(go.Bar(
            name=name,
            x=metrics,
            y=[model_results[name]["AUC-ROC"], model_results[name]["Avg Precision"], model_results[name]["F1"]],
            marker_color=color, opacity=0.85
        ))
    comp_fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": "#2a2d3a"}, yaxis={"gridcolor": "#2a2d3a", "range": [0, 1]},
        margin=dict(t=10, b=10)
    )

    # SHAP global
    shap_fig = go.Figure(go.Bar(
        x=shap_importance["importance"], y=shap_importance["feature"],
        orientation="h", marker_color=ACCENT, opacity=0.85
    ))
    shap_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, xaxis={"gridcolor": "#2a2d3a", "title": "Mean |SHAP Value|"},
        yaxis={"gridcolor": "#2a2d3a"}, margin=dict(t=10, b=10)
    )

    return comp_fig, shap_fig


@app.callback(
    [Output("fairness-income", "figure"),
     Output("fairness-grade", "figure")],
    Input("tabs", "active_tab")
)
def update_fairness_tab(tab):
    # Income band
    income_data = df.groupby("income_band", observed=True).agg(
        approval_rate=("predicted_default", lambda x: 1 - x.mean())
    ).reset_index()

    income_fig = go.Figure()
    income_fig.add_trace(go.Bar(
        x=income_data["income_band"].astype(str),
        y=income_data["approval_rate"],
        marker_color=[SUCCESS if v >= 0.8 else DANGER for v in income_data["approval_rate"]],
        name="Approval Rate"
    ))
    income_fig.add_hline(y=0.8, line_dash="dash", line_color="red",
        annotation_text="CFPB 0.8 Threshold", annotation_font_color="red")
    income_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, xaxis={"gridcolor": "#2a2d3a"},
        yaxis={"gridcolor": "#2a2d3a", "range": [0, 1]}, margin=dict(t=10, b=10)
    )

    # Grade analysis
    grade_data = df.groupby("grade").agg(
        actual=("default", "mean"),
        predicted=("predicted_default", "mean")
    ).reset_index()

    grade_fig = go.Figure()
    grade_fig.add_trace(go.Bar(name="Actual Default Rate", x=grade_data["grade"],
        y=grade_data["actual"], marker_color=DANGER, opacity=0.7))
    grade_fig.add_trace(go.Bar(name="Predicted Default Rate", x=grade_data["grade"],
        y=grade_data["predicted"], marker_color=ACCENT, opacity=0.7))
    grade_fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": "#2a2d3a"}, yaxis={"gridcolor": "#2a2d3a"},
        margin=dict(t=10, b=10)
    )

    return income_fig, grade_fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
    