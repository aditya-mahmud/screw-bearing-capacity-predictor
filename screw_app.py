
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re

# ================== CONFIG ==================
st.set_page_config(page_title="SDS Bearing Capacity Predictor", layout="wide")

# ======== LOAD ARTIFACTS ========
model    = joblib.load('final_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# ======== EXPECTED FEATURE NAMES (ORDER) ========
if hasattr(scaler_X, "feature_names_in_"):
    EXPECTED = list(scaler_X.feature_names_in_)
else:
    # Fallback to the dataset order you shared
    EXPECTED = ['e1/d','e2/d','p1/d','p2/d','t1/t2','be/b','Nr','fu1/fy1','fu2/fy2','Nb','N']

# ================== STYLES (AESTHETICS) ==================
st.markdown("""
<style>
:root{ --primary:#0ea5a4; --accent:#2563eb; --bg1:#f6f9ff; --card:#ffffff; --ink:#0f172a; --muted:#64748b; }
html, body, [class*="css"] { font-size: 13px !important; }
h1{ font-size:22px !important; color:var(--ink); margin-bottom:10px !important; }
h1:after{ content:""; display:block; width:210px; height:3px; background:linear-gradient(90deg,var(--accent),var(--primary)); border-radius:6px; margin-top:6px; }
.main{ background: linear-gradient(180deg, var(--bg1) 0%, #ffffff 100%); }
.section{ background:var(--card); border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:0 6px 24px rgba(15,23,42,.06); margin-bottom:12px; }
.stNumberInput>div{ background:#fff !important; border:1px solid #d1d5db !important; border-radius:10px !important; box-shadow:0 1px 2px rgba(0,0,0,.03) inset; }
.stNumberInput input{ padding:8px 10px !important; font-size:13px !important; }
.stNumberInput>div:focus-within{ border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(37,99,235,.18) !important; }
div[data-baseweb="input"] > div { width: 190px !important; }
.label{ font-weight:700; color:var(--ink); margin-bottom:2px; font-size:13px; }
.stButton>button{ background:linear-gradient(90deg,var(--primary),#0891b2); color:#fff; border:0; border-radius:10px; padding:10px 16px; font-weight:800; font-size:13px; box-shadow:0 6px 18px rgba(8,145,178,.35); }
.result{ background:linear-gradient(90deg,#ecfeff,#f0fdf4); border:1px solid #bbf7d0; color:#064e3b; border-radius:12px; padding:12px 14px; font-size:14px; font-weight:800; margin-top:8px; }
.caption{ font-size:11px; color:#64748b; text-align:center; margin-top:6px; }
.sds-img img{ width:320px !important; max-width:100% !important; height:auto !important; }
@media (max-width:1200px){ .sds-img img{ width:420px !important; } }
@media (max-width:640px){ .sds-img img{ width:100% !important; } }
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def safe_div(numer, denom, eps=1e-9):
    d = float(denom) if denom is not None else 0.0
    return float(numer) / (d if abs(d) > eps else eps)

def labeled_number_input(html_label, key, *, min_value=0.0, step=0.1, value=None, fmt=None, int_mode=False):
    st.markdown(f'<div class="label">{html_label}</div>', unsafe_allow_html=True)
    if int_mode:
        return st.number_input(" ", key=key, min_value=int(min_value), step=int(step),
                               value=(int(value) if value is not None else None),
                               format="%d", label_visibility="collapsed")
    return st.number_input(" ", key=key, min_value=min_value, step=step,
                           value=value, format=(fmt or None), label_visibility="collapsed")

def canon(name: str) -> str:
    """canonical key: remove non-alphanumerics, lowercase (e1/d -> e1d, fu1/fy1 -> fu1fy1)"""
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())

def engineer_features(t1,t2,d,e1,e2,p1,p2,b,be,N,Nb,Nr,fu1,fu2,fy1,fy2):
    # base engineered values with human-readable keys
    base = {
        'e1/d':    safe_div(e1, d),
        'e2/d':    safe_div(e2, d),
        'p1/d':    safe_div(p1, d),
        'p2/d':    safe_div(p2, d),
        't1/t2':   safe_div(t1, t2),
        'be/b':    safe_div(be, b),
        'Nr':      float(Nr),
        'fu1/fy1': safe_div(fu1, fy1),
        'fu2/fy2': safe_div(fu2, fy2),
        'Nb':      float(Nb),
        'N':       float(N),
    }
    # add common alias spellings to be safe (won't be used if not needed)
    aliases = {
        'e1_over_d': base['e1/d'],
        'e2_over_d': base['e2/d'],
        'p1_over_d': base['p1/d'],
        'p2_over_d': base['p2/d'],
        't1_over_t2': base['t1/t2'],
        'be_over_b': base['be/b'], 'b_e/b': base['be/b'],
        'fu1_over_fy1': base['fu1/fy1'],
        'fu2_over_fy2': base['fu2/fy2'],
        'N_r': base['Nr'], 'nr': base['Nr'],
        'N_b': base['Nb'], 'nb': base['Nb'],
    }
    base.update(aliases)

    # canonical map
    canon_map = {canon(k): v for k, v in base.items()}
    return base, canon_map

def build_vector_in_expected_order(expected_names, canon_map, raw_map):
    """Return (values_list, missing_list, debug_rows)"""
    values, missing, dbg = [], [], []
    for name in expected_names:
        c = canon(name)               # canonical key
        val = canon_map.get(c, None)  # value by canonical
        if val is None:
            # last chance: exact lookup if scaler stored exact match like 'N' or 'Nr'
            val = raw_map.get(name)
        if val is None:
            missing.append(name)
            dbg.append((name, c, None))
            values.append(np.nan)
        else:
            dbg.append((name, c, val))
            values.append(val)
    return values, missing, dbg

# ================== UI ==================
st.title("🔩 Self Drilling Screw Connection — Normalized Bearing Capacity (Pn=Pu/fudt) Predictor")
st.write("Enter parameters below maintaining consistent units. ")

col_inputs, col_image = st.columns([3, 1])

with col_inputs:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        t1 = labeled_number_input("t<sub>1</sub> (Thickness of sheet in contact with screw head)", "t1_input")
        t2 = labeled_number_input("t<sub>2</sub> (Thickness of sheet not in contact with screw head)", "t2_input")
        d  = labeled_number_input("d (Screw diameter)", "d_input")
        e1 = labeled_number_input("e<sub>1</sub> (End distance)", "e1_input")
        e2 = labeled_number_input("e<sub>2</sub> (Edge distance)", "e2_input")
        p1 = labeled_number_input("p<sub>1</sub> (Pitch parallel to loading)", "p1_input")
        p2 = labeled_number_input("p<sub>2</sub> (Pitch perpendicular to loading)", "p2_input")
        b  = labeled_number_input("b (Sheet width)", "b_input")

    with c2:
        be  = labeled_number_input("b<sub>e</sub> (Effective width)", "be_input")
        N   = labeled_number_input("N (Total number of screws)", "N_input", int_mode=True, step=1)
        Nb  = labeled_number_input("N<sub>b</sub> (Number of screws on the row with highest screws)", "Nb_input", int_mode=True, step=1)
        Nr  = labeled_number_input("N<sub>r</sub> (Number of rows)", "Nr_input", int_mode=True, step=1)
        fu1 = labeled_number_input("f<sub>u1</sub> (Tensile strength of sheet in contact with screw head)", "fu1_input", step=1.0)
        fu2 = labeled_number_input("f<sub>u2</sub> (Tensile strength of sheet not in contact with screw head)", "fu2_input", step=1.0)
        fy1 = labeled_number_input("f<sub>y1</sub> (Yield strength of sheet in contact with screw head)", "fy1_input", step=1.0)
        fy2 = labeled_number_input("f<sub>y2</sub> (Yield strength of sheet not in contact with screw head)", "fy2_input", step=1.0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Determine Normalized Connection Capacity (Pn)"):
        try:
            # 1) engineer features & canonical map
            raw_map, canon_map = engineer_features(t1,t2,d,e1,e2,p1,p2,b,be,N,Nb,Nr,fu1,fu2,fy1,fy2)

            # 2) reorder exactly as scaler expects
            values, missing, dbg_rows = build_vector_in_expected_order(EXPECTED, canon_map, raw_map)

            # If anything couldn't be mapped, surface it clearly
            if missing:
                st.error(f"Couldn't map these expected training features from inputs: {missing}\n"
                         f"Scaler expects in this order: {EXPECTED}")
            else:
                X_vec = np.array(values, dtype=np.float64).reshape(1, -1)

                # Debug: show how each expected name was matched & its value
                with st.expander("Debug (expected name → canonical key → value)"):
                    dbg_df = pd.DataFrame(dbg_rows, columns=["expected_name_from_scaler",
                                                             "canonical_key_used", "value"])
                    st.dataframe(dbg_df.style.format(precision=6), use_container_width=True)

                # 3) scale & predict (NumPy bypasses pandas name checks)
                X_scaled      = scaler_X.transform(X_vec)
                y_pred_scaled = model.predict(X_scaled)
                y_pred        = scaler_y.inverse_transform(
                    np.asarray(y_pred_scaled).reshape(-1, 1)
                ).ravel()

                st.markdown(
                    f'<div class="result">✅ Normalized Connection Capacity (Pn): {y_pred[0]:.2f}</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

with col_image:
    st.markdown('<div class="section sds-img">', unsafe_allow_html=True)
    st.image("sds_diagram.png", caption=None, use_container_width=False)
    st.markdown('<div class="caption">SDS Connection Diagram</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
