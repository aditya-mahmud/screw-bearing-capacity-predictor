import streamlit as st
import numpy as np
import joblib

# Load model and scalers
model = joblib.load('final_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

st.set_page_config(
    page_title="SDS Bearing Capacity Predictor",
    layout="wide"
)

# Inject CSS to control font sizes and input box width
st.markdown("""
    <style>
    /* Set the font size of the entire app to 10px for A4 print size */
    body {
        font-size: 10px !important;
    }

    /* Reduce title size for A4 fit */
    h1 {
        font-size: 14px !important;
    }

    /* Control the width of the input fields */
    div[data-baseweb="input"] > div {
        width: 140px !important;  /* Adjust input width */
    }

    /* Adjust margins for print simulation */
    .css-1v0mbdj {
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }

    /* Image width adjustment for A4 */
    .css-1yefw6c {
        max-width: 80% !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Buttons should be compact */
    .stButton>button {
        padding: 6px 12px;
        font-size: 10px;
    }

    /* Add padding around sections for neatness */
    .css-1v0mbdj {
        padding-left: 8px;
        padding-right: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔩 Self Drilling Screw Connection Normalized Bearing Capacity Predictor")

st.write(
    "Enter the parameters below to predict the **Normalized Bearing Capacity (Pn)**."
)

# Create 2 main columns: inputs and image
col_inputs, col_image = st.columns([3, 1])

with col_inputs:
    # Create two columns for inputs, ensuring they have the same width
    c1, c2 = st.columns(2)

    with c1:
        t1 = st.number_input("t1 (The thickness of the sheet in contact with screw head)", min_value=0.0, step=0.1, key="t1_input")
        t2 = st.number_input("t2 (The thickness of the sheet not in contact with screw head)", min_value=0.0, step=0.1, key="t2_input")
        d = st.number_input("d (Screw diameter)", min_value=0.0, step=0.1, key="d_input")
        e1 = st.number_input("e1 (End distance)", min_value=0.0, step=0.1, key="e1_input")
        e2 = st.number_input("e2 (Edge distance)", min_value=0.0, step=0.1, key="e2_input")
        p1 = st.number_input("p1 (Pitch parallel)", min_value=0.0, step=0.1, key="p1_input")
        p2 = st.number_input("p2 (Pitch perpendicular)", min_value=0.0, step=0.1, key="p2_input")
        b = st.number_input("b (Sheet width)", min_value=0.0, step=0.1, key="b_input")

    with c2:
        be = st.number_input("be (Effective width)", min_value=0.0, step=0.1, key="be_input")
        N = st.number_input("N (Total number of screws)", min_value=0, step=1, key="N_input")
        Nb = st.number_input("Nb (Number of screw on the row with highest screw)", min_value=0, step=1, key="Nb_input")
        Nr = st.number_input("Nr (No. of rows)", min_value=0, step=1, key="Nr_input")
        fu1 = st.number_input("fu1 (Tensile strength of the sheet in contact with screw head)", min_value=0.0, step=1.0, key="fu1_input")
        fu2 = st.number_input("fu2 (Tensile strength of the sheet not in contact with screw head)", min_value=0.0, step=1.0, key="fu2_input")
        fy1 = st.number_input("fy1 (Yield strength of the sheet in contact with screw head)", min_value=0.0, step=1.0, key="fy1_input")  
        fy2 = st.number_input("fy2 (Yield strength of the sheet not in contact with screw head)", min_value=0.0, step=1.0, key="fy2_input")

    if st.button("Predict Normalized Bearing Capacity (Pn)"):
        try:
            input_values = np.array([
                t1, t2, d, e1, e2, p1, p2, b, be, N, Nb, Nr, fu1, fu2, fy1, fy2
            ]).reshape(1, -1)

            X_scaled = scaler_X.transform(input_values)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            st.success(f"✅ Predicted Normalized Bearing Capacity (Pn): **{y_pred[0]:.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

with col_image:
    st.image(
        "sds_diagram.png",
        caption="SDS Connection Diagram",
        width=300
    )
