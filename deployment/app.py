import streamlit as st
import pandas as pd
import joblib, json
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🧳", layout="wide")

HF_MODEL_REPO = "naidu1999/tourism-model"

@st.cache_resource
def load_artifacts():
    model    = joblib.load(hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pkl"))
    features = json.load(open(hf_hub_download(repo_id=HF_MODEL_REPO, filename="feature_names.json")))
    metadata = json.load(open(hf_hub_download(repo_id=HF_MODEL_REPO, filename="metadata.json")))
    enc_map  = json.load(open(hf_hub_download(repo_id=HF_MODEL_REPO, filename="encoding_map.json")))

    # Safety: remove any index/target columns that should not be features
    features = [f for f in features if f not in ["Unnamed: 0", "ProdTaken"]]
    return model, features, metadata, enc_map

model, FEATURES, META, ENC = load_artifacts()

st.title("🧳 Visit With Us — Wellness Tourism Package Predictor")
st.markdown("Predict whether a customer will purchase the **Wellness Tourism Package**.")
st.info(f"Model: **{META['model_name']}** | AUC-ROC: **{META['test_auc']:.4f}** | Accuracy: **{META['test_accuracy']:.4f}**")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Details")
    age            = st.slider("Age", 18, 80, 35)
    monthly_income = st.number_input("Monthly Income (INR)", 5000, 100000, 30000, step=1000)
    gender         = st.selectbox("Gender", list(ENC.get("Gender", {"Male":1,"Female":0}).keys()))
    marital_status = st.selectbox("Marital Status", list(ENC.get("MaritalStatus", {"Single":2,"Married":1,"Divorced":0}).keys()))
    occupation     = st.selectbox("Occupation", list(ENC.get("Occupation", {"Salaried":2,"Freelancer":0}).keys()))
    designation    = st.selectbox("Designation", list(ENC.get("Designation", {"Executive":1,"Manager":2}).keys()))

with col2:
    st.subheader("Trip Preferences")
    city_tier       = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.selectbox("Type of Contact", list(ENC.get("TypeofContact", {"Company Invited":0,"Self Inquiry":1}).keys()))
    num_trips       = st.slider("Number of Trips/Year", 0, 20, 3)
    num_persons     = st.slider("Persons Visiting", 1, 10, 2)
    num_children    = st.slider("Children Below 5", 0, 5, 0)
    prop_star       = st.selectbox("Preferred Hotel Stars", [3, 4, 5])

with col3:
    st.subheader("Sales Interaction")
    pitch_score     = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    pitch_duration  = st.slider("Pitch Duration (mins)", 5, 60, 20)
    followups       = st.slider("Number of Follow-ups", 0, 10, 3)
    product_pitched = st.selectbox("Product Pitched", list(ENC.get("ProductPitched", {"Basic":0,"Standard":3}).keys()))
    passport        = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car         = st.selectbox("Owns a Car?", ["No", "Yes"])

st.divider()

if st.button("Predict Purchase Likelihood", type="primary", use_container_width=True):

    # Build a dict with ALL possible column names
    input_dict = {
        "Age":                      age,
        "TypeofContact":            ENC.get("TypeofContact", {}).get(type_of_contact, 0),
        "CityTier":                 city_tier,
        "DurationOfPitch":          pitch_duration,
        "Occupation":               ENC.get("Occupation", {}).get(occupation, 0),
        "Gender":                   ENC.get("Gender", {}).get(gender, 0),
        "NumberOfPersonVisiting":   num_persons,
        "NumberOfFollowups":        followups,
        "ProductPitched":           ENC.get("ProductPitched", {}).get(product_pitched, 0),
        "PreferredPropertyStar":    prop_star,
        "MaritalStatus":            ENC.get("MaritalStatus", {}).get(marital_status, 0),
        "NumberOfTrips":            num_trips,
        "Passport":                 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation":              ENC.get("Designation", {}).get(designation, 0),
        "MonthlyIncome":            monthly_income
    }

    # Build DataFrame using ONLY the features the model was trained on
    # This handles any column mismatch automatically
    try:
        input_df = pd.DataFrame([input_dict])[FEATURES]
    except KeyError as e:
        st.error(f"Feature mismatch error: {e}")
        st.write("Model expects these features:", FEATURES)
        st.write("Available in input:", list(input_dict.keys()))
        st.stop()

    pred      = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    r1, r2 = st.columns(2)
    with r1:
        if pred == 1:
            st.success("✅ LIKELY TO PURCHASE! Strong candidate for the Wellness Package.")
        else:
            st.warning("❌ UNLIKELY TO PURCHASE. May need more targeted engagement.")
    with r2:
        st.metric("Purchase Probability", f"{pred_prob * 100:.1f}%")
        st.progress(float(pred_prob))

    with st.expander("View Input Features Sent to Model"):
        st.dataframe(input_df)

st.divider()
st.caption("Visit With Us | MLOps Project | Wellness Tourism Package Predictor")
