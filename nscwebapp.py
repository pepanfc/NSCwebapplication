import streamlit as st
import pandas as pd
import numpy as np
import joblib
from Bio import SeqIO
from io import StringIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import requests
import py3Dmol
from stmol import showmol
from collections import Counter
import math

# ===== CSS Custom Style =====
st.set_page_config(page_title="Drug Delivery Peptide Prediction", layout="wide")
st.markdown("""
<style>
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #3498DB 0%, #1A4B7A 100%);
    color: #FFF !important;
    border-bottom: 4px solid #FFD700;
}
.stApp { background-color: #F8FBFF; font-family: 'Sarabun', 'Roboto', sans-serif; }
h1, .title {
    background-color: #3498DB;
    color: #FFF; text-align: center; padding: 18px 0;
    border-radius: 18px; font-size: 2.2rem; font-weight: 800;
    margin-bottom: 20px; box-shadow: 0 6px 30px rgba(52,152,219,0.10);
}
.st-emotion-cache-1kyxreq { color: #1A4B7A !important; }
section[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #3498DB; background: #E3F1FB; border-radius: 16px;
}
.card {
    background: #FFF; border-radius: 18px; box-shadow: 0 2px 16px 0 rgba(80,100,150,.13);
    padding: 24px 32px; margin-bottom: 28px; transition: box-shadow .2s;
}
.card:hover { box-shadow: 0 8px 38px 0 rgba(40,100,170,0.22); }
[data-testid="stMetric"] {
    background: #F8FBFF; border-radius: 14px; padding: 8px 8px 2px 8px;
    box-shadow: 0 1px 4px rgba(52,152,219,0.07); margin-bottom: 8px;
}
button[kind="primary"], .stButton>button {
    background: linear-gradient(90deg, #63A2DF 0%, #3498DB 100%);
    color: #FFF; font-weight: 700; font-size: 1.06rem; padding: 9px 28px;
    border-radius: 12px; border: 2px solid #FFD700; margin-top: 7px; margin-bottom: 5px;
    transition: background .17s, box-shadow .17s; box-shadow: 0 2px 8px 0 rgba(52,152,219,0.14);
}
button[kind="primary"]:hover, .stButton>button:hover { background: #1A4B7A !important; color: #FFD700; }
div[data-baseweb="select"] > div {
    background: #E3F1FB !important; border-radius: 12px; color: #1A4B7A; font-weight: 600; border: 2px solid #63A2DF;
}
::-webkit-scrollbar { width: 9px; }
::-webkit-scrollbar-track { background: #F8FBFF; }
::-webkit-scrollbar-thumb { background-color: #63A2DF; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

def DDact(pct):
    if pct >= 80: return "⭐️ Very High Potential: This peptide has very high potential to be drug delivery. Further experimental validation is recommended."
    elif pct >= 60: return "🟡 High Potential: This peptide demonstrates good potential for drug delivery. Laboratory validation is advised."
    elif pct >= 40: return "🟠 Moderate Potential: This peptide shows moderate potential. Further testing is required."
    elif pct >= 20: return "🔴 Low Potential: This peptide is unlikely to be effective. Proceed cautiously and consider further investigations."
    else: return "⚫ Very Low Potential: This peptide is predicted to have very low suitability."

AA_COLOR_MAP = {
    'A': '#1F77B4',  'C': '#FFD700', 'D': '#2CA02C', 'E': '#D62728', 'F': '#984EA3',
    'G': '#A9A9A9',  'H': '#1F78B4', 'I': '#4DAF4A', 'K': '#E7298A', 'L': '#377EB8',
    'M': '#FF1493',  'N': '#A65628', 'P': '#F781BF', 'Q': '#F0E442', 'R': '#000000',
    'S': '#FFFF00',  'T': '#00FF00', 'V': '#FF4500', 'W': '#A52A2A', 'Y': '#00BFFF'
}
AA_NAME_MAP = {
    'A': 'Alanine',   'C': 'Cysteine',    'D': 'Aspartic acid', 'E': 'Glutamic acid',
    'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine',     'I': 'Isoleucine',
    'K': 'Lysine',    'L': 'Leucine',     'M': 'Methionine',    'N': 'Asparagine',
    'P': 'Proline',   'Q': 'Glutamine',   'R': 'Arginine',      'S': 'Serine',
    'T': 'Threonine', 'V': 'Valine',      'W': 'Tryptophan',    'Y': 'Tyrosine',
}

@st.cache_resource
def load_model():
    return joblib.load('anewxgboost_protein_classifier.pkl')
model = load_model()

def color_by_sequence(view, sequence):
    for i, aa in enumerate(sequence):
        color = AA_COLOR_MAP.get(aa.upper(), '#FFFFFF')
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': color}})

def render_mol_colored(pdb_str, sequence):
    view = py3Dmol.view()
    view.addModel(pdb_str, 'pdb')
    view.setBackgroundColor('#E3F1FB')
    color_by_sequence(view, sequence)
    view.zoomTo()
    view.spin(False)
    st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
    showmol(view, height=440, width=640)
    st.markdown('</div>', unsafe_allow_html=True)

def render_legend():
    aa_order = [
        'A','C','D','E','F',
        'G','H','I','K','L',
        'M','N','P','Q','R',
        'S','T','V','W','Y'
    ]
    col1 = aa_order[:10]
    col2 = aa_order[10:]
    c1, c2 = st.columns(2)
    with c1:
        for aa in col1:
            color = AA_COLOR_MAP[aa]
            name = AA_NAME_MAP[aa]
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; margin-bottom: 7px;'>
                    <div style='background: {color}; width: 26px; height: 26px; border-radius: 6px; border: 1.3px solid #222; margin-right: 10px; display:inline-block;'></div>
                    <span style='font-size: 15px; vertical-align:middle;'>{name} <b>({aa})</b></span>
                </div>
                """, unsafe_allow_html=True
            )
    with c2:
        for aa in col2:
            color = AA_COLOR_MAP[aa]
            name = AA_NAME_MAP[aa]
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; margin-bottom: 7px;'>
                    <div style='background: {color}; width: 26px; height: 26px; border-radius: 6px; border: 1.3px solid #222; margin-right: 10px; display:inline-block;'></div>
                    <span style='font-size: 15px; vertical-align:middle;'>{name} <b>({aa})</b></span>
                </div>
                """, unsafe_allow_html=True
            )

def predict_structure(sequence):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(
        'https://api.esmatlas.com/foldSequence/v1/pdb/',
        headers=headers,
        data=sequence
    )
    if response.status_code == 200:
        return response.text
    else:
        st.error("Failed to get 3D structure from server. Try again later.")
        return None

def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = [f'AAC_{aa}' for aa in AA]
    for record in fastas:
        sequence = str(record.seq).replace('-', '').upper()
        if len(sequence) == 0:
            code = [0.0] * len(AA)
        else:
            count = Counter(sequence)
            code = [count.get(aa, 0) / len(sequence) for aa in AA]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def PAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08",
        "Hydrophilicity  -0.5    3   0.2   3   -1   0.2   3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3   -0.4    -3.4    -2.3    -1.5"
    ]
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {aa: idx for idx, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []
    for line in records[1:]:
        parts = line.rstrip().split()
        if parts:
            AAProperty.append([float(x) for x in parts[1:]])
            AAPropertyNames.append(parts[0])
    AAProperty1 = []
    for prop in AAProperty:
        meanI = sum(prop) / len(prop)
        fenmu = math.sqrt(sum([(x - meanI) ** 2 for x in prop]) / len(prop))
        if fenmu == 0:
            normalized_prop = [0.0 for x in prop]
        else:
            normalized_prop = [(x - meanI) / fenmu for x in prop]
        AAProperty1.append(normalized_prop)
    encodings = []
    header = [f'PAAC_Xc1_{aa}' for aa in AA]
    for j in range(1, lambdaValue + 1):
        header.append(f'PAAC_Xc2_lambda{j}')
    for record in fastas:
        sequence = str(record.seq).replace('-', '').upper()
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            sum_theta = 0.0
            if len(sequence) > n:
                for j in range(len(AAProperty1)):
                    valid_values = [
                        AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]]
                        for k in range(len(sequence) - n)
                        if sequence[k] in AADict and sequence[k + n] in AADict
                    ]
                    if valid_values:
                        sum_theta += sum(valid_values) / (len(sequence) - n)
            theta.append(sum_theta)
        myDict = {aa: sequence.count(aa) for aa in AA}
        total_theta = sum(theta)
        if total_theta == 0:
            total_theta = 1
        code += [myDict[aa] / (1 + w * total_theta) for aa in AA]
        code += [w * value / (1 + w * total_theta) for value in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def APAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08",
        "Hydrophilicity  -0.5    3   0.2   3   -1   0.2   3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3   -0.4    -3.4    -2.3    -1.5"
    ]
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {aa: idx for idx, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []
    for line in records[1:]:
        parts = line.rstrip().split()
        if parts:
            AAProperty.append([float(x) for x in parts[1:]])
            AAPropertyNames.append(parts[0])
    AAProperty1 = []
    for prop in AAProperty:
        meanI = sum(prop) / len(prop)
        fenmu = math.sqrt(sum([(x - meanI) ** 2 for x in prop]) / len(prop))
        if fenmu == 0:
            normalized_prop = [0.0 for x in prop]
        else:
            normalized_prop = [(x - meanI) / fenmu for x in prop]
        AAProperty1.append(normalized_prop)
    encodings = []
    header = [f'APAAC_Pc1_{aa}' for aa in AA]
    for j in range(1, lambdaValue + 1):
        for name in AAPropertyNames:
            header.append(f'APAAC_Pc2.{name}.{j}')
    for record in fastas:
        sequence = str(record.seq).replace('-', '').upper()
        code = []
        theta = []
        for j, prop in enumerate(AAProperty1):
            for n in range(1, lambdaValue + 1):
                sum_theta = 0.0
                if len(sequence) > n:
                    valid_values = []
                    for k in range(len(sequence) - n):
                        aa1 = sequence[k]
                        aa2 = sequence[k + n]
                        if aa1 in AADict and aa2 in AADict:
                            valid_values.append(prop[AADict[aa1]] * prop[AADict[aa2]])
                    if valid_values:
                        sum_theta = sum(valid_values) / (len(sequence) - n)
                theta.append(sum_theta)
        myDict = {aa: sequence.count(aa) for aa in AA}
        total_theta = sum(theta)
        if total_theta == 0:
            total_theta = 1
        code += [myDict[aa] / (1 + w * total_theta) for aa in AA]
        code += [w * value / (1 + w * total_theta) for value in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

import streamlit as st

LICENSE_TEXT = """
<div style="border:2.5px solid #1A4B7A; border-radius:18px; padding:26px 22px; background:#f6fafd; margin:32px 0;">
<h3 style="color:#1A4B7A;">License Agreement</h3>
<p style="color:#222; font-size:1.05rem; line-height:1.7;">
This software is a work developed by SIppatorn Suwanro, Waris Ihaloh, and Poramet Sinprasert from Prince of Songkla university Demonstration School (Secondary) under the provision of Assistant Professor Salang Musikasuwan under “Machine Learning Website for Peptide-Based Drug Delivery Prediction” which has been supported by the National Science and Technology Development Agency (NSTDA), in order to encourage pupils and students to learn and practice their skills in developing software. Therefore, the intellectual property of this software shall belong to the developer and the developer gives NSTDA a permission to distribute this software as an “as is” and non-modified software for a temporary and non-exclusive use without remuneration to anyone for his or her own purpose or academic purpose, which are not commercial purposes. In this connection, NSTDA shall not be responsible to the user for taking care, maintaining, training, or developing the efficiency of this software. Moreover, NSTDA shall not be liable for any error, software efficiency and damages in connection with or arising out of the use of the software.
</p>
</div>
"""

if "license_accepted" not in st.session_state:
    st.session_state.license_accepted = False

if not st.session_state.license_accepted:
    st.markdown(LICENSE_TEXT, unsafe_allow_html=True)
    agree = st.button("I Accept The License Agreement")
    if agree:
        st.session_state.license_accepted = True
        st.experimental_rerun()
    st.stop()  # ยังไม่ไปหน้าเว็บหลักจนกว่าจะกด

def main():
    st.title("🧬 Peptide-Based Drug Delivery Prediction")
    st.markdown("""
    <div style='font-size:1.15rem; font-weight:500; color:#1A4B7A; text-align:center; margin-bottom:18px;'>
        Predict the potential of your peptide sequences as drug delivery using post-train XgBoost model.<br>
        <b>Upload your <code>.fasta</code> file to begin.</b>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your peptide sequence file (.fasta, .fa, .fna, .ffn, .faa, .frn):",
            type=["fasta", "fa", "fna", "ffn", "faa", "frn"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        fasta_string = uploaded_file.read().decode("utf-8")
        fasta_io = StringIO(fasta_string)
        fasta_records = list(SeqIO.parse(fasta_io, 'fasta'))
        if not fasta_records:
            st.error("No sequences found!")
            return
        feat_aac, _ = AAC(fasta_records)
        feat_apaac, _ = APAAC(fasta_records, lambdaValue=1)
        feat_paac, _ = PAAC(fasta_records, lambdaValue=1)
        all_feats = np.hstack((feat_aac, feat_apaac, feat_paac))
        minmax_scaler = MinMaxScaler().fit(all_feats)
        standard_scaler = StandardScaler().fit(minmax_scaler.transform(all_feats))
        X_test = standard_scaler.transform(minmax_scaler.transform(all_feats))
        y_proba = model.predict_proba(X_test)
        csv_data = []
        for rec, proba in zip(fasta_records, y_proba):
            csv_data.append({
                "Peptide Name": rec.id,
                "Sequence": str(rec.seq),
                "Probability (%)": round(proba[1]*100, 2)
            })
        csv_df = pd.DataFrame(csv_data)
        csv_file = csv_df.to_csv(index=False).encode('utf-8')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔎 Prediction Result")
        options = [rec.id for rec in fasta_records]
        selected_id = st.selectbox("Select a Sequence from your file:", options)
        for rec, proba in zip(fasta_records, y_proba):
            if rec.id == selected_id:
                pct = proba[1] * 100
                state_key = f'predict_pressed_{rec.id}'
                if state_key not in st.session_state:
                    st.session_state[state_key] = False
                st.subheader("📌 Peptide Sequence")
                st.code(str(rec.seq), language="markdown")
                col3d, collegend = st.columns([1.7, 1.0])
                with col3d:
                    if st.button("🔬 Show 3D Structure", key=f"btn_{rec.id}"):
                        st.session_state[state_key] = True
                    if st.session_state[state_key]:
                        pdb_str = predict_structure(str(rec.seq))
                        if pdb_str:
                            render_mol_colored(pdb_str, str(rec.seq))
                            st.markdown('<div style="margin-top:12px; margin-bottom:0; background:#F7FCEB; border-radius:12px; padding:14px;">', unsafe_allow_html=True)
                            st.metric(label="Probability of Drug Delivery Activity", value=f"{pct:.2f}%")
                            st.info(DDact(pct))
                            st.markdown('</div>', unsafe_allow_html=True)
                            colcsv, colpdb = st.columns([1, 1])
                            with colpdb:
                                st.download_button(
                                    "Download PDB",
                                    data=pdb_str,
                                    file_name=f"{rec.id}_predicted.pdb",
                                    mime="text/plain"
                                )
                            with colcsv:
                                st.download_button(
                                    label="⬇️ Download All Results (.csv)",
                                    data=csv_file,
                                    file_name="peptide_prediction_results.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.error("3D structure prediction failed.")
                with collegend:
    # ถ้ายังไม่กดปุ่ม 3D โชว์ผล/แปรผลปกติ
                    if not st.session_state[state_key]:
                        st.markdown('<div style="margin-top:8px; background:#F7FCEB; border-radius:12px; padding:14px;">', unsafe_allow_html=True)
                        st.metric(label="Probability of Drug Delivery Activity", value=f"{pct:.2f}%")
                        st.info(DDact(pct))
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
    # === Show legend ก่อน ===
                    if st.session_state[state_key]:
                        st.markdown("**Each Amino Acid Color**", unsafe_allow_html=True)
                        render_legend()
    # === แล้วค่อย Interpretation ===
                    st.markdown("""
        <div style='font-size:1.07rem; font-weight:600; margin-bottom:7px;'>**Interpretation of Probability Ranges**</div>
        <ul style='margin-left: -12px;'>
          <li>⭐️ <b>80-100%</b>: Very High Potential</li>
          <li>🟡 <b>60-79%</b>: High Potential</li>
          <li>🟠 <b>40-59%</b>: Moderate / Uncertain</li>
          <li>🔴 <b>20-39%</b>: Low Potential</li>
          <li>⚫ <b>&lt; 20%</b>: Very Low Potential</li>
        </ul>
    """, unsafe_allow_html=True)

                
    st.markdown("""
    <div style='margin-top:20px; text-align:center; font-size:1.04rem; color:#888;'>
    <b>Training Data Source:</b>
    <a href="http://crdd.osdd.net/raghava/satpdb/" target="_blank">Satpdb</a> |
    <a href="http://peptidome.jcvi.org/peptipedia/" target="_blank">Peptipedia</a> |
    <a href="https://www.uniprot.org/" target="_blank">UniProt</a> |
    <a href="http://aps.unmc.edu/APD3/" target="_blank">APD3</a><br>
    <b>Developed by:</b> Sippatorn Suwanro, Waris Ihaloh, Poramet Sinprasert</a><br>
    <b>Institution:</b> Prince of Songkla University Demonstration School (Secondary)</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
