import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import io

# ===================================================================== 
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    layout="wide",
    page_title="Nodal Analysis - Professional Tool",
    page_icon="⚙️"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .upload-section {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚙️ Nodal Analysis System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced IPR & VLP Analysis Using Beggs-Brill Multiphase Flow Correlation</p>', unsafe_allow_html=True)

# =====================================================================
# HELPER FUNCTIONS - PVT CORRELATIONS & VISCOSITY ESTIMATORS
# =====================================================================
def bubble_point_standing(Rs, gamma_g, API, T):
    """Standing correlation for bubble point pressure (psia)"""
    return 18.2 * ((Rs / max(gamma_g, 1e-6)) ** 0.83) * (10 ** (0.00091 * T - 0.0125 * API)) - 1.4

def Rs_standing(P, Pb, Rs_b, gamma_g, API, T):
    """
    Calculate solution gas-oil ratio as function of pressure
    
    Args:
        P: Current pressure (psia)
        Pb: Bubble point pressure (psia)
        Rs_b: Solution GOR at bubble point (scf/STB)
        gamma_g: Gas specific gravity
        API: API gravity
        T: Temperature (°F)
    
    Returns:
        Rs at pressure P
    """
    if P >= Pb:
        # Undersaturated: Rs remains constant at bubble point value
        return Rs_b
    else:
        # Saturated: Rs decreases as pressure drops below Pb
        Rs = gamma_g * ((P + 1.4) / (18.2 * (10 ** (0.00091 * T - 0.0125 * API)))) ** (1/0.83)
        return max(Rs, 0)

def oil_formation_volume_factor(P, Pb, Rs, Rs_b, Bo_b, gamma_g, gamma_o, T, co=1e-5):
    """
    Oil formation volume factor (bbl/STB)
    
    Args:
        P: Current pressure (psia)
        Pb: Bubble point pressure (psia)
        Rs: Solution GOR at P (scf/STB)
        Rs_b: Solution GOR at bubble point (scf/STB)
        Bo_b: Bo at bubble point (bbl/STB)
        gamma_g: Gas specific gravity
        gamma_o: Oil specific gravity
        T: Temperature (°F)
        co: Oil compressibility above Pb (1/psi)
    
    Returns:
        Bo at pressure P
    """
    if P >= Pb:
        # Undersaturated: Bo decreases slightly with pressure
        Bo = Bo_b * np.exp(-co * (P - Pb))
    else:
        # Saturated: Standing correlation
        Bo = 0.9759 + 0.00012 * (Rs * (gamma_g / gamma_o)**0.5 + 1.25 * T)**1.2
    
    return Bo

def oil_density_from_API(API):
    """Oil density in kg/m3 from API gravity (approx)."""
    return (141.5 / (131.5 + API)) * 999.016

def gas_density_estimate(P_Pa, gamma_g, T_K):
    """Gas density estimation using ideal gas-like relation (approx). Returns kg/m3."""
    R = 8.314  # J/mol.K
    M_air = 28.97 / 1000.0  # kg/mol
    M_g = max(gamma_g, 0.01) * M_air
    Z = 0.88
    return (P_Pa * M_g) / (Z * R * T_K)
# ---------------------------
# Viscosity estimation helpers
# ---------------------------
def estimate_oil_viscosity_cP(API, T_F, mu_oil_input=None):
    """
    Estimate oil viscosity (cP) at reservoir temperature if user didn't provide mu_oil.
    This is an empirical estimator—use measured viscosities if available.
    """
    if mu_oil_input is not None and mu_oil_input > 0:
        return float(mu_oil_input)

    mu_100F = 10 ** (3.0324 - 0.02023 * API)  # cP estimate at 100°F (approx)
    mu_T = mu_100F * np.exp(-0.028 * (T_F - 100.0))
    return max(float(mu_T), 0.001)


def estimate_gas_viscosity_cP(P_Pa, gamma_g, T_K, mu_gas_input=None):
    """
    Estimate gas viscosity (cP) if user didn't provide mu_gas.
    """
    if mu_gas_input is not None and mu_gas_input > 0:
        return float(mu_gas_input)

    mu_base = 1.0e-5 * (T_K / 273.15) ** 0.7  # Pa.s (rough estimate)
    mu_base *= (1.0 + 0.2 * (gamma_g - 0.6))
    mu_cP = mu_base / 0.001  # convert Pa.s -> cP
    return max(float(mu_cP), 1e-4)


# =====================================================================
# EXCEL IMPORT FUNCTION
# =====================================================================

def load_excel_data(uploaded_file):
    """Load data from Excel file and return as dictionary (first-row mapping)."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)

        data = {}
        if {'Parameter', 'Value'}.issubset(set(df.columns)):
            for _, row in df.iterrows():
                key = str(row['Parameter']).lower().strip()
                val = row['Value']
                if pd.notna(val):
                    data[key] = val
            return data, None

        first_row = df.iloc[0]
        for col in df.columns:
            col_lower = str(col).lower().strip()
            val = first_row[col]
            if pd.notna(val):
                data[col_lower] = val

        return data, None
    except Exception as e:
        return None, str(e)


def get_value_from_excel(excel_data, keys, default):
    """Try multiple key variations to get value from Excel data"""
    if excel_data is None:
        return default

    for key in keys:
        if key in excel_data:
            try:
                return float(excel_data[key])
            except:
                try:
                    return float(str(excel_data[key]).strip())
                except:
                    pass
    return default

# =====================================================================
# BEGGS-BRILL VLP CALCULATION (Cached)
# =====================================================================
import numpy as np
from math import pi, sin, cos, radians, log

@st.cache_data
def calculate_vlp_beggs_brill(q_stb, d_inch, P_wh, P_res, depth_ft, GOR,
                               rho_oil, rho_gas, mu_oil_cp, mu_gas_cp,
                               sigma, theta_deg=90):
    """
    Improved Beggs & Brill Vertical Lift Performance (VLP) calculation.

    Parameters:
        q_stb: oil flow rate (STB/day)
        d_inch: tubing diameter (inch)
        P_wh: wellhead pressure (psia)
        P_res: reservoir pressure (psia)
        depth_ft: true vertical depth (ft)
        GOR: gas-oil ratio (scf/STB)
        rho_oil: oil density (kg/m³)
        rho_gas: gas density (kg/m³)
        mu_oil_cp: oil viscosity (cp)
        mu_gas_cp: gas viscosity (cp)
        sigma: surface tension (N/m)
        theta_deg: inclination angle from horizontal (deg)

    Returns:
        pwf_psia: bottomhole flowing pressure (psia)
    """

    # --- Basic checks & conversions ---
    q_stb = max(q_stb, 0.0)
    effective_depth_ft = max(depth_ft * 0.8, 1.0)

    if q_stb < 1e-4:
        # very low flow fallback
        static_gradient = 0.35  # psi/ft typical
        return P_wh + static_gradient * effective_depth_ft

    g = 9.81  # m/s²
    d_m = max(d_inch * 0.0254, 1e-4)
    depth_m = max(effective_depth_ft * 0.3048, 0.1)
    A = pi * (d_m ** 2) / 4
    mu_oil_pas = max(mu_oil_cp, 1e-6) * 0.001
    mu_gas_pas = max(mu_gas_cp, 1e-6) * 0.001

    # --- Convert flow rates to m³/s ---
    ql_m3s = (q_stb / 86400.0) * 0.158987  # oil
    qg_std = (GOR * q_stb * 0.028316846592) / 86400.0  # gas

    # Approximate gas expansion (ideal gas)
    P_avg = (P_wh + P_res) / 2.0
    expansion = 14.7 / max(P_avg, 14.7)
    qg_m3s = qg_std * expansion * 1.5

    # --- Superficial velocities ---
    usl = ql_m3s / (A + 1e-12)
    usg = qg_m3s / (A + 1e-12)
    ut = max(usl + usg, 1e-10)

    # --- Initial holdup ---
    ho = np.clip(usl / ut, 0.01, 0.99)

    # --- Dimensionless numbers ---
    Nvl = usl * (rho_oil / (g * sigma + 1e-10)) ** 0.25 if sigma > 0 else 1.0
    Nvg = max(usg, 1e-10) * (rho_gas / (g * sigma + 1e-10)) ** 0.25 if sigma > 0 else 0.1
    L_m = Nvl / (Nvl + Nvg + 1e-10)
    Fr = (ut ** 2) / (g * d_m + 1e-10)

    # --- Flow pattern correlation constants ---
    L1 = 316.0 * (L_m ** 0.302)
    L3 = 0.1 * (L_m ** (-1.4516))
    if L_m < 0.01 and Fr < L1:
        a, b, c = 0.98, 0.4846, 0.0868
    elif Fr >= L3:
        a, b, c = 1.065, 0.5824, 0.0609
    else:
        a, b, c = 0.845, 0.5351, 0.0173

    E_L0 = np.clip(a * ho ** b / (Fr ** c + 1e-10), 0.05, 0.95)

    # --- Inclination correction ---
    if theta_deg > 0:
        try:
            psi = 1 + (1 - L_m) * log(max(0.011 * (L_m ** -3.768) * (Nvl ** 3.539) * ((theta_deg / 90) ** -1.614), 0.001)) \
                    * (sin(radians(1.8 * theta_deg)) - 0.333 * sin(radians(1.8 * theta_deg)) ** 3)
            psi = max(psi, 0.1)
        except ValueError:
            psi = 1.0
    else:
        psi = 1.0

    El = np.clip(E_L0 * psi, 0.05, 0.95)

    # --- Mixture density & viscosity (with slip effect) ---
    gas_expansion_factor = 1 + (1 - El) * (1 - expansion) * 3
    rho_g_effective = rho_gas / max(gas_expansion_factor, 1e-6)
    rho_m = El * rho_oil + (1 - El) * rho_g_effective
    mu_m = mu_oil_pas * El + mu_gas_pas * (1 - El)

    # --- Reynolds number & friction factor (Haaland) ---
    Re = max(rho_m * ut * d_m / (mu_m + 1e-10), 1.0)
    epsilon = 0.000045  # tubing roughness
    epsilon_rel = epsilon / d_m
    if Re < 2000:
        f = 64 / Re
    else:
        # Haaland approx
        f = 0.11 * (epsilon_rel + 68 / Re) ** 0.25
    f = np.clip(f, 0.001, 0.1)

    # --- Turbulent friction adjustment ---
    y = L_m / (El ** 2 + 1e-10)
    try:
        if 1.0 < y < 1.2:
            S = log(2.2 * y - 1.2)
        else:
            ln_y = log(max(y, 0.01))
            S = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y ** 2 + 0.01853 * ln_y ** 4 + 1e-10)
    except ValueError:
        S = 0
    f_tp = np.clip(f * np.exp(S), f, f * 10)

    # --- Pressure drops ---
    dp_elevation = rho_m * g * cos(radians(theta_deg))
    dp_friction = 2 * f_tp * rho_m * ut ** 2 / d_m
    dp_total = (dp_elevation + dp_friction) * depth_m

    # --- Convert to psia ---
    pwf_pa = P_wh * 6894.76 + dp_total
    pwf_psia = pwf_pa / 6894.76

    return max(pwf_psia, 0.0)


# =====================================================================
# IPR MODELS (Cached) - NOTE: Straight-line removed by request
# =====================================================================

@st.cache_data
def calculate_ipr(q_array, method, PI_eff, P_res, q_max, n_fetkovich):
    """Calculate IPR curve (pwf in psia) for an array of rates q_array (bpd).
       Straight-line formulation has been removed per request; supported methods:
       - Vogel
       - Fetkovich
    """
    q_array = np.array(q_array, dtype=float)

    if method == "Vogel":
        pwf = np.zeros_like(q_array)
        a = 0.8
        b = 0.2
        for i, q in enumerate(q_array):
            ratio = float(q) / max(q_max, 1e-12)
            c = ratio - 1.0
            disc = b * b - 4 * a * c
            if disc < 0:
                x = 0.0
            else:
                x1 = (-b + np.sqrt(disc)) / (2 * a)
                x2 = (-b - np.sqrt(disc)) / (2 * a)
                candidates = [x for x in (x1, x2) if 0.0 <= x <= 1.0]
                x = candidates[0] if candidates else max(min(x1, 1.0), 0.0)
            pwf[i] = x * P_res
        return pwf

    elif method == "Fetkovich":
        C = q_max / (max(P_res, 1e-12) ** n_fetkovich)
        pwf = np.zeros_like(q_array)
        for i, q in enumerate(q_array):
            term = max(P_res ** n_fetkovich - q / max(C, 1e-12), 0.0)
            pwf[i] = term ** (1.0 / n_fetkovich) if term > 0 else 0.0
        return pwf

    # fallback: return NaN array if method is unknown
    return np.full_like(q_array, np.nan)

# =====================================================================
# EXCEL UPLOAD SECTION
# =====================================================================

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📁 Data Import (Optional)")

col_upload1, col_upload2 = st.columns([2, 1])

with col_upload1:
    uploaded_file = st.file_uploader(
        "Upload Excel file with parameters",
        type=['xlsx', 'xls'],
        help="Excel file should contain parameter names in first row and values in second row or a two-column Parameter/Value layout"
    )

with col_upload2:
    if st.button("📄 Download Template", use_container_width=True):
        template_data = {
            'Parameter': ['P_res', 'T_res', 'API', 'gamma_g', 'PI', 'skin', 'q_max',
                          'Rs_b', 'Bo_b', 'mu_oil', 'mu_gas', 'sigma', 'GOR',
                          'P_wellhead', 'well_depth', 'theta_deg'],
            'Value': [4500, 180, 35, 0.65, 2.5, 0, 1500, 600, 1.25, 2.0, 0.012, 0.025,
                      500, 300, 5000, 90],
            'Unit': ['psia', '°F', '-', '-', 'STB/d/psi', '-', 'STB/d', 'scf/STB',
                     'rb/STB', 'cP', 'cP', 'N/m', 'scf/STB', 'psia', 'ft', '°']
        }
        template_df = pd.DataFrame(template_data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Parameters')

        st.download_button(
            "⬇️ Download",
            output.getvalue(),
            "nodal_analysis_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Load Excel data if uploaded
excel_data = None
if uploaded_file is not None:
    excel_data, error = load_excel_data(uploaded_file)
    if error:
        st.error(f"❌ Error loading Excel file: {error}")
    else:
        st.success("✅ Excel file loaded successfully!")

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# SIDEBAR INPUTS
# =====================================================================

with st.sidebar:
    st.markdown("## 📝 Input Parameters")

    with st.expander("🏔️ RESERVOIR DATA", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            P_res = st.number_input(
                "Reservoir Pressure (psia)",
                value=get_value_from_excel(excel_data, ['p_res', 'reservoir_pressure', 'pr'], 4500.0),
                step=100.0
            )
            T_res = st.number_input(
                "Temperature (°F)",
                value=get_value_from_excel(excel_data, ['t_res', 'temperature', 'temp'], 180.0),
                step=5.0
            )
        with col2:
            API = st.number_input(
                "API Gravity",
                value=get_value_from_excel(excel_data, ['api', 'api_gravity'], 35.0),
                step=1.0
            )
            gamma_g = st.number_input(
                "Gas Sp. Gravity",
                value=get_value_from_excel(excel_data, ['gamma_g', 'gas_gravity', 'sg_gas'], 0.65),
                step=0.01
            )

    # NOTE: Removed "Straight Line (PI)" option by request.
    with st.expander("📊 IPR PARAMETERS", expanded=True):
        ipr_method = st.selectbox("IPR Method", ["Vogel", "Fetkovich"])

        col1, col2 = st.columns(2)
        with col1:
            PI_nominal = st.number_input(
                "Productivity Index (STB/d/psi)",
                value=get_value_from_excel(excel_data, ['pi', 'productivity_index'], 2.5),
                format="%.4f"
            )
            skin = st.number_input(
                "Skin Factor",
                value=get_value_from_excel(excel_data, ['skin', 'skin_factor'], 0.0),
                step=0.5
            )
        with col2:
            q_max = st.number_input(
                "Max Rate (STB/d)",
                value=get_value_from_excel(excel_data, ['q_max', 'max_rate', 'qmax'], 1500.0),
                step=100.0
            )
            if ipr_method == "Fetkovich":
                n_fetkovich = st.number_input("Fetkovich Exponent (n)", value=1.0, min_value=0.5, max_value=2.0, step=0.1)
            else:
                n_fetkovich = 1.0

    PI_eff = PI_nominal / (1.0 + skin) if (1.0 + skin) != 0 else PI_nominal

    with st.expander("🧪 FLUID PROPERTIES", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Rs_b = st.number_input(
                "Solution GOR (scf/STB)",
                value=get_value_from_excel(excel_data, ['rs_b', 'rs', 'solution_gor'], 600.0),
                step=50.0
            )
            Bo_b = st.number_input(
                "Bo (rb/STB)",
                value=get_value_from_excel(excel_data, ['bo_b', 'bo', 'fvf'], 1.25),
                step=0.01
            )
            mu_oil = st.number_input(
                "Oil Viscosity (cP) (optional - leave 0 to estimate)",
                value=get_value_from_excel(excel_data, ['mu_oil', 'oil_viscosity', 'viscosity'], 2.0),
                step=0.1
            )
        with col2:
            mu_gas = st.number_input(
                "Gas Viscosity (cP) (optional - leave 0 to estimate)",
                value=get_value_from_excel(excel_data, ['mu_gas', 'gas_viscosity'], 0.012),
                step=0.001,
                format="%.4f"
            )
            sigma = st.number_input(
                "Surface Tension (N/m)",
                value=get_value_from_excel(excel_data, ['sigma', 'surface_tension', 'ift'], 0.025),
                step=0.001,
                format="%.4f"
            )
            GOR = st.number_input(
                "Producing GOR (scf/STB)",
                value=get_value_from_excel(excel_data, ['gor', 'producing_gor'], 500.0),
                step=50.0
            )

    with st.expander("🔧 WELLBORE CONFIGURATION", expanded=True):
        st.markdown("**Wellhead Pressure:**")
        P_wellhead = st.number_input(
            "Pwh (psia)",
            value=get_value_from_excel(excel_data, ['p_wellhead', 'pwh', 'wellhead_pressure'], 300.0),
            step=50.0,
            help="Flowing wellhead pressure"
        )

        col1, col2 = st.columns(2)
        with col1:
            well_depth = st.number_input(
                "Well Depth (ft)",
                value=get_value_from_excel(excel_data, ['well_depth', 'depth', 'tvd'], 5000.0),
                step=100.0
            )
        with col2:
            theta_deg = st.number_input(
                "Deviation (°)",
                value=get_value_from_excel(excel_data, ['theta_deg', 'deviation', 'angle'], 90.0),
                step=5.0
            )

        st.markdown("**Tubing Configuration:**")

        tubing_options = ["2.259", "2.441", "2.875", "3.5", "4.0", "4.5"]
        tubing_sizes = st.multiselect(
            "Select Standard Sizes (inches)",
            tubing_options,
            default=["2.875", "3.5"],
            help="Select multiple standard tubing sizes"
        )

        st.markdown("**Add Custom Tubing IDs:**")
        custom_tubing_input = st.text_input(
            "Custom IDs (comma-separated)",
            placeholder="e.g., 2.5, 3.25, 5.0",
            help="Enter custom tubing inner diameters in inches, separated by commas"
        )

        if custom_tubing_input:
            try:
                custom_sizes = [float(x.strip()) for x in custom_tubing_input.split(',') if x.strip()]
                for size in custom_sizes:
                    if 1.0 <= size <= 10.0:
                        size_str = f"{size:.3f}"
                        if size_str not in tubing_sizes:
                            tubing_sizes.append(size_str)
                    else:
                        st.warning(f"⚠️ Size {size} is outside reasonable range (1-10 inches)")
            except ValueError:
                st.error("❌ Invalid format. Use numbers separated by commas.")

    with st.expander("⚙️ CALCULATION SETTINGS"):
        n_points = st.slider("Curve Resolution", 100, 500, 300, 50)
        show_markers = st.checkbox("Show Data Points", value=True)
        show_grid = st.checkbox("Show Grid", value=True)

# =====================================================================
# CALCULATIONS
# =====================================================================

rho_oil = oil_density_from_API(API)
T_K = (T_res - 32) * 5 / 9 + 273.15
P_avg_pa = ((P_res + P_wellhead) / 2.0) * 6894.76
rho_gas = gas_density_estimate(P_avg_pa, gamma_g, T_K)
Pb_est = bubble_point_standing(Rs_b, gamma_g, API, T_res)

mu_oil_estimated = estimate_oil_viscosity_cP(API, T_res, mu_oil_input=mu_oil if mu_oil > 0 else None)
mu_gas_estimated = estimate_gas_viscosity_cP(P_avg_pa, gamma_g, T_K, mu_gas_input=mu_gas if mu_gas > 0 else None)

mu_oil_used = mu_oil if mu_oil > 0 else mu_oil_estimated
mu_gas_used = mu_gas if mu_gas > 0 else mu_gas_estimated

q_range = np.linspace(0, q_max * 1.3, n_points)

pwf_ipr = calculate_ipr(q_range, ipr_method, PI_eff, P_res, q_max, n_fetkovich)

vlp_results = {}
if not tubing_sizes:
    tubing_sizes = ["3.5"]

with st.spinner("🔄 Computing VLP curves using Beggs-Brill correlation..."):
    for tubing_id in tubing_sizes:
        d_inch = float(tubing_id)
        pwf_vlp = []

        for q in q_range:
            pwf = calculate_vlp_beggs_brill(
                q, d_inch, P_wellhead, P_res, well_depth, GOR,
                rho_oil, rho_gas, mu_oil_used, mu_gas_used, sigma, theta_deg
            )
            pwf_vlp.append(pwf)

        vlp_results[f'{tubing_id}"'] = np.array(pwf_vlp)

operating_points = []
for label, pwf_vlp in vlp_results.items():
    diff = pwf_ipr - pwf_vlp
    sign_changes = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]

    if len(sign_changes) > 0:
        idx = sign_changes[0]
        q1, q2 = q_range[idx], q_range[idx + 1]
        p1, p2 = diff[idx], diff[idx + 1]

        if (p2 - p1) != 0:
            q_op = q1 - p1 * (q2 - q1) / (p2 - p1)
            pwf_op = np.interp(q_op, q_range, pwf_ipr)
        else:
            q_op = (q1 + q2) / 2.0
            pwf_op = (pwf_ipr[idx] + pwf_ipr[idx + 1]) / 2.0

        operating_points.append({
            "Tubing": label,
            "Rate (bpd)": round(q_op, 1),
            "Pwf (psia)": round(pwf_op, 1),
            "Drawdown (psi)": round(P_res - pwf_op, 1),
            "Drawdown %": round(100.0 * (P_res - pwf_op) / max(P_res, 1e-6), 1)
        })

# =====================================================================
# DISPLAY RESULTS
# =====================================================================

st.markdown("---")
st.markdown("## 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Reservoir Pr", f"{P_res:.0f} psia", help="Static pressure")
with col2:
    st.metric("Wellhead Pwh", f"{P_wellhead:.0f} psia", help="Flowing WHP")
with col3:
    st.metric("PI Effective", f"{PI_eff:.3f}", delta=f"Skin: {skin:+.1f}")
with col4:
    st.metric("Bubble Point", f"{Pb_est:.0f} psia", help="Estimated Pb")
with col5:
    if operating_points:
        max_rate = max([op["Rate (bpd)"] for op in operating_points])
        st.metric("Max Rate", f"{max_rate:.0f} bpd", help="Best tubing")
    else:
        st.metric("Max Rate", "N/A")

# =====================================================================
# MAIN NODAL PLOT
# =====================================================================

st.markdown("---")
st.markdown("## 📈 Nodal Analysis Chart")

fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

vlp_colors = ['#FF1493', '#FF6347', '#FF8C00', '#FFD700', '#32CD32', '#1E90FF']
line_styles = ['-', '-', '-', '--', '--', '-.']

for idx, (label, pwf_vlp) in enumerate(vlp_results.items()):
    color = vlp_colors[idx % len(vlp_colors)]
    style = line_styles[idx % len(line_styles)]

    ax.plot(q_range, pwf_vlp,
            color=color,
            linewidth=4.0,
            linestyle=style,
            label=f"VLP - Tubing {label}",
            marker='o' if show_markers else None,
            markersize=5,
            markevery=max(1, int(n_points / 12)),
            alpha=0.9,
            zorder=5)

ax.plot(q_range, pwf_ipr,
        color='#0000CD',
        linewidth=5.0,
        linestyle='-',
        label=f"IPR ({ipr_method})",
        marker='s' if show_markers else None,
        markersize=6,
        markevery=max(1, int(n_points / 12)),
        zorder=10,
        alpha=0.95)

for idx, op in enumerate(operating_points):
    q_op = op["Rate (bpd)"]
    pwf_op = op["Pwf (psia)"]

    ax.scatter(q_op, pwf_op,
               color='yellow',
               s=600,
               marker='o',
               alpha=0.3,
               zorder=18)

    ax.scatter(q_op, pwf_op,
               color='#FF1493',
               s=400,
               marker='o',
               edgecolors='black',
               linewidths=4,
               zorder=20,
               alpha=1.0)

    ax.scatter(q_op, pwf_op,
               color='white',
               s=100,
               marker='o',
               zorder=21)

    x_offset = q_max * 0.12
    y_offset = P_res * 0.08

    if q_op > q_max * 0.7:
        x_offset = -x_offset - q_max * 0.05
    if pwf_op > P_res * 0.7:
        y_offset = -y_offset

    ax.annotate(
        f"Operating Point {idx+1}\n" +
        f"q = {q_op:.1f} bpd\n" +
        f"Pwf = {pwf_op:.1f} psia\n" +
        f"Tubing: {op['Tubing']}",
        xy=(q_op, pwf_op),
        xytext=(q_op + x_offset, pwf_op + y_offset),
        fontsize=12,
        color='black',
        weight='bold',
        bbox=dict(boxstyle="round,pad=1.0",
                  facecolor='#FFFACD',
                  edgecolor='#FF1493',
                  alpha=0.95,
                  linewidth=3),
        arrowprops=dict(arrowstyle="->, head_width=0.6, head_length=0.8",
                        color='#FF1493',
                        lw=3,
                        connectionstyle="arc3,rad=0.3"),
        zorder=22)

ax.axhline(Pb_est,
           color='#8B4513',
           linestyle='--',
           linewidth=3.0,
           alpha=0.8,
           label=f'Bubble Point = {Pb_est:.0f} psia',
           zorder=3)

ax.text(q_range.max() * 0.02, Pb_est + P_res * 0.03,
        f'Pb ≈ {Pb_est:.0f} psia',
        color='white',
        fontsize=13,
        weight='bold',
        bbox=dict(boxstyle='round,pad=0.7',
                  facecolor='#8B4513',
                  edgecolor='black',
                  linewidth=2,
                  alpha=0.9))

ax.set_xlabel("Rate, q (bpd)", fontsize=18, fontweight='bold', color='black', labelpad=10)
ax.set_ylabel("Pressure, Pwf (psia)", fontsize=18, fontweight='bold', color='black', labelpad=10)
ax.set_title("Nodal Analysis", fontsize=40, fontweight='bold', color='#1f77b4', pad=25)

if show_grid:
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5, color='#CCCCCC', zorder=1)
    ax.set_axisbelow(True)

ax.tick_params(colors='black', labelsize=14, width=2.5, length=8)
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2.5)

# Enhanced legend (removed invalid kwargs)
legend = ax.legend(loc='lower left',
                   fontsize=13,
                   framealpha=0.98,
                   edgecolor='black',
                   fancybox=True,
                   shadow=True,
                   borderpad=1.2,
                   labelspacing=1.0,
                   handlelength=2.5)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_linewidth(2)
for text in legend.get_texts():
    text.set_color('black')
    text.set_weight('bold')

max_pressure = max(P_res * 1.15, np.max(pwf_ipr) * 1.05)
max_rate = q_range.max() * 1.05

ax.set_xlim(0, max_rate)
ax.set_ylim(0, max_pressure)

ax.minorticks_on()
# use 'colors' (plural) to set tick color; do not pass alpha here (not supported by tick_params)
ax.tick_params(which='minor', length=4, colors='gray')

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =====================================================================
# OPERATING POINTS TABLE
# =====================================================================

st.markdown("---")
st.markdown("## 🎯 Operating Points Summary")

if operating_points:
    df_op = pd.DataFrame(operating_points)

    st.dataframe(
        df_op.style.background_gradient(cmap='YlOrRd', subset=['Rate (bpd)']),
        use_container_width=True,
        height=250
    )
else:
    st.warning("⚠️ No intersection found. Adjust parameters.")



# =====================================================================
# PVT PROPERTIES ANALYSIS - NEW SECTION
# =====================================================================

st.markdown("---")
st.markdown('<p class="pvt-header">🧬 PVT Properties Analysis</p>', unsafe_allow_html=True)
st.markdown("### Pressure-Dependent Fluid Behavior")

# Calculate PVT properties over pressure range
P_range_pvt = np.linspace(500, P_res * 1.2, 200)
Rs_array = np.zeros_like(P_range_pvt)
Bo_array = np.zeros_like(P_range_pvt)

for i, P in enumerate(P_range_pvt):
    Rs_array[i] = Rs_standing(P, Pb_est, Rs_b, gamma_g, API, T_res)
    Bo_array[i] = oil_formation_volume_factor(P, Pb_est, Rs_array[i], Rs_b, Bo_b, gamma_g, rho_oil / 999.016 * 141.5 / 131.5, T_res)

# Create comprehensive PVT plot
fig_pvt = plt.figure(figsize=(18, 12), dpi=100)
fig_pvt.patch.set_facecolor('white')
gs = fig_pvt.add_gridspec(3, 2, hspace=0.35, wspace=0.30)

# Plot 1: Rs vs Pressure
ax1 = fig_pvt.add_subplot(gs[0, :])
ax1.set_facecolor('white')
ax1.plot(P_range_pvt, Rs_array, 'b-', linewidth=4, label='Solution GOR (Rs)', marker='o', 
         markevery=15, markersize=8, markerfacecolor='white', markeredgewidth=3, markeredgecolor='blue')
ax1.axvline(Pb_est, color='red', linestyle='--', linewidth=3, alpha=0.8, label=f'Bubble Point = {Pb_est:.0f} psia')
ax1.axhline(Rs_b, color='green', linestyle=':', linewidth=2.5, alpha=0.7, label=f'Rs at Pb = {Rs_b:.0f} scf/STB')

# Add shaded regions
ax1.axvspan(P_range_pvt.min(), Pb_est, alpha=0.15, color='orange', label='Saturated Region')
ax1.axvspan(Pb_est, P_range_pvt.max(), alpha=0.15, color='lightblue', label='Undersaturated Region')

ax1.set_xlabel("Pressure (psia)", fontsize=14, fontweight='bold', color='black')
ax1.set_ylabel("Solution GOR, Rs (scf/STB)", fontsize=14, fontweight='bold', color='black')
ax1.set_title("Solution Gas-Oil Ratio vs Pressure", fontsize=16, fontweight='bold', color='#1f77b4')
ax1.grid(True, alpha=0.4, linestyle='--', color='#CCCCCC')
ax1.legend(fontsize=11, framealpha=0.95, edgecolor='black', loc='best')
ax1.tick_params(colors='black', labelsize=12)
for spine in ax1.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2.5)

# Add annotation
ax1.annotate(f'At Pb: Rs remains constant\nabove bubble point',
             xy=(Pb_est, Rs_b), xytext=(Pb_est + (P_range_pvt.max() - Pb_est) * 0.3, Rs_b * 0.85),
             fontsize=11, color='black', weight='bold',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='black', linewidth=2),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Plot 2: Bo vs Pressure
ax2 = fig_pvt.add_subplot(gs[1, :])
ax2.set_facecolor('white')
ax2.plot(P_range_pvt, Bo_array, 'g-', linewidth=4, label='Oil FVF (Bo)', marker='s',
         markevery=15, markersize=8, markerfacecolor='white', markeredgewidth=3, markeredgecolor='green')
ax2.axvline(Pb_est, color='red', linestyle='--', linewidth=3, alpha=0.8, label=f'Bubble Point = {Pb_est:.0f} psia')
ax2.axhline(Bo_b, color='purple', linestyle=':', linewidth=2.5, alpha=0.7, label=f'Bo at Pb = {Bo_b:.3f} rb/STB')

# Add shaded regions
ax2.axvspan(P_range_pvt.min(), Pb_est, alpha=0.15, color='orange')
ax2.axvspan(Pb_est, P_range_pvt.max(), alpha=0.15, color='lightblue')

ax2.set_xlabel("Pressure (psia)", fontsize=14, fontweight='bold', color='black')
ax2.set_ylabel("Oil FVF, Bo (rb/STB)", fontsize=14, fontweight='bold', color='black')
ax2.set_title("Oil Formation Volume Factor vs Pressure", fontsize=16, fontweight='bold', color='#1f77b4')
ax2.grid(True, alpha=0.4, linestyle='--', color='#CCCCCC')
ax2.legend(fontsize=11, framealpha=0.95, edgecolor='black', loc='best')
ax2.tick_params(colors='black', labelsize=12)
for spine in ax2.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2.5)

# Plot 3: Combined Rs & Bo (Dual Y-axis)
ax3 = fig_pvt.add_subplot(gs[2, 0])
ax3.set_facecolor('white')
ax3_twin = ax3.twinx()

line1 = ax3.plot(P_range_pvt, Rs_array, 'o-', color='#0066CC', linewidth=3, 
                 markersize=6, label='Rs (scf/STB)', markevery=20)
line2 = ax3_twin.plot(P_range_pvt, Bo_array, 's-', color='#CC0000', linewidth=3,
                      markersize=6, label='Bo (rb/STB)', markevery=20)

ax3.axvline(Pb_est, color='black', linestyle='--', linewidth=2.5, alpha=0.7)
ax3.set_xlabel("Pressure (psia)", fontsize=13, fontweight='bold', color='black')
ax3.set_ylabel("Rs (scf/STB)", fontsize=13, fontweight='bold', color='#0066CC')
ax3_twin.set_ylabel("Bo (rb/STB)", fontsize=13, fontweight='bold', color='#CC0000')
ax3.set_title("Combined PVT Behavior", fontsize=14, fontweight='bold', color='#1f77b4')
ax3.tick_params(axis='y', labelcolor='#0066CC', labelsize=11)
ax3.tick_params(axis='x', colors='black', labelsize=11)
ax3_twin.tick_params(axis='y', labelcolor='#CC0000', labelsize=11)
ax3.grid(True, alpha=0.4, color='#CCCCCC')

for spine in ax3.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)
for spine in ax3_twin.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='best', fontsize=10, framealpha=0.95, edgecolor='black')

# Plot 4: Saturation Status
ax4 = fig_pvt.add_subplot(gs[2, 1])
ax4.set_facecolor('white')

# Create saturation indicator
saturation_status = np.where(P_range_pvt < Pb_est, 1, 2)  # 1=Saturated, 2=Undersaturated
colors_sat = ['#FF6B6B' if s == 1 else '#4ECDC4' for s in saturation_status]

scatter = ax4.scatter(P_range_pvt, Rs_array / Rs_b * 100, c=colors_sat, s=80, alpha=0.7, edgecolors='black', linewidth=1.5)
ax4.axvline(Pb_est, color='black', linestyle='--', linewidth=3, alpha=0.8)
ax4.axhline(100, color='gray', linestyle=':', linewidth=2)

ax4.set_xlabel("Pressure (psia)", fontsize=13, fontweight='bold', color='black')
ax4.set_ylabel("Rs (% of Rs at Pb)", fontsize=13, fontweight='bold', color='black')
ax4.set_title("Saturation Status Map", fontsize=14, fontweight='bold', color='#1f77b4')
ax4.grid(True, alpha=0.4, color='#CCCCCC')
ax4.tick_params(colors='black', labelsize=11)
for spine in ax4.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Add legend for saturation status
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#FF6B6B', edgecolor='black', label='Saturated (P < Pb)'),
                   Patch(facecolor='#4ECDC4', edgecolor='black', label='Undersaturated (P > Pb)')]
ax4.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.95, edgecolor='black')

# Add text annotation


# ==========================================================
# FINALIZE AND DISPLAY PLOT
# ==========================================================
plt.tight_layout()
st.pyplot(fig_pvt)
plt.close(fig_pvt)

# ==========================================================
# PVT SUMMARY TABLE
# ==========================================================
st.markdown("### 📋 PVT Properties Summary Table")

df_pvt = pd.DataFrame({
    "Pressure (psia)": P_range_pvt,
    "Rs (scf/STB)": Rs_array,
    "Bo (rb/STB)": Bo_array,
    "Saturation State": np.where(P_range_pvt < Pb_est, "Saturated", "Undersaturated")
})

st.dataframe(
    df_pvt.style
        .format({
            "Pressure (psia)": "{:.1f}",
            "Rs (scf/STB)": "{:.2f}",
            "Bo (rb/STB)": "{:.4f}"
        })
        .applymap(
            lambda x: 'background-color: #FFEEEE' if x == "Saturated"
            else 'background-color: #EEFFFF',
            subset=["Saturation State"]
        ),
    use_container_width=True,
    height=350
)

# ==========================================================
# ENGINEERING INSIGHT BOX
# ==========================================================
st.markdown("### 🧠 Engineering Insight")

if P_res >= Pb_est:
    st.success(
        f"""
        **Reservoir Condition: UNDERSATURATED**

        • Reservoir pressure ({P_res:.0f} psia) is above bubble point  
        • No free gas exists in the reservoir  
        • Rs remains constant at **{Rs_b:.0f} scf/STB**  
        • Oil compressibility dominates behavior
        """
    )
else:
    st.warning(
        f"""
        **Reservoir Condition: SATURATED**

        • Reservoir pressure ({P_res:.0f} psia) is below bubble point  
        • Free gas is present in the reservoir  
        • Rs decreases with pressure  
        • Oil volume factor decreases rapidly
        """
    )

# ==========================================================
# END OF PVT SECTION
# ==========================================================

# =====================================================================
# SENSITIVITY ANALYSIS
# =====================================================================

st.markdown("---")
st.markdown("## 🔬 Sensitivity Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Wellhead Pressure Impact", "🔩 Tubing Size Comparison", "💧 GOR Sensitivity"])

with tab1:
    st.markdown("### Effect of Wellhead Pressure on VLP Performance")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        whp_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        colors_whp = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']

        for idx, factor in enumerate(whp_factors):
            whp_test = P_wellhead * factor
            pwf_test = []

            for q in q_range[::5]:
                pwf = calculate_vlp_beggs_brill(
                    q, 3.5, whp_test, P_res, well_depth, GOR,
                    rho_oil, rho_gas, mu_oil_used, mu_gas_used, sigma, theta_deg
                )
                pwf_test.append(pwf)

            ax1.plot(q_range[::5], pwf_test,
                     color=colors_whp[idx],
                     linewidth=3,
                     label=f"Pwh = {whp_test:.0f} psia ({factor*100:.0f}%)",
                     marker='o',
                     markersize=4)

        ax1.plot(q_range, pwf_ipr, 'c--', linewidth=2.5, label='IPR', alpha=0.7)
        ax1.set_xlabel("Rate (bpd)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Pwf (psia)", fontsize=12, fontweight='bold')
        ax1.set_title("Wellhead Pressure Sensitivity", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown("#### 📝 Key Insights")
        st.info("""
        **Lower Pwh:**
        - Steeper VLP curves
        - Lower production rates
        - Higher required Pwf

        **Higher Pwh:**
        - Flatter VLP curves  
        - Higher production rates
        - More efficient lifting
        """)

with tab2:
    st.markdown("### Tubing Size Impact on Production Performance")

    tubing_test = sorted([float(t.strip('"')) for t in vlp_results.keys()])
    q_ops_test = []
    pwf_ops_test = []
    drawdown_test = []
    efficiency_test = []
    tubing_labels = []

    for tub in tubing_test:
        pwf_test = []
        for q in q_range:
            pwf = calculate_vlp_beggs_brill(
                q, tub, P_wellhead, P_res, well_depth, GOR,
                rho_oil, rho_gas, mu_oil_used, mu_gas_used, sigma, theta_deg
            )
            pwf_test.append(pwf)

        diff = pwf_ipr - np.array(pwf_test)
        sc = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]

        if len(sc) > 0:
            idx = sc[0]
            q1, q2 = q_range[idx], q_range[idx + 1]
            p1, p2 = diff[idx], diff[idx + 1]
            if (p2 - p1) != 0:
                q_op = q1 - p1 * (q2 - q1) / (p2 - p1)
                pwf_op = np.interp(q_op, q_range, pwf_ipr)
                q_ops_test.append(q_op)
                pwf_ops_test.append(pwf_op)
                drawdown_test.append(P_res - pwf_op)
                efficiency_test.append(100.0 * q_op / max(q_max, 1e-12))
                tubing_labels.append(f'{tub:.3f}"')

    if q_ops_test:
        fig2 = plt.figure(figsize=(18, 11))
        fig2.patch.set_facecolor('white')
        gs = fig2.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

        ax1 = fig2.add_subplot(gs[0, :])
        ax1.set_facecolor('white')

        colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(q_ops_test)))
        bars = ax1.bar(tubing_labels, q_ops_test,
                       color=colors_bar,
                       edgecolor='black',
                       linewidth=3,
                       alpha=0.85,
                       width=0.6)

        for bar, q in zip(bars, q_ops_test):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{q:.0f}',
                     ha='center', va='bottom',
                     fontsize=14, fontweight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                               edgecolor='black', linewidth=2, alpha=0.8))

        x_numeric = np.arange(len(tubing_labels))
        z = np.polyfit(x_numeric, q_ops_test, 2 if len(q_ops_test) >= 3 else 1)
        p = np.poly1d(z)
        ax1.plot(x_numeric, p(x_numeric), "r--", linewidth=4, alpha=0.8,
                 label='Polynomial Trend', marker='d', markersize=8)

        ax1.set_xlabel("Tubing Inner Diameter (inches)", fontsize=16, fontweight='bold', color='black')
        ax1.set_ylabel("Production Rate (bpd)", fontsize=16, fontweight='bold', color='black')
        ax1.set_title("Production Rate vs Tubing Size", fontsize=18, fontweight='bold',
                      color='#1f77b4', pad=15)
        ax1.grid(True, alpha=0.4, axis='y', linestyle='--', color='#CCCCCC')
        ax1.legend(fontsize=13, framealpha=0.95, edgecolor='black')
        ax1.tick_params(colors='black', labelsize=13)
        for spine in ax1.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2.5)

        ax2 = fig2.add_subplot(gs[1, 0])
        ax2.set_facecolor('white')
        ax2_twin = ax2.twinx()

        line1 = ax2.plot(tubing_labels, q_ops_test, 'o-',
                         color='#32CD32', linewidth=4, markersize=12,
                         label='Production Rate', markerfacecolor='white',
                         markeredgewidth=3, markeredgecolor='#32CD32')

        line2 = ax2_twin.plot(tubing_labels, pwf_ops_test, 's-',
                              color='#DC143C', linewidth=4, markersize=12,
                              label='Bottomhole Pressure', markerfacecolor='white',
                              markeredgewidth=3, markeredgecolor='#DC143C')

        ax2.set_xlabel("Tubing ID (inches)", fontsize=14, fontweight='bold', color='black')
        ax2.set_ylabel("Production Rate (bpd)", fontsize=14, fontweight='bold', color='#32CD32')
        ax2_twin.set_ylabel("Pwf (psia)", fontsize=14, fontweight='bold', color='#DC143C')
        ax2.set_title("Rate & Pressure Profiles", fontsize=16, fontweight='bold',
                      color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#32CD32', labelsize=12)
        ax2.tick_params(axis='x', colors='black', labelsize=12)
        ax2_twin.tick_params(axis='y', labelcolor='#DC143C', labelsize=12)
        ax2.grid(True, alpha=0.4, color='#CCCCCC')

        for spine in ax2.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2.5)
        for spine in ax2_twin.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2.5)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best', fontsize=11, framealpha=0.95, edgecolor='black')

        ax3 = fig2.add_subplot(gs[1, 1])
        ax3.set_facecolor('white')

        colors_eff = ['#DC143C' if e < 60 else '#FFA500' if e < 80 else '#32CD32' for e in efficiency_test]
        bars_eff = ax3.barh(tubing_labels, efficiency_test,
                            color=colors_eff,
                            edgecolor='black',
                            linewidth=2.5,
                            alpha=0.85)

        for bar, eff in zip(bars_eff, efficiency_test):
            width = bar.get_width()
            ax3.text(width + 1, bar.get_y() + bar.get_height() / 2.,
                     f'{eff:.1f}%',
                     ha='left', va='center',
                     fontsize=13, fontweight='bold', color='black')

        ax3.set_xlabel("Production Efficiency (%)", fontsize=14, fontweight='bold', color='black')
        ax3.set_ylabel("Tubing ID (inches)", fontsize=14, fontweight='bold', color='black')
        ax3.set_title("Production Efficiency (% of Max Rate)", fontsize=16, fontweight='bold',
                      color='#1f77b4')
        ax3.grid(True, alpha=0.4, axis='x', linestyle='--', color='#CCCCCC')
        ax3.set_xlim(0, 105)
        ax3.tick_params(colors='black', labelsize=12)
        for spine in ax3.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2.5)

        ax3.axvspan(0, 60, alpha=0.15, color='red', label='Low (<60%)')
        ax3.axvspan(60, 80, alpha=0.15, color='orange', label='Medium (60-80%)')
        ax3.axvspan(80, 105, alpha=0.15, color='green', label='High (>80%)')
        ax3.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='black')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("#### 📊 Detailed Performance Metrics")
        col1, col2 = st.columns([3, 2])

        with col1:
            if q_ops_test:
                df_tubing = pd.DataFrame({
                    "Tubing ID": tubing_labels,
                    "Rate (bpd)": q_ops_test,
                    "Pwf (psia)": pwf_ops_test,
                    "Drawdown (psi)": drawdown_test,
                    "Efficiency (%)": efficiency_test
                })

                st.dataframe(
                    df_tubing.style.background_gradient(cmap='RdYlGn', subset=['Efficiency (%)']),
                    use_container_width=True
                )

        with col2:
            if q_ops_test:
                best_idx = int(np.argmax(q_ops_test))
                worst_idx = int(np.argmin(q_ops_test))

                st.markdown("#### 🎯 Key Findings")
                st.success(f"**Best:** {tubing_labels[best_idx]} → {q_ops_test[best_idx]:.0f} bpd")
                st.error(f"**Worst:** {tubing_labels[worst_idx]} → {q_ops_test[worst_idx]:.0f} bpd")

                improvement = ((max(q_ops_test) - min(q_ops_test)) / max(min(q_ops_test), 1e-6)) * 100
                st.info(f"**Improvement:** {improvement:.1f}% increase from smallest to largest tubing")
    else:
        st.info("No operating points detected across tubing sizes to show detailed comparison.")

with tab3:
    st.markdown("### Gas-Oil Ratio (GOR) Sensitivity")

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    ax3.set_facecolor('white')
    fig3.patch.set_facecolor('white')

    gor_values = [200, 400, 600, 800, 1000]
    colors_gor = ['#32CD32', '#FFA500', '#FF8C00', '#FF6347', '#DC143C']

    for idx, gor_test in enumerate(gor_values):
        pwf_gor = []

        for q in q_range[::5]:
            pwf = calculate_vlp_beggs_brill(
                q, 3.5, P_wellhead, P_res, well_depth, gor_test,
                rho_oil, rho_gas, mu_oil_used, mu_gas_used, sigma, theta_deg
            )
            pwf_gor.append(pwf)

        ax3.plot(q_range[::5], pwf_gor,
                 color=colors_gor[idx],
                 linewidth=4,
                 label=f"GOR = {gor_test} scf/STB",
                 marker='s',
                 markersize=6,
                 alpha=0.9)

    ax3.plot(q_range, pwf_ipr, 'b--', linewidth=3.5, label='IPR', alpha=0.8)
    ax3.set_xlabel("Rate (bpd)", fontsize=14, fontweight='bold', color='black')
    ax3.set_ylabel("Pwf (psia)", fontsize=14, fontweight='bold', color='black')
    ax3.set_title("GOR Impact on VLP Curves", fontsize=16, fontweight='bold', color='#1f77b4')
    ax3.legend(fontsize=12, framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.4, linestyle='--', color='#CCCCCC')
    ax3.tick_params(colors='black', labelsize=12)
    for spine in ax3.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    st.pyplot(fig3)
    plt.close(fig3)

    st.info("💡 **Higher GOR** → More gas expansion → Lower mixture density → More pronounced U-shape in VLP curve")

# =====================================================================
# DATA EXPORT
# =====================================================================

st.markdown("---")
st.markdown("## 💾 Export Analysis Data")

col1, col2, col3 = st.columns(3)

with col1:
    ipr_export = pd.DataFrame({
        "Rate_bpd": q_range,
        "IPR_Pwf_psia": pwf_ipr
    })
    st.download_button(
        "📥 Download IPR Data",
        ipr_export.to_csv(index=False),
        "nodal_ipr_data.csv",
        "text/csv"
    )

with col2:
    vlp_export = pd.DataFrame({"Rate_bpd": q_range})
    for label, pwf in vlp_results.items():
        vlp_export[f"VLP_{label}_psia"] = pwf

    st.download_button(
        "📥 Download VLP Data",
        vlp_export.to_csv(index=False),
        "nodal_vlp_data.csv",
        "text/csv"
    )

with col3:
    if operating_points:
        op_export = pd.DataFrame(operating_points)
        st.download_button(
            "📥 Download Operating Points",
            op_export.to_csv(index=False),
            "operating_points.csv",
            "text/csv"
        )

# =====================================================================
# METHODOLOGY
# =====================================================================

with st.expander("📚 Methodology & Theory"):
    st.markdown("""
    ## Beggs-Brill Multiphase Flow Correlation

    ### VLP Curve Characteristic "U-Shape" Explained

    **Why does VLP start high, dip, then rise?**

    1. **At q=0 (Zero Flow):**
       - Maximum Pwf = Pwh + Full hydrostatic head
       - Dense liquid column (no gas)
       - ΔP = ρ_liquid × g × h

    2. **Low Flow Rates (Initial Drop):**
       - Gas liberation below bubble point
       - Gas expansion: ~100-300× volume increase
       - Mixture density ↓↓↓
       - Lighter column → Lower hydrostatic pressure
       - Friction still small
       - **Result: Pwf DECREASES**

    3. **High Flow Rates (Pressure Rise):**
       - Friction losses dominate: ΔP ∝ v²
       - Turbulent flow
       - **Result: Pwf INCREASES**

    ### Pressure Gradient Components:
    ```
    dP/dz = (dP/dz)_hydrostatic + (dP/dz)_friction

    Hydrostatic: ρ_mixture × g × cos(θ)
    Friction: 2·f_tp·ρ_m·v² / D
    ```

    ### IPR Methods:

    **Vogel:** q/qmax = 1 - 0.2(Pwf/Pr) - 0.8(Pwf/Pr)²  (we solve this quadratic for Pwf)

    **Fetkovich:** q = C(Pr^n - Pwf^n)

    ### Notes on viscosity estimators:
    - Simple empirical estimators were added as fallbacks if the user does not provide measured viscosities.
    - For accurate engineering design use measured viscosity data or more detailed correlations (e.g., Lohrenz-Brill-Jackson for gas).
    """)
    st.markdown("---")
