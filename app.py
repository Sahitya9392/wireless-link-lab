"""Wireless Link Lab: Monte Carlo bit error rate of BPSK over fading channels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import erfc
from scipy.stats import gamma as gamma_dist

try:  # apply the dark theme from code so no separate config file is needed
    for _k, _v in {"base": "dark", "primaryColor": "#4ea8ff", "backgroundColor": "#0a0f20",
                   "secondaryBackgroundColor": "#0e1428", "textColor": "#f1f4ff"}.items():
        st._config.set_option(f"theme.{_k}", _v)
except Exception:
    pass

st.set_page_config(page_title="Wireless Link Lab", page_icon="📡", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens (colour-blind-safe Okabe-Ito hues on a cool paper background)
# ---------------------------------------------------------------------------
BG, PANEL, SURFACE = "#0a0f20", "#0e1428", "#141c36"
INK, SOFT, MUTED = "#f1f4ff", "#cfd8f6", "#9fabd0"
RULE, GRID = "#2a3558", "#222c4a"
VERMILION, BLUE, GREEN = "#ff7a59", "#4ea8ff", "#34d399"      # coral, azure, mint
AMBER, PINK, VIOLET, SKY = "#fbbf24", "#f472b6", "#a78bfa", "#7dd3fc"
SECTION_COLORS = {
    "Channel comparison": AMBER, "Weibull BER surface": VIOLET, "Receive diversity with MRC": BLUE,
    "Multi-carrier transmission with OFDM": GREEN, "Spatial multiplexing with 2×2 MIMO": PINK,
    "Forward error correction with Hamming (7,4)": SKY, "Raw results": AMBER, "Fading envelope and PDF": SKY,
}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@85..100,500..800&family=Public+Sans:wght@400;500;600&display=swap');

:root, .stApp {{ color-scheme: dark; }}
.stApp {{ font-family: 'Public Sans', system-ui, sans-serif;
    background:
      radial-gradient(900px 520px at 6% -6%, rgba(78,168,255,0.20), transparent 62%),
      radial-gradient(760px 480px at 100% 2%, rgba(167,139,250,0.16), transparent 60%),
      radial-gradient(900px 600px at 55% 112%, rgba(52,211,153,0.10), transparent 60%),
      {BG} !important; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"], header[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{ padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1240px; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, #131c3a 0%, {PANEL} 55%, #0b1122 100%) !important; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {RULE}; }}

/* text colours */
[data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *, .stSpinner *, .stTabs button * {{ color: {SOFT} !important; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{ color: {INK} !important; font-weight: 500; }}
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: {SKY} !important; font-weight: 600; }}
header[data-testid="stHeader"] *, [data-testid="stToolbar"] * {{ color: {SOFT} !important; }}
.stTabs button[aria-selected="true"] * {{ color: {INK} !important; font-weight: 600; }}
.stTabs button[aria-selected="false"] * {{ color: {MUTED} !important; }}

.stApp .brand {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.3rem; letter-spacing: -0.01em;
    background: linear-gradient(95deg, #ffffff, {SKY}); -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important; }}
.stApp .brand-sub {{ color: {VIOLET} !important; font-size: 0.86rem; margin: 0.15rem 0 1.4rem 0; line-height: 1.45; }}
.stApp .side-group {{ font-family: 'Archivo', sans-serif; font-weight: 600; font-size: 1rem; color: {AMBER} !important;
    margin: 1.5rem 0 0.15rem 0; padding-top: 0.9rem; border-top: 1px solid {RULE}; }}
.stApp .side-note {{ color: {MUTED} !important; font-size: 0.84rem; line-height: 1.5; margin-bottom: 0.4rem; }}

.stApp .masthead h1 {{ font-family: 'Archivo', sans-serif !important; font-weight: 800 !important;
    font-size: clamp(2.1rem, 4.4vw, 3.4rem) !important; line-height: 1.08 !important;
    letter-spacing: -0.012em !important; margin: 0 0 1rem 0 !important; padding: 0 !important; max-width: 20ch;
    text-wrap: balance; background: linear-gradient(95deg, #ffffff 0%, #a9d2ff 45%, {VIOLET} 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent !important; }}
.stApp .masthead p {{ color: {SOFT} !important; font-size: 1.06rem; line-height: 1.6; max-width: 62ch; margin: 0 0 0.4rem 0; }}
.masthead svg {{ display: block; width: 100%; height: 110px; margin-top: 1rem; }}
.wave {{ stroke-dasharray: 1; stroke-dashoffset: 0; animation: draw 2.4s ease-out both; }}
@keyframes draw {{ from {{ stroke-dashoffset: 1; }} to {{ stroke-dashoffset: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .wave {{ animation: none; }} }}

.stApp .readout-head {{ color: {MUTED} !important; font-size: 0.92rem; margin: 1.6rem 0 0.7rem 0; }}
.readout {{ border-radius: 12px; padding: 0.95rem 1.05rem; height: 100%;
    background: linear-gradient(145deg, color-mix(in srgb, var(--c) 20%, transparent), color-mix(in srgb, var(--c) 4%, transparent) 75%);
    border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); border-top: 3px solid var(--c); }}
.stApp .r-label {{ font-weight: 600; font-size: 0.92rem; color: {INK} !important; }}
.stApp .r-value {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 2.2rem; line-height: 1.15;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--c) !important; }}
.stApp .r-note {{ color: {SOFT} !important; font-size: 0.86rem; line-height: 1.45; margin-top: 0.15rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {RULE}; }}
.stTabs button[data-baseweb="tab"] {{ padding: 0.6rem 0.9rem; font-weight: 500; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, {BLUE}, {VIOLET}) !important; height: 3px; }}

.stApp .sec {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.6rem; letter-spacing: -0.015em;
    margin: 1.4rem 0 0.4rem 0; padding-left: 0.75rem; border-left: 4px solid var(--c); color: var(--c) !important; }}
.stApp .lede {{ color: {SOFT} !important; max-width: 74ch; line-height: 1.65; font-size: 1rem; margin-bottom: 0.6rem; }}
.stApp .note {{ color: {SOFT} !important; max-width: 74ch; line-height: 1.65; margin-top: 0.7rem; }}
.stApp .note b {{ color: {AMBER} !important; font-weight: 600; }}
.stApp .footer {{ color: {MUTED} !important; font-size: 0.85rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid {RULE}; }}

[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] input {{
    background: {SURFACE} !important; color: {INK} !important; border-color: {RULE} !important; }}
[data-testid="stNumberInput"] button {{ background: {SURFACE} !important; }}
[data-testid="stNumberInput"] button * {{ color: {INK} !important; fill: {INK} !important; }}
div[data-baseweb="slider"] div[role="slider"] {{ background: {SKY} !important; box-shadow: 0 0 0 4px rgba(125,211,252,0.22); }}
[data-testid="stTooltipIcon"] svg {{ fill: {MUTED} !important; }}
.stDownloadButton button {{ background: linear-gradient(135deg, {BLUE}, {VIOLET}) !important; border: 0; border-radius: 8px; padding: 0.55rem 1.2rem; }}
.stDownloadButton button, .stDownloadButton button * {{ color: #07101f !important; font-weight: 600; }}
.stDownloadButton button:focus-visible {{ outline: 2px solid {SKY}; outline-offset: 2px; }}

.tbl-wrap {{ overflow-x: auto; border: 1px solid {RULE}; border-radius: 10px; background: {SURFACE}; margin-bottom: 1rem; }}
.tbl {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.9rem; }}
.tbl th {{ text-align: right; font-weight: 600; padding: 0.6rem 0.85rem; border-bottom: 1px solid {RULE};
    background: #1a2444; white-space: nowrap; color: {SKY}; }}
.tbl td {{ text-align: right; padding: 0.45rem 0.85rem; border-bottom: 1px solid {GRID}; color: {SOFT}; }}
.tbl tr:hover td {{ background: rgba(125,211,252,0.06); }}
.tbl th:first-child, .tbl td:first-child {{ text-align: left; font-weight: 600; color: {AMBER}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.markdown(
    '<div class="brand">Wireless Link Lab</div>'
    '<div class="brand-sub">BPSK bit error rate over fading channels</div>',
    unsafe_allow_html=True,
)

sb.markdown('<div class="side-group">Simulation</div>'
            '<div class="side-note">More bits give smoother curves and resolve lower error rates.</div>',
            unsafe_allow_html=True)
num_bits = sb.select_slider("Bits per SNR point", options=[10**3, 10**4, 10**5, 10**6],
                            value=10**5, format_func=lambda v: f"{v:,}")
max_snr = sb.slider("Maximum SNR (dB)", 10, 40, 30, step=2)
target_ber = sb.select_slider("Target BER", options=[1e-2, 1e-3, 1e-4], value=1e-3,
                              format_func=lambda v: f"1e{int(round(math.log10(v)))}")
seed = int(sb.number_input("Random seed", 0, 99_999, 7, help="Same seed, same curves."))

sb.markdown('<div class="side-group">Channel models</div>'
            '<div class="side-note">All models are normalised to unit average power, so '
            'differences come from the fading shape alone.</div>', unsafe_allow_html=True)
weibull_beta = sb.slider("Weibull shape (β)", 1.0, 4.0, 1.5, step=0.1,
                         help="β = 2 is exactly Rayleigh fading.")
gamma_k = sb.slider("Gamma shape (k)", 0.5, 6.0, 2.0, step=0.1,
                    help="The fading envelope is Gamma-distributed with shape k, scaled to unit power. "
                         "Smaller k means deeper, more frequent fades.")

snr_db = np.arange(0, max_snr + 1, 2)
snr_dense = np.linspace(0, max_snr, 160)

# ---------------------------------------------------------------------------
# Simulation core
#
# Every simulator is a "batch" function that returns (error counts per SNR
# point, bits simulated). `_monte_carlo` runs batches in chunks so memory stays
# flat even at 10^6 bits. Fading draws are shared across SNR points (common
# random numbers), which keeps the curves smooth and monotone.
# ---------------------------------------------------------------------------
CHUNK = 50_000


def _lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


F32 = np.float32


def _sigma(snr):
    """Noise standard deviation per real dimension for unit symbol energy."""
    return np.sqrt(0.5 / snr).astype(F32)


def _real_noise(rng, snr, n):
    return _sigma(snr)[:, None] * rng.standard_normal((snr.size, n), dtype=F32)


def _cplx_noise(rng, snr, *shape):
    scale = _sigma(snr).reshape((-1,) + (1,) * len(shape))
    full = (snr.size,) + shape
    return scale * (rng.standard_normal(full, dtype=F32) + np.complex64(1j) * rng.standard_normal(full, dtype=F32))


def _cn(rng, *shape):
    """Unit-variance circular complex Gaussian."""
    return (rng.standard_normal(shape, dtype=F32) + np.complex64(1j) * rng.standard_normal(shape, dtype=F32)) * F32(0.70710678)


def _monte_carlo(batch, n_bits, snr, seed, **kw):
    rng = np.random.default_rng(seed)
    errors, total, left = 0, 0, n_bits
    while left > 0:
        n = min(CHUNK, left)
        e, t = batch(rng, n, snr, **kw)
        errors, total, left = errors + e, total + t, left - n
    return errors / total


# Fading amplitude generators, all with E[h^2] = 1
def _rayleigh(rng, n, _=None):
    return np.hypot(rng.standard_normal(n, dtype=F32), rng.standard_normal(n, dtype=F32)) * F32(0.70710678)


def _weibull(rng, n, beta):
    return rng.weibull(beta, n) / math.sqrt(math.gamma(1 + 2 / beta))


def _gamma(rng, n, k):
    # Gamma-distributed envelope; scale chosen so E[h^2] = k(k+1)θ² = 1
    return rng.gamma(k, 1 / math.sqrt(k * (k + 1)), n)


def _gaussian(rng, n, _=None):
    # Line-of-sight model: envelope |1 + N(0,1)|, scaled by 1/sqrt(2) so that E[h^2] = 1
    return np.abs(1 + rng.standard_normal(n)) / math.sqrt(2)


CHANNELS = {"rayleigh": _rayleigh, "weibull": _weibull, "gamma": _gamma, "gaussian": _gaussian}


def _flat_batch(rng, n, snr, draw, param):
    # BPSK is symmetric, so sending +1 every time gives the same BER as random bits.
    y = draw(rng, n, param) + _real_noise(rng, snr, n)
    return (y < 0).sum(axis=1), n


def _mrc_batch(rng, n, snr, branches):
    h = _cn(rng, branches, n)
    y = h + _cplx_noise(rng, snr, branches, n)          # +1 sent on every branch
    stat = np.real((np.conj(h) * y).sum(axis=1))        # maximal ratio combiner
    return (stat < 0).sum(axis=1), n


N_SC, CP_LEN, N_TAPS = 64, 16, 4
PDP = np.exp(-np.arange(N_TAPS) / 1.5)
PDP /= PDP.sum()                                         # exponential power delay profile


def _ofdm_batch(rng, n, snr):
    n_sym = max(1, n // N_SC)
    s = rng.choice([-1.0, 1.0], size=(n_sym, N_SC))
    x = np.fft.ifft(s, axis=1) * np.sqrt(N_SC)           # unit power per sample
    x_cp = np.concatenate([x[:, -CP_LEN:], x], axis=1)
    taps = np.sqrt(PDP) * _cn(rng, n_sym, N_TAPS)        # one multipath channel per OFDM symbol
    rx = np.zeros_like(x_cp)
    for lag in range(N_TAPS):
        rx[:, lag:] += taps[:, lag:lag + 1] * x_cp[:, : x_cp.shape[1] - lag]
    y = rx + _cplx_noise(rng, snr, *x_cp.shape)
    y = np.fft.fft(y[..., CP_LEN:], axis=-1) / np.sqrt(N_SC)
    h_freq = np.fft.fft(taps, N_SC, axis=1)
    stat = np.real(y * np.conj(h_freq)) * s              # one-tap equaliser per subcarrier
    return (stat < 0).sum(axis=(1, 2)), n_sym * N_SC


def _mimo_zf_batch(rng, n, snr):
    pairs = max(1, n // 2)
    s = rng.choice([-1.0, 1.0], size=(2, pairs))
    h = _cn(rng, 2, 2, pairs)
    # Total transmit power is fixed at 1, so each antenna sends s / sqrt(2).
    y = np.einsum("abn,bn->an", h, s) / np.sqrt(2) + _cplx_noise(rng, snr, 2, pairs)
    det = h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]
    w = np.array([[h[1, 1], -h[0, 1]], [-h[1, 0], h[0, 0]]]) / det
    x_hat = np.sqrt(2) * np.einsum("abn,kbn->kan", w, y)
    return (np.real(x_hat) * s < 0).sum(axis=(1, 2)), 2 * pairs


G = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1],
              [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
H_PARITY = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
DATA_COLS = [2, 4, 5, 6]
H_PARITY_T = H_PARITY.T.astype(np.int8)
WEIGHTS = np.array([1, 2, 4], dtype=np.int8)
G_T = G.T.astype(np.int8)
RATE = 4 / 7


def _hamming_batch(rng, n, snr):
    blocks = max(1, n // 4)
    msg = rng.integers(0, 2, (blocks, 4), dtype=np.int8)
    tx = 2.0 * ((msg @ G_T) % 2) - 1.0
    h = _rayleigh(rng, blocks * 7).reshape(blocks, 7)    # ideal interleaving: independent fades
    # Equal energy per information bit: each coded symbol carries RATE of the energy.
    rx = h * tx + _sigma(snr * RATE)[:, None, None] * rng.standard_normal((snr.size, blocks, 7), dtype=F32)
    bits = (rx > 0).astype(np.int8)
    pos = ((bits @ H_PARITY_T) % 2) @ WEIGHTS   # syndrome value = 1-based error position
    k_idx, b_idx = np.nonzero(pos)
    bits[k_idx, b_idx, pos[k_idx, b_idx] - 1] ^= 1
    err_coded = (bits[..., DATA_COLS] != msg).sum(axis=(1, 2))
    y_unc = _rayleigh(rng, blocks * 4) + _real_noise(rng, snr, blocks * 4)
    return np.stack([(y_unc < 0).sum(axis=1), err_coded]), blocks * 4


@st.cache_data(show_spinner=False)
def sim_flat(kind, param, n_bits, snrs_db, seed):
    return _monte_carlo(_flat_batch, n_bits, _lin(snrs_db), seed, draw=CHANNELS[kind], param=param)


@st.cache_data(show_spinner=False)
def sim_mrc(n_bits, snrs_db, seed, branches=2):
    return _monte_carlo(_mrc_batch, n_bits, _lin(snrs_db), seed + 1, branches=branches)


@st.cache_data(show_spinner=False)
def sim_ofdm(n_bits, snrs_db, seed):
    return _monte_carlo(_ofdm_batch, n_bits, _lin(snrs_db), seed + 2)


@st.cache_data(show_spinner=False)
def sim_mimo(n_bits, snrs_db, seed):
    return _monte_carlo(_mimo_zf_batch, n_bits, _lin(snrs_db), seed + 3)


@st.cache_data(show_spinner=False)
def sim_hamming(n_bits, snrs_db, seed):
    uncoded, coded = _monte_carlo(_hamming_batch, n_bits, _lin(snrs_db), seed + 4)
    return uncoded, coded


@st.cache_data(show_spinner=False)
def sim_surface(n_bits, snrs_db, seed):
    betas = np.linspace(1.0, 4.0, 13)
    z = np.vstack([sim_flat("weibull", float(b), n_bits, snrs_db, seed) for b in betas])
    return betas, z


def rayleigh_theory(snr):
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


def awgn_theory(snr):
    return 0.5 * erfc(np.sqrt(snr))


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def snr_at_target(ber, target):
    """SNR (dB) where the curve first reaches `target`, interpolated in log-BER."""
    for i, b1 in enumerate(ber):
        if b1 <= target:
            if i == 0 or b1 <= 0:
                return float(snr_db[i])
            b0 = ber[i - 1]
            frac = (math.log10(b0) - math.log10(target)) / (math.log10(b0) - math.log10(b1))
            return float(snr_db[i - 1] + frac * (snr_db[i] - snr_db[i - 1]))
    return None


def fmt_db(v):
    return f"{v:.1f} dB" if v is not None else f"> {int(snr_db[-1])} dB"


def compare(base, other, vs):
    if base is not None and other is not None:
        d = base - other
        if abs(d) < 0.05:
            return f"Same as {vs}"
        return f"{abs(d):.1f} dB {'lower' if d > 0 else 'higher'} than {vs}"
    if other is not None:
        return f"Reaches the target; {vs} does not"
    if base is not None:
        return f"Misses the target; {vs} needs {fmt_db(base)}"
    return "Target not reached in this range"


def readout(label, value, note, color):
    return (f'<div class="readout" style="--c:{color}"><div class="r-label">{label}</div>'
            f'<div class="r-value">{value}</div><div class="r-note">{note}</div></div>')


def section(title, text):
    c = SECTION_COLORS.get(title, SKY)
    st.markdown(f'<h3 class="sec" style="--c:{c}">{title}</h3><p class="lede">{text}</p>', unsafe_allow_html=True)


def note(html):
    st.markdown(f'<p class="note">{html}</p>', unsafe_allow_html=True)


Y_FLOOR = -math.log10(num_bits) - 0.4
TARGET_LABEL = f"Target BER (1e{int(round(math.log10(target_ber)))})"


def ber_chart(series, title, reference=()):
    """series: (label, ber, colour, marker). reference: (label, theory_fn, dash)."""
    fig = go.Figure()
    for label, fn, dash in reference:
        fig.add_trace(go.Scatter(x=snr_dense, y=fn(_lin(snr_dense)), name=label, mode="lines",
                                 line=dict(color=MUTED, width=1.3, dash=dash), hoverinfo="skip"))
    for label, ber, color, symbol in series:
        y = np.where(np.asarray(ber) <= 0, np.nan, ber)       # zero errors cannot be shown on a log axis
        fig.add_trace(go.Scatter(
            x=snr_db, y=y, name=label, mode="lines+markers",
            line=dict(color=color, width=2.4),
            marker=dict(symbol=symbol, size=7, color=color, line=dict(width=1, color=SURFACE)),
            hovertemplate="%{y:.2e}<extra>" + label + "</extra>"))
    fig.add_hline(y=target_ber, line=dict(color=INK, width=1, dash="dot"),
                  annotation_text=TARGET_LABEL, annotation_position="top left",
                  annotation_font=dict(size=11, color=MUTED))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Archivo", size=17, color=INK), x=0),
        height=520, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family="Public Sans", color=MUTED, size=13),
        margin=dict(l=10, r=10, t=60, b=90),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font=dict(color=INK)),
        xaxis=dict(title=dict(text="SNR (dB)", standoff=10), automargin=True, gridcolor=GRID, zeroline=False, linecolor=RULE, range=[0, snr_db[-1]]),
        yaxis=dict(title=dict(text="Bit error rate", standoff=12), automargin=True, type="log", range=[Y_FLOOR, 0], gridcolor=GRID,
                   zeroline=False, linecolor=RULE, exponentformat="power"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(color=INK)))
    return fig


def show(fig):
    cfg = {"displaylogo": False}
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=cfg)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg)


def masthead_wave():
    xs = np.linspace(0, 1200, 360)
    carrier = np.sin(xs / 16)
    env = 0.25 + 0.75 * np.abs(np.sin(xs / 190) + 0.5 * np.sin(xs / 61 + 1.0)) / 1.5
    path = lambda ys: "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return ('<svg viewBox="0 0 1200 110" preserveAspectRatio="none" aria-hidden="true">'
            f'<defs><linearGradient id="wg" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{BLUE}"/>'
            f'<stop offset="0.55" stop-color="{VIOLET}"/><stop offset="1" stop-color="{VERMILION}"/></linearGradient></defs>'
            f'<path d="{path(55 - 38 * carrier)}" fill="none" stroke="{RULE}" stroke-width="1.4"/>'
            f'<path class="wave" pathLength="1" d="{path(55 - 48 * carrier * env)}" fill="none" '
            f'stroke="url(#wg)" stroke-width="2.4" stroke-linecap="round"/></svg>')


# ---------------------------------------------------------------------------
# Run simulations (each is cached, so a slider only recomputes what it affects)
# ---------------------------------------------------------------------------
with st.spinner("Running simulations"):
    ber_ray = sim_flat("rayleigh", 0.0, num_bits, snr_db, seed)
    ber_wei = sim_flat("weibull", float(weibull_beta), num_bits, snr_db, seed)
    ber_gam = sim_flat("gamma", float(gamma_k), num_bits, snr_db, seed)
    ber_gau = sim_flat("gaussian", 0.0, num_bits, snr_db, seed)
    ber_mrc = sim_mrc(num_bits, snr_db, seed)
    ber_ofdm = sim_ofdm(num_bits, snr_db, seed)
    ber_zf = sim_mimo(num_bits, snr_db, seed)
    ber_unc, ber_cod = sim_hamming(num_bits, snr_db, seed)

t_ray, t_mrc, t_zf = (snr_at_target(b, target_ber) for b in (ber_ray, ber_mrc, ber_zf))
t_unc, t_cod = snr_at_target(ber_unc, target_ber), snr_at_target(ber_cod, target_ber)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>How fading degrades a radio link, and what helps</h1>'
    '<p>A Monte Carlo study of BPSK over Gamma, Rayleigh, Weibull and Gaussian fading, followed by receive diversity, '
    'OFDM, spatial multiplexing and a Hamming code. Each point averages '
    f'{num_bits:,} bits.</p>{masthead_wave()}</div>',
    unsafe_allow_html=True,
)

st.markdown(f'<div class="readout-head">SNR needed to reach a bit error rate of {target_ber:.0e}</div>',
            unsafe_allow_html=True)
cols = st.columns(4, gap="large")
cols[0].markdown(readout("Rayleigh, one antenna", fmt_db(t_ray), "Baseline", VERMILION), unsafe_allow_html=True)
cols[1].markdown(readout("Two-antenna MRC", fmt_db(t_mrc), compare(t_ray, t_mrc, "one antenna"), BLUE),
                 unsafe_allow_html=True)
cols[2].markdown(readout("2×2 MIMO, zero-forcing", fmt_db(t_zf),
                         compare(t_ray, t_zf, "one antenna") + ", at twice the data rate", PINK),
                 unsafe_allow_html=True)
cols[3].markdown(readout("Hamming (7,4) coded", fmt_db(t_cod), compare(t_unc, t_cod, "uncoded"), GREEN),
                 unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["Fading channels", "Envelope & PDF", "Weibull surface", "Diversity (MRC)", "OFDM", "MIMO 2×2",
                "Hamming coding", "Data"])

with tabs[0]:
    section("Channel comparison",
            "Bit error rate against SNR for Gamma, Rayleigh, Weibull and Gaussian fading, all with the same average power. "
            "Rayleigh has no line of sight; the Gaussian model has a strong direct path. The Gamma shape k and "
            "the Weibull shape β set how severe the fading is, and both are adjustable from the sidebar.")
    show(ber_chart(
        [(f"Gamma (k = {gamma_k})", ber_gam, GREEN, "diamond"),
         ("Rayleigh", ber_ray, VERMILION, "square"),
         (f"Weibull (β = {weibull_beta})", ber_wei, AMBER, "triangle-up"),
         ("Gaussian (line of sight)", ber_gau, BLUE, "circle")],
        "BER vs. SNR across fading models",
        reference=[("Rayleigh theory", lambda s: rayleigh_theory(s), "dot"),
                   ("No fading (AWGN)", lambda s: awgn_theory(s), "dash")]))
    note("Rayleigh drops only about tenfold for every 10 dB, because a deep fade can erase a bit however strong "
         "the average signal is. A larger Gamma k or Weibull β means milder fading and a steeper curve; "
         "a smaller value means deeper fades and a flatter one. Weibull with β = 2 is exactly Rayleigh. The Gaussian model's direct path keeps its curve the steepest. "
         "The simulated Rayleigh points should sit on the dotted theory line.")

with tabs[1]:
    section("Fading envelope and PDF",
            "Left: sample envelope values in dB for each channel (independent draws, so this shows depth of fades, "
            "not Doppler timing). Right: the envelope probability density. All four have unit average power.")
    rng_w = np.random.default_rng(seed + 9)
    n_w = 300
    envs = [(f"Gamma (k = {gamma_k})", _gamma(rng_w, n_w, gamma_k), GREEN),
            ("Rayleigh", _rayleigh(rng_w, n_w), VERMILION),
            (f"Weibull (β = {weibull_beta})", _weibull(rng_w, n_w, weibull_beta), AMBER),
            ("Gaussian (line of sight)", _gaussian(rng_w, n_w), BLUE)]
    r = np.linspace(0.004, 3.0, 500)
    lam = 1 / math.sqrt(math.gamma(1 + 2 / weibull_beta))
    theta = 1 / math.sqrt(gamma_k * (gamma_k + 1))
    pdfs = [gamma_dist.pdf(r, gamma_k, scale=theta), 2 * r * np.exp(-r ** 2),
            (weibull_beta / lam) * (r / lam) ** (weibull_beta - 1) * np.exp(-((r / lam) ** weibull_beta)),
            (np.exp(-(r - 0.5 ** 0.5) ** 2) + np.exp(-(r + 0.5 ** 0.5) ** 2)) / math.sqrt(math.pi * 0.5) * math.sqrt(0.5)]
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=RULE, automargin=True)

    def styled(fig, title, xt, yt, **yextra):
        fig.update_layout(
            title=dict(text=title, font=dict(family="Archivo", size=16, color=INK), x=0), height=430,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE, hovermode="x unified",
            font=dict(family="Public Sans", color=MUTED, size=13), margin=dict(l=10, r=10, t=55, b=80),
            legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font=dict(color=INK)),
            xaxis=dict(title=dict(text=xt, standoff=10), **axis),
            yaxis=dict(title=dict(text=yt, standoff=12), **axis, **yextra),
            hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(color=INK)))
        return fig

    c1, c2 = st.columns(2, gap="large")
    fig_w = go.Figure([go.Scatter(x=np.arange(n_w), y=20 * np.log10(np.maximum(h, 1e-3)), name=lab, mode="lines",
                                  line=dict(color=col, width=1.4), opacity=0.9) for lab, h, col in envs])
    fig_p = go.Figure([go.Scatter(x=r, y=p, name=lab, mode="lines", line=dict(color=col, width=2.4))
                       for (lab, _, col), p in zip(envs, pdfs)])
    with c1:
        show(styled(fig_w, "Envelope samples", "Sample index", "Envelope (dB)", range=[-30, 10]))
    with c2:
        show(styled(fig_p, "Envelope PDF", "Envelope amplitude", "Probability density", range=[0, 2.5]))
    note("Deep fades are the dips far below 0 dB. Rayleigh and a Gamma with small k dip furthest; "
         "raising k or β pulls the envelope toward a steady level, and the Gaussian line-of-sight model stays the most stable.")

with tabs[2]:
    section("Weibull BER surface",
            "How log bit error rate changes with SNR and the Weibull shape β. At β = 2 the channel is "
            "Rayleigh; larger β means milder fading. Drag to rotate, scroll to zoom.")
    surf_bits = min(num_bits, 10**5)
    betas, z = sim_surface(surf_bits, snr_db, seed)
    z_log = np.log10(np.clip(z, 10 ** (-math.log10(surf_bits) - 0.4), 1))
    ax = dict(gridcolor=GRID, backgroundcolor=SURFACE, color=MUTED, showbackground=True)
    fig3d = go.Figure(go.Surface(
        z=z_log, x=snr_db, y=betas, colorscale="Turbo",
        colorbar=dict(title=dict(text="log₁₀ BER", font=dict(color=MUTED)), tickfont=dict(color=MUTED),
                      outlinewidth=0, thickness=14)))
    fig3d.update_layout(
        height=640, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Public Sans", color=MUTED),
        scene=dict(xaxis=dict(title="SNR (dB)", **ax), yaxis=dict(title="Weibull shape (β)", **ax),
                   zaxis=dict(title="log₁₀ BER", **ax), camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
        margin=dict(l=0, r=0, t=10, b=0))
    show(fig3d)

with tabs[3]:
    section("Receive diversity with MRC",
            "A receiver with two antennas weights each branch by its channel gain and adds them "
            "(maximal ratio combining). A deep fade on one antenna rarely hits the other at the same moment.")
    show(ber_chart([("One antenna, Rayleigh", ber_ray, VERMILION, "square"),
                    ("Two antennas, MRC", ber_mrc, BLUE, "star")],
                   "Receive diversity: 1×1 vs. 1×2"))
    note(f"MRC needs <b>{fmt_db(t_mrc)}</b> to reach {target_ber:.0e}, against <b>{fmt_db(t_ray)}</b> "
         "for a single antenna. The curve is also steeper: two branches give diversity order 2.")

with tabs[4]:
    section("Multi-carrier transmission with OFDM",
            f"{N_SC} subcarriers, a {CP_LEN}-sample cyclic prefix and a {N_TAPS}-tap multipath Rayleigh channel, "
            "with a one-tap equaliser per subcarrier. The cyclic prefix turns the multipath into a flat channel "
            "on each subcarrier.")
    show(ber_chart([("Single carrier, flat Rayleigh", ber_ray, VERMILION, "square"),
                    ("OFDM, multipath Rayleigh", ber_ofdm, GREEN, "x")],
                   "OFDM vs. single carrier",
                   reference=[("Rayleigh theory", lambda s: rayleigh_theory(s), "dot")]))
    note("Uncoded OFDM matches flat-Rayleigh BER. Its advantage is not a lower error rate but that it survives "
         "multipath with trivial equalisation. Add coding across subcarriers to also harvest frequency diversity.")

with tabs[5]:
    section("Spatial multiplexing with 2×2 MIMO",
            "Two antennas send two independent streams at once. A zero-forcing receiver inverts the channel "
            "matrix to separate them. Total transmit power is held at one antenna's worth, so the comparison is fair.")
    show(ber_chart([("1×1 SISO", ber_ray, VERMILION, "square"),
                    ("2×2 MIMO, zero-forcing", ber_zf, PINK, "cross")],
                   "2×2 MIMO with zero-forcing"))
    note(f"Zero-forcing needs <b>{fmt_db(t_zf)}</b> for {target_ber:.0e}, against <b>{fmt_db(t_ray)}</b> for SISO. "
         "That extra SNR buys twice the data rate. Inverting the channel amplifies noise, and a 2×2 zero-forcing "
         "receiver has no diversity gain, so the curve is no steeper than SISO.")

with tabs[6]:
    section("Forward error correction with Hamming (7,4)",
            "Each 4-bit message becomes a 7-bit codeword. The receiver computes a syndrome from the "
            "parity-check matrix and corrects any single-bit error in a block. The x-axis is energy per "
            "information bit, so the rate loss is paid for.")
    show(ber_chart([("Uncoded, Rayleigh", ber_unc, VERMILION, "square"),
                    ("Hamming (7,4), hard decision", ber_cod, GREEN, "hexagon")],
                   "Coding gain over Rayleigh fading"))
    note(f"Coding reaches {target_ber:.0e} at <b>{fmt_db(t_cod)}</b> versus <b>{fmt_db(t_unc)}</b> uncoded. "
         "Fades are assumed independent per coded bit (ideal interleaving). At low SNR the code loses, "
         "because mis-corrected blocks add errors.")

with tabs[7]:
    section("Raw results", "Every simulated BER value in one table. Download it as CSV for your own analysis.")
    df = pd.DataFrame({"SNR (dB)": snr_db, "Rayleigh": ber_ray, "Gamma": ber_gam, "Weibull": ber_wei, "Gaussian": ber_gau,
                       "MRC 1x2": ber_mrc, "OFDM": ber_ofdm, "MIMO 2x2 ZF": ber_zf,
                       "Hamming uncoded": ber_unc, "Hamming coded": ber_cod})
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join(
        "<tr><td>" + f"{r['SNR (dB)']:.0f}" + "</td>"
        + "".join(f"<td>{r[c]:.2e}</td>" for c in df.columns[1:]) + "</tr>"
        for _, r in df.iterrows())
    st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
                f'<tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="fading_simulation_data.csv", mime="text/csv")

st.markdown(f'<div class="footer">BPSK with coherent detection and perfect channel knowledge. '
            f'Seed {seed}. Zero-error points are not drawn on log axes.</div>', unsafe_allow_html=True)