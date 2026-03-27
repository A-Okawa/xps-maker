import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os

# ===== ページ設定 =====
st.set_page_config(page_title="XPS Analyzer", page_icon="⚡", layout="wide")

# ===== パスワード認証 =====
def check_password():
    if st.session_state.get("authenticated"):
        return True
    pwd = st.secrets.get("password", "")
    st.title("⚡ XPS Analyzer")
    entered = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    return False

if not check_password():
    st.stop()

st.markdown("""
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif !important; }
</style>
""", unsafe_allow_html=True)

st.title("XPS Analyzer")
st.caption("複数スペクトル重ね合わせ・バックグラウンド・ピーク成分・論文用TIFF出力")

# ===== 24色パレット（6系統 × 4色） =====
PALETTE_24 = OrderedDict([
    ("赤・ピンク系", [
        ("#b71c1c", "Dark Red"),    ("#e53935", "Bright Red"),
        ("#ef9a9a", "Light Red"),   ("#f06292", "Pink"),
    ]),
    ("橙・黄系", [
        ("#e64a19", "Dark Orange"), ("#ff7043", "Orange"),
        ("#ffa726", "Amber"),       ("#ffca28", "Yellow"),
    ]),
    ("緑系", [
        ("#2e7d32", "Dark Green"),  ("#43a047", "Green"),
        ("#00897b", "Teal"),        ("#26c6da", "Cyan"),
    ]),
    ("青系", [
        ("#1565c0", "Dark Blue"),   ("#1976d2", "Blue"),
        ("#1e88e5", "Medium Blue"), ("#42a5f5", "Sky Blue"),
    ]),
    ("紫系", [
        ("#6a1b9a", "Dark Purple"), ("#8e24aa", "Purple"),
        ("#ba68c8", "Violet"),      ("#ce93d8", "Lavender"),
    ]),
    ("茶・グレー系", [
        ("#5d4037", "Brown"),       ("#8d6e63", "Medium Brown"),
        ("#616161", "Dark Gray"),   ("#000000", "Black"),
    ]),
])
ALL_COLORS = [h for fam in PALETTE_24.values() for h, _ in fam]

COMPONENT_COLORS = [
    "#e53935", "#1e88e5", "#43a047", "#ffa726",
    "#8e24aa", "#00897b", "#f06292", "#5d4037",
]

# ===== ラベル入力（書式ボタン付き） =====
def label_input(key: str, default: str = "") -> str:
    val_key = f"_val_{key}"
    ver_key = f"_ver_{key}"
    if val_key not in st.session_state:
        st.session_state[val_key] = default
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0
    inp_key = f"_inp_{key}_v{st.session_state[ver_key]}"
    inp = st.text_input("ラベル", value=st.session_state[val_key], key=inp_key)
    st.session_state[val_key] = inp
    c1, c2, c3 = st.columns(3)
    if c1.button("＋italic", key=f"_bi_{key}", width='stretch'):
        st.session_state[val_key] = inp + r"$\it{TEXT}$"
        st.session_state[ver_key] += 1
        st.rerun()
    if c2.button("＋下付き", key=f"_bs_{key}", width='stretch'):
        st.session_state[val_key] = inp + r"$_{N}$"
        st.session_state[ver_key] += 1
        st.rerun()
    if c3.button("＋上付き", key=f"_bp_{key}", width='stretch'):
        st.session_state[val_key] = inp + r"$^{N}$"
        st.session_state[ver_key] += 1
        st.rerun()
    return st.session_state[val_key]

# ===== カラーポップアップ =====
def color_picker_popover(key: str, default_hex: str):
    if key not in st.session_state:
        st.session_state[key] = default_hex
    current = st.session_state[key]
    st.markdown(
        f'<div style="background:{current};height:28px;border-radius:5px;'
        f'border:1px solid #ccc;margin-bottom:4px"></div>',
        unsafe_allow_html=True,
    )
    with st.popover("🎨 色を変更", width='stretch'):
        for family, colors in PALETTE_24.items():
            st.caption(family)
            cols = st.columns(len(colors))
            for j, (hex_c, name) in enumerate(colors):
                with cols[j]:
                    selected = (hex_c == current)
                    st.markdown(
                        f'<div style="background:{hex_c};height:24px;border-radius:3px;'
                        f'border:{"3px solid #333" if selected else "1px solid #ccc"};'
                        f'margin-bottom:2px"></div>',
                        unsafe_allow_html=True,
                    )
                    def _cb(k=key, h=hex_c):
                        st.session_state[k] = h
                    st.button("✓" if selected else " ",
                             key=f"{key}_{family}_{j}", help=name,
                             width='stretch',
                             on_click=_cb)
    return st.session_state[key]

# ===== CSV 読み込み =====
def read_xps_csv(file_bytes: bytes):
    try:
        content = file_bytes.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content))
        if len(df.columns) <= 2:
            try:
                df_num = df.apply(pd.to_numeric, errors="coerce")
                if df_num.notna().all().all():
                    return {
                        "energy": df_num.iloc[:, 0].values,
                        "spectrum": df_num.iloc[:, 1].values,
                        "background": None,
                        "components": [],
                        "component_names": [],
                    }
            except Exception:
                pass
        energy_col = None
        for col in df.columns:
            if "energy" in col.lower() or "binding" in col.lower():
                energy_col = col
                break
        if energy_col is None:
            energy_col = df.columns[0]
        result = {
            "energy": pd.to_numeric(df[energy_col], errors="coerce").values,
            "spectrum": None,
            "background": None,
            "components": [],
            "component_names": [],
        }
        for col in df.columns:
            if col == energy_col:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").values
            col_lower = col.lower()
            if "spectrum" in col_lower or col_lower == "intensity":
                result["spectrum"] = vals
            elif "background" in col_lower or "bg" in col_lower or "backgr" in col_lower:
                result["background"] = vals
            elif col.startswith("[") or "component" in col_lower or "peak" in col_lower:
                result["components"].append(vals)
                result["component_names"].append(col)
        if result["spectrum"] is None:
            for col in df.columns:
                if col != energy_col:
                    result["spectrum"] = pd.to_numeric(df[col], errors="coerce").values
                    break
        return result
    except Exception as e:
        st.warning(f"CSV 読み込みエラー: {e}")
        return None

# ===== サイドバー =====
st.sidebar.header("📂 XPS データ (CSV)")
xps_files = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=["csv"], accept_multiple_files=True,
)

# ファイルから auto xlim と成分数を事前読み込み
_comp_counts = {}
if xps_files:
    _all_e = []
    for _idx, _f in enumerate(xps_files):
        try:
            _d = read_xps_csv(_f.read())
            _f.seek(0)
            if _d:
                if _d["energy"] is not None:
                    _all_e.extend(_d["energy"].tolist())
                _comp_counts[_idx] = len(_d["components"])
        except Exception:
            _comp_counts[_idx] = 0
    if _all_e:
        st.session_state["auto_xlim"] = (float(min(_all_e)), float(max(_all_e)))

_auto_lo, _auto_hi = st.session_state.get("auto_xlim", (0.0, 1200.0))

st.sidebar.header("⚙️ グラフ設定")
reverse_x     = st.sidebar.checkbox("X軸を反転（Binding Energy）", value=True)
show_background = st.sidebar.checkbox("バックグラウンドを表示", value=True)
show_components = st.sidebar.checkbox("ピーク成分を表示", value=True)
fill_components = st.sidebar.checkbox("ピーク成分を塗りつぶす", value=True) if show_components else False
normalize     = st.sidebar.checkbox("強度を正規化（最大=1）", value=False)
show_legend   = st.sidebar.checkbox("凡例を表示", value=True)
global_offset = st.sidebar.slider("オフセット間隔（倍率）", 0.0, 3.0, 1.0, step=0.05) if normalize else None

x_range_auto = st.sidebar.checkbox("X軸範囲を自動設定", value=True)
if not x_range_auto:
    x_min_xps = st.sidebar.number_input("X軸 最小 (eV)", value=_auto_lo)
    x_max_xps = st.sidebar.number_input("X軸 最大 (eV)", value=_auto_hi)
else:
    x_min_xps = None
    x_max_xps = None

st.sidebar.subheader("目盛り設定")
major_tick = st.sidebar.number_input("主目盛り間隔 (eV)", min_value=1.0, max_value=200.0, value=5.0, step=1.0)
show_minor = st.sidebar.checkbox("副目盛りを表示", value=True)
minor_tick = st.sidebar.number_input("副目盛り間隔 (eV)", min_value=0.5, max_value=50.0, value=1.0, step=0.5) if show_minor else None

st.sidebar.subheader("サンプルラベル（グラフ内）")
show_side_labels = st.sidebar.checkbox("グラフ内にラベルを表示", value=False)
if show_side_labels:
    label_side     = st.sidebar.radio("ラベル位置", ["左", "右"], horizontal=True)
    label_fontsize = st.sidebar.slider("ラベル文字サイズ", 5, 24, 11)
    label_offset_x = st.sidebar.slider("横オフセット (eV)", -5.0, 5.0, 0.5, step=0.1)
    label_offset_y = st.sidebar.slider("縦オフセット（パターン高さ比）", -0.3, 1.0, 0.05, step=0.01)
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = "右", 11, 0.5, 0.05

st.sidebar.subheader("バックグラウンド・成分の表示")
bg_color  = st.sidebar.color_picker("バックグラウンドの色", value="#808080")
bg_alpha  = st.sidebar.slider("バックグラウンド透明度", 0.1, 1.0, 0.5, step=0.05)
comp_alpha = st.sidebar.slider("ピーク成分透明度", 0.1, 1.0, 0.6, step=0.05) if show_components else 0.6

st.sidebar.header("📐 図サイズ・出力")
fig_width  = st.sidebar.slider("図の幅 (inch)", 4.0, 20.0, 8.0, step=0.5)
fig_height = st.sidebar.slider("図の高さ (inch)", 3.0, 20.0, 6.0, step=0.5)
dpi_export = st.sidebar.selectbox("出力 DPI", [300, 600], index=0)
font_size  = st.sidebar.slider("フォントサイズ", 8, 20, 14)

# ===== メインエリア =====
if xps_files:
    show_panel = st.toggle("⚙️ パターン設定パネルを表示", value=True)
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph = st.container()
        col_settings = None

    orders, visibles, labels, colors_sel = [], [], [], []
    sort_idx = []

    if show_panel and col_settings is not None:
        with col_settings:
            with st.container(height=700):
                st.markdown("#### XPS スペクトル")
                for i, f in enumerate(xps_files):
                    default_name = os.path.splitext(f.name)[0]
                    default_hex  = "#000000"
                    with st.expander(f"**{i+1}. {default_name}**", expanded=True):
                        order   = st.number_input("表示順", value=i+1, min_value=1, max_value=50, key=f"ord_{i}")
                        visible = st.checkbox("表示する", value=True, key=f"vis_{i}")
                        label   = label_input(key=f"lbl_{i}", default=default_name)
                        chosen_color = color_picker_popover(f"xps_color_{f.name}", default_hex)

                        # 個別オフセットスライダー
                        eoff_key = f"extra_offset_{i}"
                        eoff_num_key = f"extra_offset_num_{i}"
                        if eoff_key not in st.session_state:
                            st.session_state[eoff_key] = 0.0
                        if eoff_num_key not in st.session_state:
                            st.session_state[eoff_num_key] = 0.0
                        if normalize:
                            st.slider("オフセット調整", min_value=-5.0, max_value=15.0,
                                      step=0.05, key=eoff_key)
                        else:
                            def _from_slider(ek=eoff_key, nk=eoff_num_key):
                                st.session_state[nk] = st.session_state[ek]
                            def _from_num(ek=eoff_key, nk=eoff_num_key):
                                st.session_state[ek] = st.session_state[nk]
                            col_sl, col_ni = st.columns([3, 2])
                            with col_sl:
                                st.slider("Y位置（絶対値）", min_value=-100000.0, max_value=500000.0,
                                          step=100.0, key=eoff_key, on_change=_from_slider)
                            with col_ni:
                                st.number_input("数値入力 (eV)", step=100.0,
                                               key=eoff_num_key, on_change=_from_num)

                        # 成分の色設定
                        n_comps = _comp_counts.get(i, 0)
                        if n_comps > 0 and show_components:
                            st.markdown("**ピーク成分の色**")
                            st.caption("同じグループ番号の成分は同色になります（ダブレット対応）")
                            # グループ番号ごとに色を1つ設定
                            # まずグループ番号をユーザーが設定
                            groups = []
                            _period = max(1, n_comps // 2)
                            for j in range(n_comps):
                                gkey = f"comp_group_{i}_{j}"
                                if gkey not in st.session_state:
                                    st.session_state[gkey] = (j % _period) + 1
                                g = st.number_input(
                                    f"成分 {j+1} グループ",
                                    min_value=1, max_value=20,
                                    key=gkey,
                                )
                                groups.append(g)
                            # グループごとに色ピッカーを表示
                            unique_groups = sorted(set(groups))
                            st.markdown("**グループの色**")
                            for g in unique_groups:
                                gcolor_key = f"group_color_{i}_{g}"
                                g_idx = unique_groups.index(g)
                                default_comp_color = COMPONENT_COLORS[g_idx % len(COMPONENT_COLORS)]
                                st.caption(f"グループ {g}")
                                color_picker_popover(gcolor_key, default_comp_color)

                    orders.append(order)
                    visibles.append(visible)
                    labels.append(label)
                    colors_sel.append(chosen_color)

                sort_idx = sorted(range(len(xps_files)), key=lambda i: orders[i])
    else:
        for i, f in enumerate(xps_files):
            eoff_key = f"extra_offset_{i}"
            if eoff_key not in st.session_state:
                st.session_state[eoff_key] = 0.0
            orders.append(i + 1)
            visibles.append(st.session_state.get(f"vis_{i}", True))
            labels.append(st.session_state.get(f"_val_lbl_{i}", os.path.splitext(f.name)[0]))
            colors_sel.append(st.session_state.get(f"xps_color_{f.name}", "#000000"))
        sort_idx = list(range(len(xps_files)))

    # ===== データをキャッシュ読み込み =====
    def load_data(i):
        raw = xps_files[i].read()
        xps_files[i].seek(0)
        return read_xps_csv(raw)

    # ===== X軸範囲の決定 =====
    def get_xlim():
        if x_min_xps is not None:
            return x_min_xps, x_max_xps
        all_e = []
        for i in sort_idx:
            if not visibles[i]:
                continue
            d = load_data(i)
            if d and d["spectrum"] is not None:
                all_e.extend(d["energy"].tolist())
        if not all_e:
            return 0.0, 1200.0
        return min(all_e), max(all_e)

    xlim_lo, xlim_hi = get_xlim()

    # ===== 成分色取得ヘルパー =====
    def get_comp_color(file_idx, comp_idx):
        g         = st.session_state.get(f"comp_group_{file_idx}_{comp_idx}", comp_idx + 1)
        gcolor_key = f"group_color_{file_idx}_{g}"
        return st.session_state.get(gcolor_key, COMPONENT_COLORS[(g - 1) % len(COMPONENT_COLORS)])

    # ===== matplotlib 図（TIFF/PNG 出力用）=====
    def build_figure():
        plt.rcParams.update({
            "font.family":      ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size":        font_size,
            "mathtext.fontset": "custom",
            "mathtext.it":      "Arial:italic",
            "mathtext.rm":      "Arial",
        })
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        cumulative_y = 0.0
        side_labels  = []

        for i in sort_idx:
            if not visibles[i]:
                continue
            d = load_data(i)
            if d is None or d["spectrum"] is None:
                continue

            energy   = d["energy"]
            spectrum = d["spectrum"].copy()

            # 範囲フィルタ
            if x_min_xps is not None:
                mask = (energy >= x_min_xps) & (energy <= x_max_xps)
                energy   = energy[mask]
                spectrum = spectrum[mask]
                bkg   = d["background"][mask] if d["background"] is not None else None
                comps = [c[mask] for c in d["components"]]
            else:
                bkg   = d["background"]
                comps = d["components"]

            if np.max(spectrum) <= 0:
                continue

            extra_off = float(st.session_state.get(f"extra_offset_{i}", 0.0))

            if normalize:
                scale    = np.max(spectrum)
                spectrum = spectrum / scale
                bkg      = bkg / scale if bkg is not None else None
                comps    = [c / scale for c in comps]
                y_base   = cumulative_y + extra_off
                cumulative_y += (global_offset or 1.0)
            else:
                y_base = extra_off

            y_plot = spectrum + y_base
            side_labels.append((y_base + label_offset_y * np.max(spectrum), colors_sel[i], labels[i]))

            ax.plot(energy, y_plot, color=colors_sel[i], linewidth=1.5,
                    label=labels[i], zorder=3)

            if show_background and bkg is not None:
                ax.plot(energy, bkg + y_base, color=bg_color,
                        linestyle="--", linewidth=1.0, alpha=bg_alpha, zorder=2)

            if show_components and comps:
                for j, comp in enumerate(comps):
                    cc = get_comp_color(i, j)
                    y_comp = comp + y_base
                    base_line = (bkg + y_base) if (bkg is not None and show_background) \
                                else np.full_like(comp, y_base)
                    if fill_components:
                        ax.fill_between(energy, base_line, y_comp,
                                        color=cc, alpha=comp_alpha, zorder=1)
                    ax.plot(energy, y_comp, color=cc, linewidth=0.8, alpha=0.8, zorder=2)

        # サンプルラベル（Plotly と同じ座標ロジック）
        if show_side_labels:
            if reverse_x:
                left_x  = xlim_hi - label_offset_x
                right_x = xlim_lo + label_offset_x
            else:
                left_x  = xlim_lo + label_offset_x
                right_x = xlim_hi - label_offset_x
            x_pos = left_x if label_side == "左" else right_x
            ha    = "left"  if label_side == "左" else "right"
            for y_c, col, txt in side_labels:
                ax.text(x_pos, y_c, txt, color=col, fontsize=label_fontsize,
                        ha=ha, va="center",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

        # 軸
        if reverse_x:
            ax.set_xlim(xlim_hi, xlim_lo)
        else:
            ax.set_xlim(xlim_lo, xlim_hi)

        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.tick_params(which="major", axis="x", length=5, direction="in")
        if show_minor and minor_tick:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
            ax.tick_params(which="minor", axis="x", length=2.5, direction="in")

        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (a.u.)", labelpad=2)
        ax.set_yticks([])
        if show_legend:
            ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        return fig

    # ===== Plotly プレビュー =====
    def build_plotly_figure():
        pfig = make_subplots(rows=1, cols=1)
        cumulative_y = 0.0

        for i in sort_idx:
            if not visibles[i]:
                continue
            d = load_data(i)
            if d is None or d["spectrum"] is None:
                continue

            energy   = d["energy"]
            spectrum = d["spectrum"].copy()

            if x_min_xps is not None:
                mask = (energy >= x_min_xps) & (energy <= x_max_xps)
                energy   = energy[mask]
                spectrum = spectrum[mask]
                bkg   = d["background"][mask] if d["background"] is not None else None
                comps = [c[mask] for c in d["components"]]
                cnames = d["component_names"]
            else:
                bkg   = d["background"]
                comps = d["components"]
                cnames = d["component_names"]

            if len(spectrum) == 0 or np.max(spectrum) <= 0:
                continue

            extra_off = float(st.session_state.get(f"extra_offset_{i}", 0.0))

            if normalize:
                scale    = np.max(spectrum)
                spectrum = spectrum / scale
                bkg      = bkg / scale if bkg is not None else None
                comps    = [c / scale for c in comps]
                y_base   = cumulative_y + extra_off
                cumulative_y += (global_offset or 1.0)
            else:
                y_base = extra_off

            y_plot = spectrum + y_base

            pfig.add_trace(go.Scatter(
                x=energy, y=y_plot, name=labels[i],
                line=dict(color=colors_sel[i], width=1.5),
                mode="lines", showlegend=show_legend,
            ))

            if show_background and bkg is not None:
                pfig.add_trace(go.Scatter(
                    x=energy, y=bkg + y_base,
                    line=dict(color=bg_color, width=1.0, dash="dash"),
                    mode="lines", opacity=bg_alpha, showlegend=False,
                ))

            if show_components and comps:
                for j, comp in enumerate(comps):
                    cc = get_comp_color(i, j)
                    y_comp = comp + y_base
                    cname  = cnames[j] if j < len(cnames) else f"Comp {j+1}"
                    if fill_components:
                        base_vals = (bkg + y_base).tolist() if bkg is not None else [y_base] * len(comp)
                        pfig.add_trace(go.Scatter(
                            x=np.concatenate([energy, energy[::-1]]).tolist(),
                            y=np.concatenate([y_comp, np.array(base_vals)[::-1]]).tolist(),
                            fill="toself", fillcolor=cc,
                            line=dict(color=cc, width=0),
                            opacity=comp_alpha, showlegend=False,
                            name=f"{labels[i]} {cname}",
                        ))
                    pfig.add_trace(go.Scatter(
                        x=energy, y=y_comp,
                        line=dict(color=cc, width=0.8),
                        mode="lines", opacity=0.8, showlegend=False,
                    ))

            if show_side_labels:
                y_label = y_base + label_offset_y * np.max(spectrum)
                if reverse_x:
                    left_x  = xlim_hi - label_offset_x
                    right_x = xlim_lo + label_offset_x
                else:
                    left_x  = xlim_lo + label_offset_x
                    right_x = xlim_hi - label_offset_x
                x_pos = left_x  if label_side == "左" else right_x
                tpos  = "middle right" if label_side == "左" else "middle left"
                pfig.add_trace(go.Scatter(
                    x=[x_pos], y=[y_label], mode="text",
                    text=[labels[i]], textposition=tpos,
                    textfont=dict(color=colors_sel[i], size=label_fontsize, family="Arial"),
                    showlegend=False, hoverinfo="skip",
                ))

        x_range = [xlim_hi, xlim_lo] if reverse_x else [xlim_lo, xlim_hi]
        minor_cfg = dict(ticks="inside", dtick=minor_tick, showgrid=False) if (show_minor and minor_tick) else {}
        pfig.update_xaxes(
            range=x_range, dtick=major_tick, ticks="inside",
            showline=True, linecolor="black", mirror=False,
            showgrid=False, zeroline=False,
            tickfont=dict(color="black", size=font_size),
            title_text="Binding Energy (eV)",
            title_font=dict(color="black", size=font_size),
            minor=minor_cfg,
        )
        pfig.update_yaxes(
            showticklabels=False, showline=True, linecolor="black",
            showgrid=False, zeroline=False,
            title_text="Intensity (a.u.)",
            title_font=dict(color="black", size=font_size),
            title_standoff=5,
        )
        pfig.update_layout(
            dragmode="zoom",
            font=dict(family="Arial", size=font_size, color="black"),
            height=int(fig_height * 85),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=show_legend,
            legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top",
                        font=dict(color="black")),
            margin=dict(l=60, r=60, t=30, b=60),
        )
        return pfig

    # ===== 描画 =====
    with col_graph:
        st.caption("ドラッグで範囲ズーム ／ ダブルクリックでリセット")
        pfig = build_plotly_figure()
        st.plotly_chart(pfig, width='stretch', config={
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawrect"],
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

        fig = build_figure()

        # 出力プレビュー（ダウンロード不要で確認）
        with st.expander("📄 出力画像プレビュー（論文用 matplotlib）", expanded=False):
            st.pyplot(fig, width='stretch')

        buf = io.BytesIO()
        fig.savefig(buf, format="tiff", dpi=dpi_export, bbox_inches="tight")
        buf.seek(0)
        st.download_button(f"📥 TIFF として保存 ({dpi_export} DPI)",
                           data=buf, file_name="xps_result.tiff", mime="image/tiff")
        plt.close(fig)

else:
    st.info("サイドバーから XPS データファイル（.csv）をアップロードしてください。")
    st.markdown("""
    **対応 CSV フォーマット:**
    - シンプル形式: `Energy, Intensity`（2列）
    - 詳細形式: `Binding Energy, Spectrum, Background, [Component1], [Component2], ...`
    """)
