"""
Dashboard local do sinal semanal — E1 D7 + H1 (BTC/CDI, no-short).

Le scripts/production/data/signal_history.csv + dataset_production.csv e mostra:
  - Visao geral: KPIs, alocacao x preco x regime, acumulado vs BTC/CDI, drawdown
  - Por que da semana: decomposicao visual da alocacao de cada rebalance
    (regime, previsao, confianca), cenarios "e se", semanas vizinhas e o
    resultado realizado da janela
  - Visao mensal: por que o mes ficou pouco/muito alocado
  - Dados & ajuda: tabela completa e explicacao do calculo

Uso:
    streamlit run scripts/production/dashboard.py

Somente leitura: nao roda o modelo nem altera nenhum CSV. As metricas aqui sao
aproximacoes em janelas de rebalance (USD, gross); as oficiais do modelo sao
diarias em BRL (ver docs/MODEL_FINAL.md).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent / "data"
SIGNAL_CSV = DATA_DIR / "signal_history.csv"
DATASET_CSV = DATA_DIR / "dataset_production.csv"

# ── Paleta (referencia validada do guia de dataviz; light/dark selecionados) ──
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s1_dark": "#1c5cab",
        "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b",
        "div_neg": "#e34948", "div_mid": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s1_dark": "#86b6ef",
        "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b",
        "div_neg": "#e66767", "div_mid": "#383835",
    },
}
REGIME_COLOR_KEY = {"BULL": "good", "MILD": "warning", "BEAR": "critical"}
REGIME_DESC = {
    "BULL": "preco acima das medias de 50 e 200 dias — tendencia confirmada",
    "MILD": "preco acima da media de 200 dias, sem tendencia plena",
    "BEAR": "preco abaixo das medias — postura defensiva",
}
MESES = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
         7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}

FEATURES_CONTEXTO = {
    "volatility_7d": "Volatilidade 7d",
    "bb_position": "Posicao na banda de Bollinger",
    "nupl_ma30": "NUPL (media 30d)",
    "funding_rate_ma7": "Funding rate (MA7)",
    "reserveRisk": "Reserve Risk",
    "puellMultiple": "Puell Multiple",
    "m2_yoy_growth": "Crescimento M2 YoY",
    "sortino_30d": "Sortino 30d",
    "adx": "ADX (forca de tendencia)",
}


def theme_name() -> str:
    try:
        t = st.context.theme.type
        return "dark" if t == "dark" else "light"
    except Exception:
        return "dark"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════ Dados ═══════════════════════════

@st.cache_data(show_spinner=False)
def load_signals() -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["is_emergency"] = df["is_emergency"].astype(str).str.lower().eq("true")
    for c in ["previsao", "p_up", "confidence_factor", "allocation",
              "K_base", "K_effective", "retorno_btc", "retorno_strat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["alloc_pct"] = df["allocation"] * 100
    df["backfilled"] = df["action"].str.contains(r"\[BACKFILLED", na=False)
    df["emerg_ret"] = df["action"].str.extract(r"EMERGENCY \(ret ([+-]?[\d.]+)%\)")[0].astype(float)

    # CDI por janela, reconstruido de retorno_strat = a*btc + (1-a)*cdi (aprox.)
    a, rb, rs = df["allocation"], df["retorno_btc"], df["retorno_strat"]
    cdi = pd.Series(np.nan, index=df.index)
    cdi[a <= 0.001] = rs[a <= 0.001]
    mid = (a > 0.001) & (a < 0.9)
    cdi[mid] = (rs[mid] - a[mid] * rb[mid]) / (1 - a[mid])
    cdi = cdi.clip(0, 0.008)
    cdi = cdi.fillna(cdi.rolling(8, min_periods=1).median()).fillna(cdi.median())
    cdi[rs.isna()] = np.nan
    df["retorno_cdi"] = cdi
    return df


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    cols = ["date", "price_usd"] + list(FEATURES_CONTEXTO)
    df = pd.read_csv(DATASET_CSV, usecols=lambda c: c in cols)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_daily(sig: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    """Serie DIARIA reconstruida (como o metodo oficial, que mede risco em base
    diaria): retorno do dia t = alloc vigente x BTC diario + (1 - alloc) x CDI
    diario. A alocacao vigente e a do ultimo rebalance ate t-1; o CDI diario e
    o da janela, distribuido geometricamente pelos dias."""
    px = ds[["date", "price_usd"]].dropna().sort_values("date")
    px = px[(px["date"] >= sig["date"].iloc[0]) &
            (px["date"] <= sig["date"].iloc[-1])].reset_index(drop=True)
    px["ret_btc"] = px["price_usd"].pct_change()
    asof = pd.merge_asof(px[["date"]], sig[["date", "allocation"]],
                         on="date", direction="backward")
    px["alloc"] = asof["allocation"].shift(1)
    rate = np.full(len(px), np.nan)
    for i in range(len(sig) - 1):
        c = sig["retorno_cdi"].iloc[i]
        if pd.isna(c):
            continue
        d0, d1 = sig["date"].iloc[i], sig["date"].iloc[i + 1]
        n = max((d1 - d0).days, 1)
        mask = (px["date"] > d0) & (px["date"] <= d1)
        rate[mask.values] = (1 + c) ** (1 / n) - 1
    px["ret_cdi"] = rate
    px = px.dropna(subset=["ret_btc", "alloc", "ret_cdi"]).reset_index(drop=True)
    px["ret_strat"] = px["alloc"] * px["ret_btc"] + (1 - px["alloc"]) * px["ret_cdi"]
    out = pd.DataFrame({
        "date": px["date"],
        "ret_strat": px["ret_strat"], "ret_btc": px["ret_btc"],
        "ret_cdi": px["ret_cdi"],
        "strat": 100 * (1 + px["ret_strat"]).cumprod(),
        "btc": 100 * (1 + px["ret_btc"]).cumprod(),
        "cdi": 100 * (1 + px["ret_cdi"]).cumprod(),
    })
    out["dd"] = (out["strat"] / out["strat"].cummax() - 1) * 100
    return out


# ═══════════════════════════ Graficos ═══════════════════════════

def base_layout(fig: go.Figure, T: dict, height: int, unified: bool = True):
    fig.update_layout(
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=T["text"], size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=8, r=8, t=40, b=8),
        hovermode="x unified" if unified else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(color=T["text2"])),
        hoverlabel=dict(bgcolor=T["surface"], font=dict(color=T["text"]),
                        bordercolor=T["grid"]),
    )
    fig.update_xaxes(gridcolor=T["grid"], linecolor=T["baseline"], zeroline=False,
                     tickfont=dict(color=T["muted"]),
                     minor=dict(showgrid=False))
    fig.update_yaxes(gridcolor=T["grid"], linecolor=T["baseline"], zeroline=False,
                     tickfont=dict(color=T["muted"]),
                     minor=dict(showgrid=False))
    return fig


def plot(fig: go.Figure):
    """Renderiza sem o template do Streamlit (a paleta controla tudo)."""
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displayModeBar": False})


def regime_runs(sig: pd.DataFrame):
    runs, start, cur = [], sig["date"].iloc[0], sig["regime"].iloc[0]
    for i in range(1, len(sig)):
        if sig["regime"].iloc[i] != cur:
            runs.append((start, sig["date"].iloc[i], cur))
            start, cur = sig["date"].iloc[i], sig["regime"].iloc[i]
    runs.append((start, sig["date"].iloc[-1], cur))
    return runs


def fig_alloc_price(sig: pd.DataFrame, T: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.06)
    for x0, x1, reg in regime_runs(sig):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                      xref="x", yref="paper", layer="below", line_width=0,
                      fillcolor=rgba(T[REGIME_COLOR_KEY[reg]], 0.09))
    for reg in ["BULL", "MILD", "BEAR"]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=f"Regime {reg}",
            marker=dict(symbol="square", size=10,
                        color=rgba(T[REGIME_COLOR_KEY[reg]], 0.35)),
            hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sig["date"], y=sig["price_usd"], name="BTC (USD)",
        line=dict(color=T["text2"], width=2),
        hovertemplate="US$ %{y:,.0f}<extra>BTC</extra>"), row=1, col=1)

    emerg = sig[sig["is_emergency"]]
    fig.add_trace(go.Scatter(
        x=emerg["date"], y=emerg["price_usd"], mode="markers",
        name="Emergencia (|ret| > 8%)",
        marker=dict(symbol="diamond", size=9, color=T["critical"],
                    line=dict(color=T["surface"], width=2)),
        hovertemplate="US$ %{y:,.0f} (ret diario %{customdata:+.1f}%)<extra>Emergencia</extra>",
        customdata=emerg["emerg_ret"]), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sig["date"], y=sig["alloc_pct"], name="Alocacao em BTC",
        line=dict(color=T["s1"], width=2, shape="hv"),
        fill="tozeroy", fillcolor=rgba(T["s1"], 0.12),
        customdata=np.stack([sig["regime"], sig["p_up"], sig["previsao"] * 100], axis=-1),
        hovertemplate=("%{y:.1f}% BTC · regime %{customdata[0]} · "
                       "P(up) %{customdata[1]:.2f} · prev. 3d %{customdata[2]:+.2f}%"
                       "<extra>Alocacao</extra>")), row=2, col=1)

    fig.update_yaxes(type="log", title_text="Preco BTC (log)", row=1, col=1,
                     title_font=dict(color=T["muted"], size=12))
    fig.update_yaxes(range=[0, 105], title_text="Alocacao (%)", row=2, col=1,
                     ticksuffix="%", title_font=dict(color=T["muted"], size=12))
    # Os traces de legenda com x=[None] entram primeiro e fariam o plotly
    # inferir eixo "linear" — força "date" para as datas mapearem.
    fig.update_xaxes(type="date")
    return base_layout(fig, T, 540)


def fig_performance(curves: pd.DataFrame, T: dict) -> go.Figure:
    fig = go.Figure()
    series = [("strat", "Estrategia", T["s1"]),
              ("btc", "BTC", T["s2"]),
              ("cdi", "CDI (aprox.)", T["s3"])]
    for col, name, color in series:
        fig.add_trace(go.Scatter(
            x=curves["date"], y=curves[col], name=name,
            line=dict(color=color, width=2),
            hovertemplate="%{y:,.0f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(
            x=[curves["date"].iloc[-1]], y=[curves[col].iloc[-1]],
            mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=8, color=color, line=dict(color=T["surface"], width=2))))
        fig.add_annotation(
            x=curves["date"].iloc[-1], y=np.log10(curves[col].iloc[-1]),
            text=f"{name.split(' ')[0]} {curves[col].iloc[-1]:,.0f}",
            showarrow=False, xanchor="left", xshift=8,
            font=dict(color=T["text2"], size=12))
    fig.update_yaxes(type="log", title_text="Base 100 (log)",
                     title_font=dict(color=T["muted"], size=12))
    fig.update_layout(margin_r=95)
    return base_layout(fig, T, 380)


def fig_drawdown(curves: pd.DataFrame, T: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=curves["date"], y=curves["dd"], name="Drawdown",
        line=dict(color=T["critical"], width=2),
        fill="tozeroy", fillcolor=rgba(T["critical"], 0.12),
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>"))
    fig.update_yaxes(ticksuffix="%", title_text="Drawdown (%)",
                     title_font=dict(color=T["muted"], size=12))
    fig.update_layout(showlegend=False)
    return base_layout(fig, T, 380)


def fig_monthly(monthly: pd.DataFrame, selected: str, T: dict) -> go.Figure:
    colors = [T["s1_dark"] if m == selected else T["s1"] for m in monthly["mes"]]
    fig = go.Figure(go.Bar(
        x=monthly["mes"], y=monthly["alloc_media"],
        marker=dict(color=colors, cornerradius=4),
        customdata=np.stack([monthly["regime_dom"], monthly["p_up_media"],
                             monthly["n_emerg"]], axis=-1),
        hovertemplate=("%{y:.1f}% medio · regime dominante %{customdata[0]} · "
                       "P(up) media %{customdata[1]:.2f} · "
                       "%{customdata[2]} emergencia(s)<extra>%{x}</extra>")))
    fig.update_yaxes(ticksuffix="%", title_text="Alocacao media (%)",
                     title_font=dict(color=T["muted"], size=12))
    fig.update_layout(showlegend=False, bargap=0.35)
    return base_layout(fig, T, 320, unified=False)


def fig_vizinhas(sig_all: pd.DataFrame, idx: int, T: dict) -> go.Figure:
    """Alocacao das ~20 semanas ao redor do rebalance selecionado."""
    lo, hi = max(0, idx - 12), min(len(sig_all), idx + 8)
    win = sig_all.iloc[lo:hi]
    labels = win["date"].dt.strftime("%d/%m/%y")
    colors = [T["s1"] if i == idx else rgba(T["s1"], 0.35) for i in win.index]
    fig = go.Figure(go.Bar(
        x=labels, y=win["alloc_pct"],
        marker=dict(color=colors, cornerradius=4),
        customdata=np.stack([win["regime"], win["p_up"]], axis=-1),
        hovertemplate="%{y:.1f}% · %{customdata[0]} · P(up) %{customdata[1]:.2f}<extra>%{x}</extra>"))
    media = sig_all["alloc_pct"].mean()
    fig.add_hline(y=media, line=dict(color=T["baseline"], width=1),
                  annotation_text=f"media historica {media:.0f}%",
                  annotation_font=dict(color=T["muted"], size=11))
    fig.update_yaxes(ticksuffix="%", range=[0, 105])
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig.update_layout(showlegend=False, bargap=0.3)
    return base_layout(fig, T, 300, unified=False)


def fig_e_se(row: pd.Series, T: dict) -> go.Figure:
    """Cenarios contrafactuais: quanto cada fator cortou da posicao."""
    prev, K, conf = row["previsao"], row["K_base"], row["confidence_factor"]
    real = float(np.clip(prev * K * conf, 0, 1)) * 100
    sem_conf = float(np.clip(prev * K * 1.0, 0, 1)) * 100
    se_bull = float(np.clip(prev * 60 * conf, 0, 1)) * 100
    labels = ["Se o regime fosse BULL (K=60)", "Sem corte de confianca", "Alocacao real"]
    vals = [se_bull, sem_conf, real]
    colors = [rgba(T["s1"], 0.4), rgba(T["s1"], 0.4), T["s1"]]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        textfont=dict(color=T["text2"]),
        hovertemplate="%{x:.1f}%<extra>%{y}</extra>"))
    fig.update_xaxes(range=[0, max(vals + [10]) * 1.3], ticksuffix="%")
    fig.update_yaxes(tickfont=dict(color=T["text2"], size=12))
    fig.update_layout(showlegend=False, bargap=0.45)
    return base_layout(fig, T, 240, unified=False)


def metricas_periodo(sig: pd.DataFrame, daily: pd.DataFrame) -> dict | None:
    """Metricas de factsheet: risco em base DIARIA reconstruida (USD, gross,
    365 dias/ano, excesso sobre CDI); capturas e melhor/pior janela em base de
    janelas de rebalance (nivel de decisao)."""
    d = sig.dropna(subset=["retorno_strat", "retorno_btc", "retorno_cdi"])
    if len(d) < 8 or len(daily) < 60:
        return None
    years = (daily["date"].iloc[-1] - daily["date"].iloc[0]).days / 365.25
    if years <= 0:
        return None
    cagr = (daily["strat"].iloc[-1] / daily["strat"].iloc[0]) ** (1 / years) - 1
    cagr_btc = (daily["btc"].iloc[-1] / daily["btc"].iloc[0]) ** (1 / years) - 1
    cagr_cdi = (daily["cdi"].iloc[-1] / daily["cdi"].iloc[0]) ** (1 / years) - 1
    rd, cd = daily["ret_strat"], daily["ret_cdi"]
    vol = rd.std() * np.sqrt(365)
    ex = rd - cd
    sharpe = ex.mean() / ex.std() * np.sqrt(365) if ex.std() > 0 else np.nan
    dd_dev = float(np.sqrt((np.minimum(ex, 0) ** 2).mean()))
    sortino = (ex.mean() * 365) / (dd_dev * np.sqrt(365)) if dd_dev > 0 else np.nan
    dd_daily = (daily["strat"] / daily["strat"].cummax() - 1)
    maxdd = float(dd_daily.min())
    calmar = cagr / abs(maxdd) if maxdd < -0.0001 else np.nan
    rs, rb = d["retorno_strat"], d["retorno_btc"]
    up, dn = rb > 0, rb < 0
    cap_up = rs[up].mean() / rb[up].mean() if up.any() and rb[up].mean() != 0 else np.nan
    cap_dn = rs[dn].mean() / rb[dn].mean() if dn.any() and rb[dn].mean() != 0 else np.nan
    i_best, i_worst = rs.idxmax(), rs.idxmin()
    # Comparacao com renda fixa: % do CDI (razao dos acumulados, convencao
    # de fundos BR) e consistencia mensal (% dos meses em que bateu o CDI)
    ret_acc = daily["strat"].iloc[-1] / daily["strat"].iloc[0] - 1
    cdi_acc = daily["cdi"].iloc[-1] / daily["cdi"].iloc[0] - 1
    pct_cdi = ret_acc / cdi_acc if cdi_acc > 0.001 else np.nan
    comp = lambda r: (1 + r).prod() - 1
    dm = daily.copy()
    dm["mes"] = dm["date"].dt.to_period("M")
    ms = dm.groupby("mes")["ret_strat"].apply(comp)
    mc = dm.groupby("mes")["ret_cdi"].apply(comp)
    consist = float((ms >= mc).mean()) if len(ms) else np.nan
    return {
        "anos": years, "cagr": cagr, "cagr_btc": cagr_btc, "cagr_cdi": cagr_cdi,
        "vol": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "maxdd": maxdd, "cap_up": cap_up, "cap_dn": cap_dn,
        "pct_cdi": pct_cdi, "consist": consist, "n_meses": len(ms),
        "best": rs.max(), "best_dt": d.loc[i_best, "date"],
        "worst": rs.min(), "worst_dt": d.loc[i_worst, "date"],
        "expo": sig["allocation"].mean(),
        "zeradas": (sig["allocation"] == 0).mean(),
    }


def fig_heatmap_mensal(sig: pd.DataFrame, T: dict) -> go.Figure | None:
    """Retornos mensais compostos (janela atribuida ao mes do rebalance)."""
    d = sig.dropna(subset=["retorno_strat", "retorno_btc", "retorno_cdi"]).copy()
    if d.empty:
        return None
    d["ano"], d["m"] = d["date"].dt.year, d["date"].dt.month
    comp = lambda r: (1 + r).prod() - 1
    ms = d.groupby(["ano", "m"])["retorno_strat"].apply(comp) * 100
    mc = d.groupby(["ano", "m"])["retorno_cdi"].apply(comp) * 100
    mb = d.groupby(["ano", "m"])["retorno_btc"].apply(comp) * 100
    ys = d.groupby("ano")["retorno_strat"].apply(comp) * 100
    yc = d.groupby("ano")["retorno_cdi"].apply(comp) * 100
    yb = d.groupby("ano")["retorno_btc"].apply(comp) * 100
    anos = sorted(d["ano"].unique(), reverse=True)
    cols = [MESES[m] for m in range(1, 13)] + ["Ano"]
    z, txt, cdata = [], [], []
    for ano in anos:
        zrow, trow, crow = [], [], []
        for m in range(1, 13):
            v = ms.get((ano, m), np.nan)
            zrow.append(v)
            trow.append("" if pd.isna(v) else f"{v:+.1f}")
            crow.append([mc.get((ano, m), np.nan), mb.get((ano, m), np.nan)])
        zrow.append(ys.get(ano, np.nan))
        trow.append("" if pd.isna(ys.get(ano, np.nan)) else f"{ys[ano]:+.1f}")
        crow.append([yc.get(ano, np.nan), yb.get(ano, np.nan)])
        z.append(zrow); txt.append(trow); cdata.append(crow)
    scale = [[0.0, T["div_neg"]], [0.5, T["div_mid"]], [1.0, T["s1"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=[str(a) for a in anos], text=txt,
        customdata=np.array(cdata),
        texttemplate="%{text}", textfont=dict(size=11, color=T["text"]),
        colorscale=scale, zmid=0, zmin=-20, zmax=20,
        xgap=2, ygap=2, showscale=False,
        hovertemplate=("%{y} %{x}: estrategia %{z:.1f}% · "
                       "CDI %{customdata[0]:.1f}% · BTC %{customdata[1]:.1f}%"
                       "<extra></extra>")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(side="top")
    return base_layout(fig, T, 60 + 42 * len(anos), unified=False)


# ═══════════════ Interpretacao (regras da formula real) ═══════════════

def classifica_posicao(pct: float) -> tuple:
    if pct <= 0.01:
        return "Zerada", "critical"
    if pct < 10:
        return "Minima", "critical"
    if pct < 30:
        return "Pequena", "warning"
    if pct < 60:
        return "Moderada", "s1"
    if pct < 85:
        return "Grande", "good"
    return "Quase maxima", "good"


def principal_freio(row: pd.Series) -> str:
    prev, K, conf = row["previsao"], row["K_base"], row["confidence_factor"]
    if prev <= 0:
        return "previsao negativa — no-short zera a posicao"
    real = np.clip(prev * K * conf, 0, 1)
    sem_conf = np.clip(prev * K, 0, 1)
    se_bull = np.clip(prev * 60 * conf, 0, 1)
    potencial = np.clip(prev * 60, 0, 1)
    if potencial < 0.30:
        return f"previsao fraca — mesmo em BULL sem corte daria so {potencial * 100:.0f}%"
    if real >= 0.85:
        return "nenhum — posicao perto do teto"
    if (se_bull - real) >= (sem_conf - real):
        return f"regime {row['regime']} — em BULL a posicao iria a {se_bull * 100:.0f}%"
    return f"incerteza do classificador — sem o corte iria a {sem_conf * 100:.0f}%"


def cartoes_fatores(row: pd.Series) -> list:
    """3 cartoes (titulo, valor, leitura, cor)."""
    reg = row["regime"]
    reg_status = {"BULL": ("apoia a posicao", "good"),
                  "MILD": ("corta moderadamente", "warning"),
                  "BEAR": ("corta forte", "critical")}[reg]
    cards = [{
        "titulo": "Regime de mercado",
        "valor": f"{reg} · K = {row['K_base']:.0f}",
        "leitura": f"{REGIME_DESC[reg]} ({reg_status[0]})",
        "cor": reg_status[1],
    }]
    prev = row["previsao"]
    if prev <= 0:
        p = ("nula/negativa — no-short zera a posicao", "critical")
    elif prev < 0.005:
        p = ("muito fraca — quase nao sustenta posicao", "critical")
    elif prev < 0.015:
        p = ("moderada", "warning")
    else:
        p = ("forte — sustenta posicao relevante", "good")
    cards.append({"titulo": "Previsao de retorno (3d)",
                  "valor": f"{prev * 100:+.2f}%", "leitura": p[0], "cor": p[1]})
    p_up, conf = row["p_up"], row["confidence_factor"]
    if conf < 0.6:
        c = ("classificador em cima do muro — corta quase metade", "critical")
    elif conf < 0.85:
        c = ("confianca mediana — corte moderado", "warning")
    else:
        c = ("classificador confiante — corte minimo", "good")
    cards.append({"titulo": "Confianca do classificador",
                  "valor": f"{conf:.2f} · P(up) {p_up:.2f}", "leitura": c[0], "cor": c[1]})
    return cards


def leitura_semana(row: pd.Series) -> str:
    a = row["allocation"]
    raw = row["previsao"] * row["K_base"] * row["confidence_factor"]
    if row["previsao"] <= 0:
        return ("Como o modelo e **no-short** (mandato long-biased), previsao <= 0 zera a "
                "posicao: **0% BTC / 100% CDI** — independentemente do regime e da confianca.")
    if raw >= 1:
        return (f"O produto previsao x K x confianca ({raw:.2f}) estourou o teto: "
                f"**alocacao travada em 100% BTC**.")
    return (f"Formula: {row['previsao'] * 100:+.2f}% x {row['K_base']:.0f} x "
            f"{row['confidence_factor']:.2f} = **{a * 100:.1f}% BTC / {(1 - a) * 100:.1f}% CDI**.")


def resultado_janela(sig_all: pd.DataFrame, idx: int) -> dict | None:
    row = sig_all.iloc[idx]
    if pd.isna(row["retorno_btc"]) or pd.isna(row["retorno_strat"]):
        return None
    fim = sig_all["date"].iloc[idx + 1] if idx + 1 < len(sig_all) else None
    rb, rs, rc = row["retorno_btc"], row["retorno_strat"], row["retorno_cdi"]
    a = row["allocation"]
    if rb > rc and a >= 0.5:
        veredito = "Conviccao recompensada: BTC superou o CDI na janela e a carteira estava posicionada."
    elif rb > rc and a < 0.5:
        veredito = (f"Custo da cautela: BTC rendeu {rb * 100:+.1f}% na janela e a carteira "
                    f"capturou {rs * 100:+.1f}%.")
    elif rb <= rc and a < 0.5:
        veredito = (f"A cautela protegeu: BTC fez {rb * 100:+.1f}% na janela e a carteira "
                    f"segurou em {rs * 100:+.1f}%.")
    else:
        veredito = "Posicao grande numa janela ruim do BTC — a semana custou vs CDI."
    return {"fim": fim, "rb": rb * 100, "rs": rs * 100, "rc": rc * 100,
            "vs_btc": (rs - rb) * 100, "vs_cdi": (rs - rc) * 100, "veredito": veredito}


def justificar_mes(g: pd.DataFrame, media_geral: float) -> str:
    alloc = g["alloc_pct"].mean()
    if alloc < media_geral - 10:
        nivel = "bem abaixo da"
    elif alloc < media_geral - 3:
        nivel = "abaixo da"
    elif alloc > media_geral + 10:
        nivel = "bem acima da"
    elif alloc > media_geral + 3:
        nivel = "acima da"
    else:
        nivel = "em linha com a"
    shares = g["regime"].value_counts(normalize=True)
    dom = shares.idxmax()
    razoes = []
    if shares.get("BEAR", 0) >= 0.5:
        razoes.append(f"o regime foi **BEAR em {shares['BEAR'] * 100:.0f}%** dos rebalances "
                      f"— K cai para 15 e ate previsoes positivas viram posicoes pequenas")
    elif shares.get("BULL", 0) >= 0.5:
        razoes.append(f"o regime foi **BULL em {shares['BULL'] * 100:.0f}%** dos rebalances "
                      f"(K = 60, espaco para posicoes grandes)")
    else:
        razoes.append(f"o regime dominante foi **{dom}** ({shares[dom] * 100:.0f}% dos rebalances)")
    prev_m = g["previsao"].mean() * 100
    if prev_m <= 0.3:
        razoes.append(f"as previsoes de retorno 3d foram **fracas ou negativas** na media ({prev_m:+.2f}%)")
    else:
        razoes.append(f"as previsoes de retorno 3d ficaram em **{prev_m:+.2f}%** na media")
    incerteza = (g["p_up"] - 0.5).abs().mean()
    if incerteza < 0.1:
        razoes.append(f"o classificador passou o mes **incerto** (P(up) media {g['p_up'].mean():.2f}), "
                      f"e o fator sigmoide cortou as posicoes")
    n_emerg = int(g["is_emergency"].sum())
    if n_emerg:
        razoes.append(f"houve **{n_emerg} rebalance(s) de emergencia** (|ret diario| > 8%)")
    zeradas = (g["allocation"] == 0).mean()
    if zeradas >= 0.5:
        razoes.append(f"**{zeradas * 100:.0f}% dos rebalances zeraram a posicao** (100% CDI)")
    return (f"Alocacao media de **{alloc:.1f}%**, {nivel} media historica "
            f"({media_geral:.1f}%). Por que: " + "; ".join(razoes) + ".")


def contexto_features(ds: pd.DataFrame, quando: pd.Timestamp) -> pd.DataFrame:
    upto = ds[ds["date"] <= quando]
    if upto.empty:
        return pd.DataFrame()
    atual = upto.iloc[-1]
    linhas = []
    for feat, label in FEATURES_CONTEXTO.items():
        if feat not in upto.columns:
            continue
        serie = upto[feat].dropna()
        if serie.empty or pd.isna(atual[feat]):
            continue
        pct = float((serie <= atual[feat]).mean() * 100)
        linhas.append({"Feature": label, "Valor": round(float(atual[feat]), 4),
                       "Percentil historico": round(pct, 1)})
    return pd.DataFrame(linhas)


# ═══════════════════════════════ UI ═══════════════════════════════

st.set_page_config(page_title="Sinal BTC/CDI — E1 D7 + H1", page_icon="📈",
                   layout="wide")
T = PALETTE[theme_name()]

st.markdown(f"""<style>
div[data-testid="stMetric"] {{
  background: var(--secondary-background-color);
  border: 1px solid rgba(128,128,128,.18);
  border-radius: 12px; padding: 12px 16px;
}}
.factor-card {{
  background: var(--secondary-background-color);
  border: 1px solid rgba(128,128,128,.18);
  border-radius: 12px; padding: 14px 16px; height: 100%;
  border-left-width: 4px; border-left-style: solid;
}}
.factor-title {{ font-size: .78rem; letter-spacing: .02em; opacity: .65;
                 text-transform: uppercase; margin-bottom: 2px; }}
.factor-value {{ font-size: 1.45rem; font-weight: 650; line-height: 1.25; }}
.factor-sub {{ font-size: .86rem; opacity: .8; margin-top: 4px; }}
.verdict-card {{
  background: var(--secondary-background-color);
  border: 1px solid rgba(128,128,128,.18);
  border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
  display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
  justify-content: space-between;
}}
.v-label {{ font-size: .8rem; opacity: .65; text-transform: uppercase;
            letter-spacing: .02em; }}
.v-value {{ font-size: 2.1rem; font-weight: 700; line-height: 1.15; }}
.v-cdi {{ font-size: 1.1rem; font-weight: 500; opacity: .6; }}
.pills {{ display: flex; flex-wrap: wrap; gap: 8px; max-width: 60%; }}
.pill {{ border-radius: 999px; padding: 4px 12px; font-size: .82rem;
         font-weight: 600; white-space: nowrap; }}
</style>""", unsafe_allow_html=True)

sig_all = load_signals()
ds = load_dataset()
ultimo = sig_all.iloc[-1]

st.title("Sinal semanal BTC/CDI — E1 D7 + H1")
st.caption("Modelo long-biased no-short · rebalance sexta + emergencia (|ret| > 8%) · "
           "alocacao = clip(previsao 3d x K[regime] x sigmoide(confianca), 0, 1). "
           "Metricas desta pagina sao aproximacoes em janelas de rebalance (USD, gross); "
           "as oficiais sao diarias em BRL.")

# ── Filtro de periodo (escopa Visao geral e Visao mensal) ──
colf1, colf2 = st.columns([3, 2])
with colf1:
    periodo = st.radio(
        "Periodo", ["Tudo", "24 meses", "12 meses", "6 meses", "Ano atual", "Personalizado"],
        horizontal=True, label_visibility="collapsed")
fim_hist = sig_all["date"].max()
ini_hist = sig_all["date"].min()
if periodo == "Personalizado":
    with colf2:
        ini_sel, fim_sel = st.slider(
            "Intervalo", min_value=ini_hist.to_pydatetime(),
            max_value=fim_hist.to_pydatetime(),
            value=(ini_hist.to_pydatetime(), fim_hist.to_pydatetime()),
            format="MM/YYYY", label_visibility="collapsed")
    inicio, fim = pd.Timestamp(ini_sel), pd.Timestamp(fim_sel)
elif periodo == "Tudo":
    inicio, fim = ini_hist, fim_hist
elif periodo == "Ano atual":
    inicio, fim = pd.Timestamp(year=fim_hist.year, month=1, day=1), fim_hist
else:
    inicio, fim = fim_hist - pd.DateOffset(months=int(periodo.split()[0])), fim_hist

sig = sig_all[(sig_all["date"] >= inicio) & (sig_all["date"] <= fim)].reset_index(drop=True)
curves_all = build_daily(sig_all, ds)
curves = curves_all[(curves_all["date"] >= inicio) & (curves_all["date"] <= fim)].copy()
if len(curves) > 1:
    for c in ["strat", "btc", "cdi"]:
        curves[c] = curves[c] / curves[c].iloc[0] * 100
    curves["dd"] = (curves["strat"] / curves["strat"].cummax() - 1) * 100

# ── KPIs (sinal atual sempre global; resto no periodo) ──
c1, c2, c3, c4, c5 = st.columns(5)
delta_alloc = (ultimo["alloc_pct"] - sig_all["alloc_pct"].iloc[-2]) if len(sig_all) > 1 else 0
c1.metric("Sinal atual (BTC)", f"{ultimo['alloc_pct']:.1f}%",
          f"{delta_alloc:+.1f} pp vs rebal anterior", delta_color="off",
          help=f"Rebalance de {ultimo['date'].date()} ({ultimo['regime']})")
c2.metric("Regime atual", ultimo["regime"], f"K = {ultimo['K_base']:.0f}", delta_color="off")
ret_strat_per = (curves["strat"].iloc[-1] / 100 - 1) * 100 if len(curves) > 1 else 0.0
ret_btc_per = (curves["btc"].iloc[-1] / 100 - 1) * 100 if len(curves) > 1 else 0.0
ret_cdi_per = (curves["cdi"].iloc[-1] / 100 - 1) * 100 if len(curves) > 1 else 0.0
pct_cdi_per = ret_strat_per / ret_cdi_per * 100 if ret_cdi_per > 0.1 else np.nan
c3.metric("Estrategia no periodo", f"{ret_strat_per:+.0f}%",
          f"{ret_strat_per - ret_cdi_per:+.0f} pp vs CDI",
          help=(f"CDI (aprox.) {ret_cdi_per:+.1f}% · BTC {ret_btc_per:+.1f}% · "
                + (f"equivale a {pct_cdi_per:,.0f}% do CDI"
                   if not np.isnan(pct_cdi_per) else "")))
c4.metric("Drawdown max.", f"{curves['dd'].min():.1f}%" if len(curves) > 1 else "—",
          help="No periodo selecionado, base diaria reconstruida (aprox., USD)")
val = sig.dropna(subset=["retorno_btc"])
val = val[val["previsao"] != 0]
acerto = ((np.sign(val["previsao"]) == np.sign(val["retorno_btc"])).mean() * 100
          if len(val) else np.nan)
c5.metric("Acerto direcional", f"{acerto:.0f}%" if not np.isnan(acerto) else "—",
          help="Sinal da previsao 3d vs sinal do retorno BTC ate o rebal seguinte (aprox.)")

tab_geral, tab_semana, tab_mes, tab_dados = st.tabs(
    ["📊 Visao geral", "🧭 Por que da semana", "📅 Visao mensal", "📄 Dados & ajuda"])

# ═══ Aba 1: Visao geral ═══
with tab_geral:
    st.subheader("Alocacao, preco e regime")
    plot(fig_alloc_price(sig, T))
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Desempenho acumulado")
        plot(fig_performance(curves, T))
    with col_b:
        st.subheader("Drawdown da estrategia")
        plot(fig_drawdown(curves, T))

    st.subheader("Metricas do periodo")
    met = metricas_periodo(sig, curves)
    if met is None:
        st.info("Periodo curto demais para metricas (minimo ~8 janelas fechadas).")
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("CAGR", f"{met['cagr'] * 100:+.1f}%",
                  f"{(met['cagr'] - met['cagr_cdi']) * 100:+.0f} pp a.a. vs CDI",
                  help=f"CDI {met['cagr_cdi'] * 100:+.1f}% · BTC {met['cagr_btc'] * 100:+.1f}% "
                       f"· {met['anos']:.1f} anos")
        m2.metric("% do CDI", f"{met['pct_cdi'] * 100:,.0f}%"
                  if not np.isnan(met['pct_cdi']) else "—",
                  help="Retorno acumulado da estrategia dividido pelo do CDI "
                       "(convencao de fundos de renda fixa). 100% = empatou com o CDI")
        m3.metric("Volatilidade anual.", f"{met['vol'] * 100:.1f}%",
                  help="Desvio-padrao diario x sqrt(365)")
        m4.metric("Sharpe (excesso CDI)", f"{met['sharpe']:.2f}")
        m5.metric("Sortino (excesso CDI)", f"{met['sortino']:.2f}",
                  help="Aprox. em janelas de rebalance. Oficial diario BRL: 3.84 "
                       "(janela canonica, 10 seeds)")
        m6.metric("Calmar", f"{met['calmar']:.2f}" if not np.isnan(met['calmar']) else "—",
                  help="CAGR / |max drawdown|")
        n1, n2, n3, n4, n5, n6 = st.columns(6)
        n1.metric("Meses acima do CDI", f"{met['consist'] * 100:.0f}%"
                  if not np.isnan(met['consist']) else "—",
                  help=f"Consistencia: % dos {met['n_meses']} meses do periodo em que a "
                       f"estrategia rendeu pelo menos o CDI")
        n2.metric("Captura de alta", f"{met['cap_up'] * 100:.0f}%",
                  help="Retorno medio da estrategia nas janelas de BTC positivo, "
                       "como % do retorno do BTC")
        n3.metric("Captura de baixa", f"{met['cap_dn'] * 100:.0f}%",
                  help="Quanto menor, melhor: participacao nas quedas do BTC")
        n4.metric("Exposicao media", f"{met['expo'] * 100:.0f}%",
                  f"{met['zeradas'] * 100:.0f}% das janelas zeradas", delta_color="off")
        n5.metric("Melhor janela", f"{met['best'] * 100:+.1f}%",
                  help=f"{met['best_dt'].date()}")
        n6.metric("Pior janela", f"{met['worst'] * 100:+.1f}%",
                  help=f"{met['worst_dt'].date()}")
        st.caption("O benchmark justo e o **CDI**: a estrategia e uma carteira de renda "
                   "fixa que ocasionalmente compra BTC (exposicao media "
                   f"{met['expo'] * 100:.0f}%) — comparar com 100% BTC superestima o risco "
                   "tomado. BTC aparece so como referencia de captura. Risco em base "
                   "diaria reconstruida (USD, gross, 365 dias/ano), CDI reconstruido das "
                   "janelas; as metricas oficiais sao diarias em BRL, multi-seed.")

# ═══ Aba 2: Por que da semana ═══
with tab_semana:
    opcoes = sig_all.sort_values("date", ascending=False)
    escolha = st.selectbox(
        "Rebalance", opcoes.index,
        format_func=lambda i: (f"{opcoes.loc[i, 'date'].date()} — "
                               f"{opcoes.loc[i, 'alloc_pct']:.1f}% BTC ({opcoes.loc[i, 'regime']})"
                               + (" · EMERGENCIA" if opcoes.loc[i, 'is_emergency'] else "")))
    row = sig_all.loc[escolha]
    idx = int(escolha)

    tam_txt, tam_cor = classifica_posicao(row["alloc_pct"])
    pills = [f'<span class="pill" style="background:{rgba(T[tam_cor] if tam_cor != "s1" else T["s1"], .16)};'
             f'color:var(--text-color)">Posicao: {tam_txt}</span>',
             f'<span class="pill" style="background:{rgba(T[REGIME_COLOR_KEY[row["regime"]]], .16)};'
             f'color:var(--text-color)">Regime {row["regime"]}</span>']
    if row["is_emergency"]:
        ret = row["emerg_ret"]
        ret_txt = f" ({ret:+.1f}%)" if pd.notna(ret) else ""
        pills.append(f'<span class="pill" style="background:{rgba(T["critical"], .16)};'
                     f'color:var(--text-color)">⚡ Emergencia{ret_txt}</span>')
    if row["backfilled"]:
        pills.append(f'<span class="pill" style="background:{rgba(T["muted"], .16)};'
                     f'color:var(--text-color)">Backfill</span>')
    pills.append(f'<span class="pill" style="background:{rgba(T["muted"], .12)};'
                 f'color:var(--text-color)">Principal freio: {principal_freio(row)}</span>')
    st.markdown(
        f'<div class="verdict-card"><div>'
        f'<div class="v-label">Alocacao em {row["date"].date()}</div>'
        f'<div class="v-value">{row["alloc_pct"]:.1f}% BTC '
        f'<span class="v-cdi">/ {100 - row["alloc_pct"]:.1f}% CDI</span></div>'
        f'</div><div class="pills">{"".join(pills)}</div></div>',
        unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    for col, card in zip([f1, f2, f3], cartoes_fatores(row)):
        col.markdown(
            f'<div class="factor-card" style="border-left-color:{T[card["cor"]] if card["cor"] != "s1" else T["s1"]}">'
            f'<div class="factor-title">{card["titulo"]}</div>'
            f'<div class="factor-value">{card["valor"]}</div>'
            f'<div class="factor-sub">{card["leitura"]}</div></div>',
            unsafe_allow_html=True)

    st.markdown("")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Quanto cada fator cortou** — cenarios com a mesma previsao")
        if row["previsao"] <= 0:
            st.info("Previsao ≤ 0: o no-short zera a posicao, independentemente do "
                    "regime e da confianca — nao ha cenario alternativo com posicao.")
        else:
            plot(fig_e_se(row, T))
        st.caption(leitura_semana(row))
    with g2:
        st.markdown("**Semanas vizinhas** — o rebalance selecionado em destaque")
        plot(fig_vizinhas(sig_all, idx, T))

    res = resultado_janela(sig_all, idx)
    st.markdown("**E o que aconteceu depois?**")
    if res is None:
        st.caption("Janela ainda aberta — o retorno desta decisao so e conhecido no "
                   "proximo rebalance.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        fim_txt = f" ate {res['fim'].date()}" if res["fim"] is not None else ""
        r1.metric(f"Estrategia na janela{fim_txt}", f"{res['rs']:+.1f}%",
                  f"{res['vs_cdi']:+.1f} pp vs CDI")
        r2.metric("CDI na janela (aprox.)", f"{res['rc']:+.2f}%",
                  help="O benchmark justo: era a alternativa sem risco da semana")
        r3.metric("BTC na janela", f"{res['rb']:+.1f}%")
        r4.metric("vs 100% BTC", f"{res['vs_btc']:+.1f} pp", delta_color="off",
                  help="Referencia de captura, nao benchmark")
        st.caption(res["veredito"])

    ctx = contexto_features(ds, row["date"])
    if not ctx.empty:
        extremos = ctx.assign(dist=(ctx["Percentil historico"] - 50).abs())
        extremos = extremos[extremos["dist"] >= 35].sort_values("dist", ascending=False).head(4)
        if not extremos.empty:
            chips = " · ".join(
                f"**{r['Feature']}** no p{r['Percentil historico']:.0f}"
                for _, r in extremos.iterrows())
            st.markdown(f"Sinais de mercado em nivel extremo na data: {chips}")
        with st.expander("Contexto completo de mercado na data (features do modelo)"):
            st.dataframe(
                ctx, hide_index=True, width="stretch",
                column_config={"Percentil historico": st.column_config.ProgressColumn(
                    "Percentil historico", min_value=0, max_value=100, format="%.0f%%")})
            st.caption("Percentil expandindo (so usa historico ate a data). 0% = minima "
                       "historica, 100% = maxima. Nao e explicacao causal do modelo "
                       "(XGBoost usa as 32 features em conjunto) — e o retrato do "
                       "mercado no dia.")

# ═══ Aba 3: Visao mensal ═══
with tab_mes:
    sig_m = sig.copy()
    sig_m["mes"] = sig_m["date"].dt.strftime("%Y-%m")
    monthly = sig_m.groupby("mes").agg(
        alloc_media=("alloc_pct", "mean"),
        p_up_media=("p_up", "mean"),
        regime_dom=("regime", lambda s: s.mode().iloc[0]),
        n_emerg=("is_emergency", "sum"),
    ).reset_index()
    if monthly.empty:
        st.info("Sem rebalances no periodo selecionado.")
    else:
        hm = fig_heatmap_mensal(sig, T)
        if hm is not None:
            st.markdown("**Retornos mensais da estrategia** — azul positivo, "
                        "vermelho negativo; passe o mouse para comparar com CDI e BTC")
            plot(hm)
        meses_opts = list(monthly["mes"])[::-1]
        mes_sel = st.selectbox("Mes", meses_opts,
                               format_func=lambda m: f"{MESES[int(m[5:7])]}/{m[:4]}")
        plot(fig_monthly(monthly, mes_sel, T))
        g = sig_m[sig_m["mes"] == mes_sel]
        st.markdown(justificar_mes(g, sig_all["alloc_pct"].mean()))

# ═══ Aba 4: Dados & ajuda ═══
with tab_dados:
    st.subheader("Tabela completa dos sinais")
    st.dataframe(sig_all.drop(columns=["backfilled", "emerg_ret"]), hide_index=True,
                 width="stretch")
    st.subheader("Como o sinal e calculado")
    st.markdown(
        "- **Regressao (160 XGBoost)** preve o retorno de 3 dias (`previsao`).\n"
        "- **Classificador (160 XGBoost)** estima `P(up)`; a distancia de 0,5 vira o "
        "**fator de confianca** via sigmoide — incerteza corta a posicao (minimo ~0,5).\n"
        "- **Regime** (SMA 50/200) define o multiplicador **K**: BULL 60, MILD 30, BEAR 15.\n"
        "- `alocacao = clip(previsao x K x confianca, 0, 1)` — **sem short**: previsao "
        "negativa vira 0% BTC / 100% CDI.\n"
        "- Rebalance toda **sexta** apos o fechamento do candle diario, mais rebalance de "
        "**emergencia** quando |retorno diario| > 8% (executado pos-fechamento).\n"
        "- `retorno_strat` de cada linha cobre a janela ate o rebalance seguinte: "
        "`alocacao x BTC + (1 - alocacao) x CDI`. O CDI mostrado e reconstruido dessa "
        "identidade (aprox.).")
