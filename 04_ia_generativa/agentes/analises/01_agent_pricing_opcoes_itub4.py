"""
Agente ReAct — Pricing de Opções ITUB4
=======================================
Demonstra o padrão ReAct aplicado a quantitative pricing:
ferramentas executam cálculos reais (Black-Scholes + Greeks);
dados obtidos via yfinance com fallback sintético calibrado.

Para usar com API real: substituir o bloco PLANO por chamadas ao
Anthropic SDK com function calling (ver nota 01_react.md).

Requer: pip install yfinance scipy numpy
"""

import warnings
import numpy as np
from scipy.stats import norm
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ── modo de dados ──────────────────────────────────────────────────────────────

def _dados_sinteticos(ticker: str, n_dias: int = 95) -> "pd.DataFrame":
    """
    Fallback: série sintética calibrada com parâmetros históricos do ITUB4.
    Spot inicial ~R$28, vol diária ~1.6% (≈25.4% a.a.), drift ~0.
    """
    import pandas as pd
    rng = np.random.default_rng(seed=42)
    S0, mu, sigma_d = 28.0, 0.0002, 0.016
    retornos = rng.normal(mu, sigma_d, n_dias)
    precos = S0 * np.cumprod(1 + retornos)
    datas = [date.today() - timedelta(days=n_dias - i) for i in range(n_dias)]
    return pd.DataFrame({"Close": precos}, index=pd.to_datetime(datas))


def _buscar_yfinance(ticker: str, period: str):
    """Tenta yfinance; retorna (DataFrame | None, fonte)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None, "yfinance retornou vazio"
        return hist, "yfinance"
    except Exception as e:
        return None, str(e)


# ── ferramentas reais ──────────────────────────────────────────────────────────

def buscar_preco_spot(ticker: str) -> dict:
    hist, fonte = _buscar_yfinance(ticker, "5d")
    if hist is None:
        hist = _dados_sinteticos(ticker)
        fonte = "sintético (calibrado ITUB4 histórico)"

    preco = round(float(hist["Close"].iloc[-1]), 2)
    retorno_1d = round(float(
        (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100
    ), 3) if len(hist) >= 2 else 0.0

    return {
        "ticker": ticker,
        "preco_spot": preco,
        "retorno_1d_%": retorno_1d,
        "data": str(hist.index[-1].date()),
        "fonte": fonte,
    }


def calcular_volatilidade_historica(ticker: str, janela: int = 30) -> dict:
    hist, fonte = _buscar_yfinance(ticker, "90d")
    if hist is None:
        hist = _dados_sinteticos(ticker, n_dias=95)
        fonte = "sintético (calibrado ITUB4 histórico)"

    retornos = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    vol_30 = float(retornos.tail(janela).std()) * np.sqrt(252)
    vol_90 = float(retornos.std()) * np.sqrt(252)

    return {
        "janela_dias": janela,
        "vol_anualizada_%": round(vol_30 * 100, 2),
        "vol_anualizada_90d_%": round(vol_90 * 100, 2),
        "fonte": fonte,
    }


def precificar_black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, tipo: str = "call"
) -> dict:
    """
    S     — preço spot
    K     — strike
    T     — tempo até vencimento em anos
    r     — taxa livre de risco anual (contínua)
    sigma — volatilidade anual
    tipo  — 'call' ou 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == "call":
        preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        prob  = norm.cdf(d2)
    else:
        preco = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        prob  = norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo == "call" else -d2)
    ) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    pct_otm   = (K / S - 1) * 100 if tipo == "call" else (S / K - 1) * 100
    moneyness = "ATM" if abs(pct_otm) < 2 else ("OTM" if pct_otm > 0 else "ITM")

    return {
        "tipo": tipo.upper(),
        "preco_teorico": round(float(preco), 4),
        "moneyness": moneyness,
        "pct_otm": round(float(pct_otm), 2),
        "prob_exercicio_%": round(float(prob) * 100, 1),
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 6),
        "theta_dia": round(float(theta), 4),
        "vega_1pct": round(float(vega), 4),
    }


# ── parâmetros do cenário ──────────────────────────────────────────────────────

TICKER          = "ITUB4.SA"
SELIC           = 0.1065      # Selic vigente jun/2026 — atualizar se necessário
VENCIMENTO_DIAS = 30
STRIKE_OTM_PCT  = 0.05        # call OTM 5% acima do spot

FERRAMENTAS = {
    "buscar_preco_spot":              buscar_preco_spot,
    "calcular_volatilidade_historica": calcular_volatilidade_historica,
    "precificar_black_scholes":        precificar_black_scholes,
}

# ── loop ReAct ─────────────────────────────────────────────────────────────────
# Ferramentas executam cálculos reais. Pensamentos e resposta final reproduzem
# o que um LLM produziria — substituir por client.messages.create() com
# function calling para usar com API Anthropic real.

SEP = "─" * 62

print(SEP)
print("  AGENTE ReAct — PRICING DE OPÇÕES ITUB4")
print(SEP)

# iteração 1 — preço spot
print("\n[Pensamento] Preciso do preço spot atual do ITUB4 para")
print("             definir o strike e os parâmetros do modelo.")
print(f"\n[Ação] buscar_preco_spot('{TICKER}')")
spot = FERRAMENTAS["buscar_preco_spot"](ticker=TICKER)
S = spot["preco_spot"]
print(f"[Observação] spot=R${S}  retorno_1d={spot['retorno_1d_%']}%"
      f"  data={spot['data']}  fonte={spot['fonte']}")

# define strike
K = round(S * (1 + STRIKE_OTM_PCT), 2)

# iteração 2 — volatilidade histórica
print(f"\n[Pensamento] Spot em R${S}. Usarei strike R${K} (OTM +5%).")
print("             Preciso da vol histórica — calcularei janelas de")
print("             30d e 90d para avaliar o regime atual de volatilidade.")
print(f"\n[Ação] calcular_volatilidade_historica('{TICKER}', janela=30)")
vol = FERRAMENTAS["calcular_volatilidade_historica"](ticker=TICKER, janela=30)
v30 = vol["vol_anualizada_%"]
v90 = vol["vol_anualizada_90d_%"]
print(f"[Observação] vol_30d={v30}%  vol_90d={v90}%  fonte={vol['fonte']}")

# iteração 3 — Black-Scholes
T     = VENCIMENTO_DIAS / 252
sigma = v30 / 100
regime = "comprimida" if v30 < v90 else "elevada"

print(f"\n[Pensamento] Vol 30d ({v30}%) está {regime} vs 90d ({v90}%).")
print("             Usarei 30d como entrada do modelo — mais representativa")
print("             do regime atual. Precificando call e calculando Greeks.")
print(f"\n[Ação] precificar_black_scholes(S={S}, K={K}, T={T:.4f},"
      f" r={SELIC}, sigma={sigma:.4f}, tipo='call')")
bs = FERRAMENTAS["precificar_black_scholes"](S=S, K=K, T=T, r=SELIC, sigma=sigma)
print(f"[Observação] preco=R${bs['preco_teorico']}  delta={bs['delta']}"
      f"  theta={bs['theta_dia']}/dia  vega={bs['vega_1pct']}/1%vol"
      f"  prob_exercício={bs['prob_exercicio_%']}%")

# resposta final
perda_10d = round(abs(bs['theta_dia']) * 10, 4)
print(f"\n[Pensamento] Delta de {bs['delta']} e prob. de exercício de"
      f" {bs['prob_exercicio_%']}% confirmam")
print(f"             opção OTM. Theta de R${abs(bs['theta_dia'])}/dia é")
print(f"             relevante: em 10 dias sem movimento o prêmio perde")
print(f"             ~R${perda_10d} apenas por decaimento. Análise completa.")

print(f"\n{SEP}")
print("  RESPOSTA FINAL")
print(SEP)
print(f"""
Ativo     : ITUB4  |  Data: {spot['data']}
Spot      : R$ {S}  |  Strike: R$ {K}  |  Venc.: {VENCIMENTO_DIAS} dias
Vol 30d   : {v30}% a.a.  |  Vol 90d: {v90}% a.a.
Taxa Selic: {SELIC*100:.2f}% a.a.

── Resultado Black-Scholes ──────────────────────────────────
  Tipo           : CALL {bs['moneyness']} ({bs['pct_otm']:+.1f}% do spot)
  Prêmio teórico : R$ {bs['preco_teorico']}
  Prob. exercício: {bs['prob_exercicio_%']}%

── Greeks ──────────────────────────────────────────────────
  Δ Delta  {bs['delta']:>8.4f}  → a cada R$1 de alta no spot, prêmio +R$ {bs['delta']:.4f}
  Γ Gamma  {bs['gamma']:>8.6f}  → aceleração do delta (convexidade)
  θ Theta  {bs['theta_dia']:>8.4f}  → perda de R$ {abs(bs['theta_dia']):.4f}/dia por decaimento
  ν Vega   {bs['vega_1pct']:>8.4f}  → ganho de R$ {bs['vega_1pct']:.4f} por +1% de vol

── Interpretação ───────────────────────────────────────────
  O delta de {bs['delta']:.2f} reflete ~{bs['prob_exercicio_%']}% de probabilidade de
  exercício: o mercado precifica baixa chance de ITUB4 superar
  R$ {K} em {VENCIMENTO_DIAS} dias a partir de R$ {S}.

  Theta de R$ {abs(bs['theta_dia']):.4f}/dia é relevante no vencimento curto:
  em 10 dias sem movimento, o prêmio perde ~R$ {perda_10d}
  apenas por decaimento temporal.

  Vol 30d {"abaixo" if regime == "comprimida" else "acima"} da vol 90d \
({v30}% vs {v90}%) sugere regime
  {"de baixa" if regime == "comprimida" else "de alta"} volatilidade recente.
  Comparar com vol implícita de mercado revelaria se o prêmio
  está caro (vol impl. > {v30}%) ou barato (vol impl. < {v30}%).
""")
