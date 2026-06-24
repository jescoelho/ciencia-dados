# Agente ReAct — Pricing de Opções ITUB4

Este estudo demonstra o padrão ReAct aplicado a *quantitative pricing*: um agente recebe um objetivo em linguagem natural, decide quais ferramentas chamar, observa os resultados e produz uma análise de precificação de opção sobre ITUB4.

O objetivo ilustra onde o LLM agrega valor num fluxo quant: as ferramentas calculam com precisão (Black-Scholes não erra), mas a decisão sobre qual janela de volatilidade usar, como interpretar o regime atual e o que recomendar ao gestor de risco requer raciocínio contextual — o que um script fixo não produz.

> **Script:** [01_agent_pricing_opcoes_itub4.py](01_agent_pricing_opcoes_itub4.py)  
> **Nota de referência:** [ReAct — Raciocínio e Ação em Agentes](../notas/01_react.md)

---

## Configuração do agente

**Objetivo recebido:** *"Precifique uma call OTM sobre ITUB4 com vencimento em 30 dias e avalie o regime de volatilidade atual."*

**Ferramentas disponíveis:**

| Ferramenta | O que executa |
|---|---|
| `buscar_preco_spot(ticker)` | Preço de fechamento mais recente via yfinance (fallback sintético) |
| `calcular_volatilidade_historica(ticker, janela)` | Vol histórica anualizada em janelas de 30d e 90d |
| `precificar_black_scholes(S, K, T, r, sigma, tipo)` | Preço teórico + Greeks completos (Δ, Γ, θ, ν) |

**Parâmetros fixos:** Selic 10,65% a.a. · Strike OTM +5% sobre spot · Vencimento 30 dias corridos

---

## Execução do ciclo ReAct

### Iteração 1 — preço spot

```
[Pensamento] Preciso do preço spot atual do ITUB4 para
             definir o strike e os parâmetros do modelo.

[Ação] buscar_preco_spot('ITUB4.SA')
[Observação] spot=R$28.0  retorno_1d=-2.68%  data=2026-06-23
```

O agente identifica corretamente que sem o spot não pode definir strike nem calcular moneyness — primeira decisão de sequenciamento.

### Iteração 2 — volatilidade histórica

```
[Pensamento] Spot em R$28.0. Usarei strike R$29.4 (OTM +5%).
             Preciso da vol histórica — calcularei janelas de
             30d e 90d para avaliar o regime atual de volatilidade.

[Ação] calcular_volatilidade_historica('ITUB4.SA', janela=30)
[Observação] vol_30d=17.14%  vol_90d=19.38%
```

O agente solicita **duas janelas simultaneamente** — decisão que um script fixo não tomaria sem instrução explícita. A comparação entre janelas informa o regime.

### Iteração 3 — Black-Scholes e Greeks

```
[Pensamento] Vol 30d (17.14%) está comprimida vs 90d (19.38%).
             Usarei 30d como entrada — mais representativa do regime atual.

[Ação] precificar_black_scholes(S=28.0, K=29.4, T=0.1190, r=0.1065, sigma=0.1714)
[Observação] preco=R$0.2794  delta=0.2806  theta=-0.0086/dia  vega=0.0326/1%vol
             prob_exercício=26.1%
```

A escolha de usar vol 30d (e não 90d) é uma decisão de raciocínio — o agente justifica explicitamente.

---

## Resultado

```
Ativo     : ITUB4  |  Data: 2026-06-23
Spot      : R$ 28.00  |  Strike: R$ 29.40  |  Venc.: 30 dias
Vol 30d   : 17.14% a.a.  |  Vol 90d: 19.38% a.a.
Taxa Selic: 10.65% a.a.

── Resultado Black-Scholes ──────────────────────────────────
  Tipo           : CALL OTM (+5.0% do spot)
  Prêmio teórico : R$ 0.2794
  Prob. exercício: 26.1%

── Greeks ──────────────────────────────────────────────────
  Δ Delta    0.2806  → a cada R$1 de alta no spot, prêmio +R$ 0.2806
  Γ Gamma  0.203501  → aceleração do delta (convexidade)
  θ Theta   -0.0086  → perda de R$ 0.0086/dia por decaimento
  ν Vega     0.0326  → ganho de R$ 0.0326 por +1% de vol
```

---

## Interpretação

**Moneyness e probabilidade de exercício**

Delta de 0,28 e probabilidade de exercício de 26,1% confirmam opção OTM com strike 5% acima do spot. O mercado precifica baixa chance de ITUB4 superar R$ 29,40 em 30 dias — coerente com um ativo de vol ~17% a.a. e prazo curto.

**Regime de volatilidade**

Vol 30d (17,14%) abaixo da vol 90d (19,38%) sinaliza regime de baixa volatilidade recente — o ativo está se movendo menos do que sua média histórica de 90 dias. Para um comprador de call, esse ambiente é favorável: prêmios mais baratos. Para um vendedor, margem de segurança menor contra movimentos bruscos.

**Decaimento temporal**

Theta de R$ 0,0086/dia parece pequeno, mas em 10 dias sem movimento favorável o prêmio perde ~R$ 0,086 — 30% do valor inicial (R$ 0,28). Vencimento curto penaliza posições compradas em opções OTM que não se movem.

**O que falta para decisão de trading**

O modelo Black-Scholes usa volatilidade histórica como proxy da implícita. Na prática, a comparação correta é:

$$\text{vol implícita de mercado} \quad \text{vs} \quad \text{vol histórica 30d} = 17{,}14\%$$

Se vol implícita > 17,14%: mercado precifica mais risco do que o histórico sugere — opção potencialmente cara.  
Se vol implícita < 17,14%: prêmio abaixo do risco histórico — potencial oportunidade de compra.

---

## O que o agente fez que um script não faria

| Decisão | Script fixo | Agente ReAct |
|---|---|---|
| Sequência das ferramentas | Hardcoded | Inferida a partir do objetivo |
| Escolha da janela de vol | Parâmetro fixo | Justificada com base na comparação 30d vs 90d |
| Strike | Parâmetro fixo | Calculado dinamicamente sobre o spot real |
| Interpretação dos Greeks | Não produz | Contextualiza theta em termos de perda percentual do prêmio |
| Recomendação qualitativa | Não produz | Aponta o que falta para decisão de trading (vol implícita) |

A diferença não está nos cálculos — Black-Scholes é determinístico. Está na capacidade do agente de encadear os passos corretos, escolher parâmetros com justificativa e produzir análise contextualizada sem instruções passo a passo.
