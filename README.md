# Ciência de Dados em Riscos no Mercado de Capitais

> Portfólio de estudos e projetos aplicando técnicas de ciência de dados à gestão e análise de riscos em mercado de capitais.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Sobre o Projeto

Este repositório reúne estudos, análises e modelos voltados à **quantificação, modelagem e gestão de riscos financeiros**, com foco em mercado de capitais. O objetivo é demonstrar o uso de ferramentas de ciência de dados para resolver problemas reais enfrentados por profissionais de risco, quant analysts e gestores de portfólio.

Os projetos cobrem desde fundamentos estatísticos até modelos mais avançados, sempre com ênfase em aplicabilidade prática e rigor metodológico.

---

## Estrutura do Repositório

```
ciencia-dados/
│
├── notebooks/
│   ├── fundamentos/          # Estatística descritiva, distribuições, inferência
│   └── preparacao-dados/     # Pipeline de ML, feature engineering, leakage
│
├── data/                     # Dados utilizados nos projetos (não versionados)
├── requirements.txt          # Dependências do projeto
└── README.md
```

---

## Notebooks

### Fundamentos
Estatística descritiva, distribuições de probabilidade e inferência estatística aplicadas ao mercado de capitais.

| # | Notebook | Tópicos |
|---|---|---|
| 01 | [Tipos de variáveis](notebooks/fundamentos/01_tipos_de_variaveis_fundos_cvm.ipynb) | Categóricas, numéricas, ordinais — dataset CVM de fundos |
| 02 | [Medidas de posição e dispersão](notebooks/fundamentos/02_medidas_posicao.ipynb) | Média, mediana, moda, variância, desvio-padrão, quartis |
| 03 | [Médias, distribuições e probabilidade](notebooks/fundamentos/03_medias_distribuicoes_probabilidade.ipynb) | Tipos de média, distribuições amostrais, conceitos de probabilidade |
| 04 | [Probabilidade, distribuições, testes e modelos](notebooks/fundamentos/04_probabilidade_distribuicoes_testes_modelos.ipynb) | Distribuições discretas/contínuas, assimetria, curtose, testes estatísticos |
| 05 | [Outliers, correlação, causalidade e hipóteses](notebooks/fundamentos/05_outliers_correlacao_causalidade_hipoteses.ipynb) | Detecção de outliers, tipos de correlação, causalidade, testes de hipótese |
| 06 | [Regressão Linear — Parte 1: Beta CAPM](notebooks/fundamentos/06_regressao_linear_parte1.ipynb) | OLS, Beta CAPM, Alfa de Jensen, R², análise de resíduos, heteroscedasticidade |

### Preparação de Dados
Pipeline aplicado de pré-processamento e feature engineering para modelagem supervisionada.

| # | Notebook | Tópicos |
|---|---|---|
| 01 | [Preparação de dados e feature engineering](notebooks/preparacao-dados/01_preparacao_dados_feature_engineering.ipynb) | CRISP-DM, divisão treino/val/teste, data leakage, encoding, escalonamento, transformações, FE, seleção de variáveis |

---

## Próximos Passos

Áreas planejadas para o portfólio, conforme avanço dos estudos:

- [ ] **Risco de Mercado** — VaR histórico e paramétrico, CVaR, stress testing, backtesting
- [ ] **Gestão de Portfólio** — Fronteira eficiente, otimização de Markowitz, análise fatorial
- [ ] **Séries Temporais** — Modelagem ARIMA/GARCH, volatilidade estocástica
- [ ] **Derivativos** — Precificação de opções (Black-Scholes, Monte Carlo), gregas
- [ ] **Risco de Crédito** — Modelos de scoring, PD/LGD/EAD, matrizes de transição

---

## Tecnologias e Bibliotecas

```python
# Análise de dados
pandas · numpy · scipy

# Visualização
matplotlib · seaborn · plotly

# Modelagem estatística e financeira
statsmodels · arch · scikit-learn

# Dados financeiros
yfinance · pandas-datareader

# Notebooks
jupyter · nbformat
```

---

## Como Executar

**1. Clone o repositório**
```bash
git clone https://github.com/jescoelho/ciencia-dados.git
cd ciencia-dados
```

**2. Crie um ambiente virtual**
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Inicie o Jupyter**
```bash
jupyter notebook
```

---

## Referências e Base Teórica

- Hull, J. C. — *Options, Futures, and Other Derivatives*
- Jorion, P. — *Value at Risk: The New Benchmark for Managing Financial Risk*
- McNeil, A. J., Frey, R., Embrechts, P. — *Quantitative Risk Management*
- Fabozzi, F. J. — *Handbook of Portfolio Management*
- Documentações: [CVM](https://www.gov.br/cvm/), [B3](https://www.b3.com.br/), [Basileia III](https://www.bcb.gov.br/)

---

## Autor

**Jéssica Coelho**
Estudante de Ciência de Dados com foco em Riscos e Mercado de Capitais

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin)](https://www.linkedin.com/in/jessicacoelhoalves)
[![GitHub](https://img.shields.io/badge/GitHub-Perfil-181717?logo=github)](https://github.com/jescoelho)

---

## Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais informações.
