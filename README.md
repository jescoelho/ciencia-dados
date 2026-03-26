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

## Tópicos Abordados

| Área | Descrição |
|------|-----------|
| **Risco de Mercado** | VaR, CVaR, stress testing, backtesting de modelos de risco |
| **Gestão de Portfólio** | Fronteira eficiente, otimização de Markowitz, análise fatorial |
| **Séries Temporais** | Modelagem ARIMA/GARCH, volatilidade estocástica |
| **Derivativos** | Precificação de opções (Black-Scholes, Monte Carlo), gregas |
| **Risco de Crédito** | Modelos de scoring, PD/LGD/EAD, matrizes de transição |
| **Análise Estatística** | Testes de hipótese, distribuições de cauda pesada, copulas |

---

## Estrutura do Repositório

```
ciencia-dados/
│
├── notebooks/                  # Jupyter Notebooks por tema
│   ├── risco-de-mercado/
│   ├── gestao-de-portfolio/
│   ├── series-temporais/
│   ├── derivativos/
│   └── risco-de-credito/
│
├── src/                        # Módulos e funções reutilizáveis
│   ├── risk/                   # Cálculo de métricas de risco
│   ├── portfolio/              # Otimização e análise de portfólio
│   └── utils/                  # Funções auxiliares e helpers
│
├── data/                       # Dados utilizados nos projetos
│   ├── raw/                    # Dados brutos (não versionados)
│   └── processed/              # Dados tratados
│
├── reports/                    # Relatórios e visualizações exportadas
└── requirements.txt            # Dependências do projeto
```

---

## Projetos em Destaque

> *Os projetos serão adicionados progressivamente conforme o desenvolvimento dos estudos.*

### Em breve
- [ ] **VaR Histórico e Paramétrico** — Comparação de metodologias para estimação do Value at Risk
- [ ] **Otimização de Portfólio com Markowitz** — Fronteira eficiente e alocação ótima de ativos brasileiros
- [ ] **Modelagem de Volatilidade com GARCH** — Aplicação em ativos da B3
- [ ] **Backtesting de Estratégias de Risco** — Validação de modelos com dados históricos
- [ ] **Stress Testing e Cenários Extremos** — Análise de cauda e simulações de choque

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
git clone https://github.com/seu-usuario/ciencia-dados.git
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

**José Coelho**
Estudante de Ciência de Dados com foco em Riscos e Mercado de Capitais

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin)](https://www.linkedin.com/in/seu-perfil)
[![GitHub](https://img.shields.io/badge/GitHub-Perfil-181717?logo=github)](https://github.com/seu-usuario)

---

## Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais informações.
