# 🤖 Estudos em Inteligência Artificial

> Documentação pessoal da minha jornada de estudos em IA — dos fundamentos matemáticos aos modelos mais avançados e projetos práticos.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🧠 O que é Inteligência Artificial?

Inteligência Artificial (IA) é o campo que estuda como fazer máquinas executarem tarefas que exigiriam inteligência humana, como reconhecer padrões, tomar decisões e usar linguagem. Seu principal subcampo é o **Machine Learning**, no qual modelos aprendem padrões a partir de dados em vez de seguir regras programadas manualmente; dentro dele, o **Deep Learning** usa redes neurais profundas para problemas complexos como visão e linguagem. A partir dessas bases surgem áreas aplicadas como **NLP e LLMs** (processamento de linguagem e grandes modelos de linguagem), **Visão Computacional**, **Aprendizado por Reforço** e a **IA Generativa e Agentes**. Sustentando tudo estão os **fundamentos matemáticos** (álgebra linear, cálculo, probabilidade) e disciplinas de engenharia e governança como **MLOps** e **IA Responsável**. Este repositório percorre justamente esse caminho — dos fundamentos aos tópicos avançados e projetos práticos.

---

## 🗺️ Roadmap e Progresso

| #  | Módulo                        | Status          |
|----|-------------------------------|-----------------|
| 00 | Fundamentos Matemáticos       | 🔄 Em andamento |
| 01 | Machine Learning              | 🔄 Em andamento |
| 02 | Deep Learning                 | ⏳ Pendente     |
| 03 | NLP e LLMs                    | ⏳ Pendente     |
| 04 | Visão Computacional           | ⏳ Pendente     |
| 05 | Aprendizado por Reforço       | ⏳ Pendente     |
| 06 | IA Generativa e Agentes       | ⏳ Pendente     |
| 07 | Grafos e GNNs                 | ⏳ Pendente     |
| 08 | MLOps                         | ⏳ Pendente     |
| 09 | IA Responsável                | ⏳ Pendente     |
| 10 | Projetos Práticos             | 🔄 Contínuo     |

---

## 📒 Conteúdo disponível

### Módulo 01 — Machine Learning · Supervisionado

| Notebook | Tema | Técnicas |
|---|---|---|
| [Previsão de Choque — ITUB4](01_machine_learning/supervisionado/analises/01_predicao_choque_itub4.md) | CAPM via OLS: dado queda de 3% no Ibovespa, quanto cai o ITUB4? | Regressão linear, erros HC3, RESET, Jarque-Bera, Breusch-Pagan, Durbin-Watson, Cook's Distance, beta rolling, IP bootstrap |

---

## 🔧 Como usar este repositório

- Cada módulo tem seu próprio `README.md` com objetivos, referências e pré-requisitos.
- `notas/` contém resumos conceituais em Markdown.
- `notebooks/` contém implementações práticas em Jupyter.
- `10_projetos/` contém aplicações end-to-end que integram vários módulos.

```
estudos-ia/
├── 00_fundamentos/          # Álgebra linear, cálculo, probabilidade, Python
├── 01_machine_learning/     # Supervisionado, não supervisionado, feature engineering
├── 02_deep_learning/        # Redes neurais, CNN, Transformers, generativos
├── 03_nlp/                  # Representação de texto, LLMs, tarefas clássicas
├── 04_visao_computacional/  # Detecção, segmentação, visão-linguagem
├── 05_aprendizado_por_reforco/
├── 06_ia_generativa_agentes/ # RAG e agentes
├── 07_grafos_e_gnns/
├── 08_mlops/
├── 09_ia_responsavel/
└── 10_projetos/             # Projetos práticos end-to-end
```

---

## 📦 Ambiente

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## ✍️ Autora

**Jéssica Coelho**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin)](https://www.linkedin.com/in/jessicacoelhoalves)

---

## 📄 Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais informações.
