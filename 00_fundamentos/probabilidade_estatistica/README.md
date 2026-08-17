# Probabilidade e Estatística

Dados são incertos por natureza: preços oscilam, medições têm erro, o futuro não é determinístico. Probabilidade é a linguagem que quantifica essa incerteza; estatística é o conjunto de ferramentas para extrair conclusões confiáveis apesar dela.

Em IA, todo modelo aprende estimando distribuições, minimizando funções de perda derivadas da teoria da informação e sendo avaliado com testes estatísticos. Sem essa base, os modelos funcionam como caixas-pretas — usáveis, mas não compreensíveis.

## Progressão

| Camada | Tópico | Conceitos-chave | Para entender em IA |
|--------|--------|-----------------|---------------------|
| 1 | Linguagem dos dados | Distribuições, medidas de posição e dispersão, correlação | Todo modelo é uma afirmação sobre distribuições |
| 2 | Raciocínio sob incerteza | Probabilidade condicional, Teorema de Bayes, variáveis aleatórias, esperança e variância | Pivô conceitual: de "qual é o valor?" para "qual é a distribuição sobre os valores?" |
| 3 | Como modelos aprendem | MLE, MAP, viés-variância, intervalos de confiança | Regressão linear, regularização e a maioria dos modelos supervisionados são derivações diretas |
| 4 | Avaliação com rigor | Validação cruzada, testes de hipótese, p-valor, poder estatístico | Comparação de modelos e A/B testing |
| 5 | Fundamento de deep learning | Entropia, divergência KL, informação mútua | Cross-entropy loss, VAEs e modelos de difusão são derivados diretamente daqui |
| 6 | Incerteza em modelos | Inferência bayesiana, prior/posterior, distribuições conjugadas | Visão unificada de modelos probabilísticos; pré-requisito para Gaussian Processes e BNNs |

## Notas

[Medidas de posição e dispersão](notas/01_medidas_posicao_dispersao.md) · [Distribuições de probabilidade](notas/02_distribuicoes.md) · [Correlação e dependência](notas/03_correlacao.md) · [Probabilidade e Bayes](notas/04_probabilidade_bayes.md) · [MLE e MAP](notas/05_mle_map.md) · [Validação cruzada](notas/06_validacao_cruzada.md) · Testes de hipótese · Teoria da informação · Inferência bayesiana

## Análises

| # | Título | Tema |
|---|--------|------|
| 01 | [Tipos de Variáveis — Fundos de Investimento (CVM)](analises/01_tipos_de_variaveis_fundos_cvm.ipynb) | Variáveis qualitativas e quantitativas, dados públicos da CVM |
| 02 | [Medidas de Posição e Dispersão — Fundos de Investimento (CVM)](analises/02_medidas_posicao.ipynb) | Média, mediana, desvio-padrão, dados da CVM |
| 03 | [Tipos de Média, Distribuições e Probabilidade — Ibovespa](analises/03_medias_distribuicoes_probabilidade.ipynb) | Médias, distribuições de probabilidade |
| 04 | [Probabilidade, Distribuições, Testes Estatísticos e Modelos — Ibovespa](analises/04_probabilidade_distribuicoes_testes_modelos.ipynb) | Testes estatísticos aplicados a modelagem |
| 05 | [Outliers, Correlação, Causalidade e Hipóteses Estatísticas](analises/05_outliers_correlacao_causalidade_hipoteses.ipynb) | Outliers, correlação vs. causalidade, Ibovespa e S&P 500 |
