# Probabilidade e Estatística

Dados são incertos por natureza: preços oscilam, medições têm erro, o futuro não é determinístico. Probabilidade é a linguagem que quantifica essa incerteza; estatística é o conjunto de ferramentas para extrair conclusões confiáveis apesar dela.

Em IA, todo modelo aprende estimando distribuições, minimizando funções de perda derivadas da teoria da informação e sendo avaliado com testes estatísticos. Sem essa base, os modelos funcionam como caixas-pretas — usáveis, mas não compreensíveis.

## Progressão

| Camada | Tópico | Conceitos-chave | Para entender em IA |
|--------|--------|-----------------|---------------------|
| 1 | Linguagem dos dados | Distribuições, medidas de posição e dispersão, correlação | Todo modelo é uma afirmação sobre distribuições |
| 2 | Raciocínio sob incerteza | Probabilidade condicional, Teorema de Bayes, variáveis aleatórias, esperança e variância | Pivô conceitual: de "qual é o valor?" para "qual é a distribuição sobre os valores?" |
| 3 | Como modelos aprendem | MLE, MAP, viés-variância, intervalos de confiança | Regressão linear, regularização e a maioria dos modelos supervisionados são derivações diretas |
| 4 | Avaliação com rigor | Testes de hipótese, p-valor, poder estatístico | Comparação de modelos e A/B testing |
| 5 | Fundamento de deep learning | Entropia, divergência KL, informação mútua | Cross-entropy loss, VAEs e modelos de difusão são derivados diretamente daqui |
| 6 | Incerteza em modelos | Inferência bayesiana, prior/posterior, distribuições conjugadas | Visão unificada de modelos probabilísticos; pré-requisito para Gaussian Processes e BNNs |

## Notas

[Medidas de posição e dispersão](notas/01_medidas_posicao_dispersao.md) · [Distribuições de probabilidade](notas/02_distribuicoes.md) · [Correlação e dependência](notas/03_correlacao.md) · Probabilidade e Bayes · MLE e MAP · Testes de hipótese · Teoria da informação · Inferência bayesiana
