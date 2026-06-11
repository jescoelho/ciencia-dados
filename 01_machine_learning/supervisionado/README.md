# Aprendizado Supervisionado

No aprendizado supervisionado, o modelo aprende a partir de exemplos rotulados: para cada entrada, existe uma resposta correta conhecida. O objetivo é aprender a relação entre entrada e saída de forma que generalize para dados novos.

É chamado "supervisionado" porque o processo de treino é guiado por esse conjunto de respostas corretas — como um aluno que aprende com gabarito. Regressão (prever um valor contínuo) e classificação (prever uma categoria) são os dois tipos fundamentais de problema.

## Progressão

| Camada | Tópico | Conceitos-chave | Para entender em IA |
|--------|--------|-----------------|---------------------|
| 1 | Regressão linear | OLS, mínimos quadrados, diagnóstico de premissas | Base de todos os modelos lineares; deriva Ridge, Lasso e redes neurais |
| 2 | Regularização | Ridge, Lasso, Elastic Net, validação cruzada temporal | Controle de overfitting; seleção de variáveis em modelos de risco |
| 3 | Regressão logística | Função sigmoide, log-odds, limiar de decisão | Porta de entrada para classificação e redes neurais |
| 4 | Árvores e ensembles | Árvores de decisão, Random Forest, bagging | Modelos não-lineares sem necessidade de padronização |
| 5 | Gradient Boosting | XGBoost, LightGBM, boosting sequencial | Estado da arte em dados tabulares; amplamente usado em crédito e risco |
| 6 | SVM | Hiperplanos, margens, kernel trick | Fundamento teórico de margens e separabilidade — base conceptual para classificadores modernos |

## Notas

[Regressão linear e logística](notas/01_regressao_linear_logistica.md) · [Regularização](notas/02_regularizacao.md) · [Árvores e ensembles](notas/04_arvores_ensembles.md) · Gradient Boosting · SVM

## Análises

| # | Título | Tema |
|---|--------|------|
| 01 | [Predição de choque — ITUB4](analises/01_predicao_choque_itub4.md) | Regressão linear simples, CAPM |
| 02 | [Modelo multifator com regularização — ITUB4](analises/02_regularizacao_multifator_itub4.md) | Ridge, Lasso, Elastic Net |
| 03 | [Probabilidade de default em cartão de crédito](analises/03_default_cartao_credito.md) | Regressão logística, credit scoring |
| 04 | [Predição de default — LendingClub (2008–2011)](analises/04_lendingclub_oot.md) | Regressão logística, validação out-of-time |
