# Aprendizado Supervisionado

No aprendizado supervisionado, o modelo aprende a partir de exemplos rotulados: para cada entrada, existe uma resposta correta conhecida. O objetivo é aprender a relação entre entrada e saída de forma que generalize para dados novos.

É chamado "supervisionado" porque o processo de treino é guiado por esse conjunto de respostas corretas — como um aluno que aprende com gabarito. Regressão (prever um valor contínuo) e classificação (prever uma categoria) são os dois tipos fundamentais de problema.

## Análises

| # | Pergunta de negócio | Método |
|---|---------------------|--------|
| [01](analises/01_predicao_choque_itub4.md) | Dado queda de 3% no Ibovespa, quanto cai o ITUB4? | OLS, diagnóstico de premissas |
| [02](analises/02_regularizacao_multifator_itub4.md) | Quais fatores de risco explicam o retorno em excesso do ITUB4? | Lasso · Ridge · Elastic Net, Fama-French 5 fatores |

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

Regressão linear e logística · Regularização
