# Roteiro de Apresentação — Análise 05 (5 min)

**Referência:** [Árvore de Decisão para Default de Cartão de Crédito](05_arvore_decisao_default.md)

---

## Problema de negócio

O problema é de crédito: dado o histórico de pagamentos de um cliente de cartão, conseguimos prever se ele vai calotear no mês seguinte?

O dataset tem 30 mil clientes de Taiwan com seis meses de histórico — de abril a setembro de 2005. O target é binário: pagou ou não pagou em outubro. A base tem desequilíbrio leve: 22% dos clientes inadimpliram, 78% pagaram normalmente.

Do ponto de vista do negócio, a previsão não serve para punir — serve para agir antes da perda. Identificar inadimplentes com um mês de antecedência permite renegociar dívida, bloquear limite ou revisar concessão enquanto ainda há margem de manobra. O critério de sucesso não é acertar todos os casos, mas capturar inadimplentes suficientes para que a intervenção valha o custo operacional.

---

## Tratamento das variáveis

O dataset bruto tem 25 colunas. A preparação passou por quatro etapas.

**Limpeza.** A coluna de ID foi removida — não é preditor. As variáveis de escolaridade e estado civil tinham 68 registros com categorias fora do dicionário, 0,2% da base, que foram agrupados na categoria "outros" já existente. As variáveis de status de pagamento tinham dois valores não documentados — mantidos como estão, porque a taxa de default deles é estável e consistente com o restante da série.

**Análise de sinal.** Antes de qualquer decisão de remoção, calculei a correlação de cada variável com o target. O resultado foi claro: as seis variáveis de status de pagamento lideram, com correlação entre 0,19 e 0,32. O limite de crédito aparece em seguida com −0,15 — quanto maior o limite, menor o risco. Todo o bloco de saldo devedor mensal concentra-se no fundo da lista, com correlação abaixo de 0,02, e três das seis variáveis não atingem sequer significância estatística.

**Remoção do bloco de fatura.** Além do sinal quase nulo, as seis variáveis de saldo têm correlação entre si de até 0,95 — são praticamente a mesma variável repetida seis vezes. Remover o bloco inteiro não perde informação relevante. No lugar, criei `util_rate`, a razão entre a fatura do mês mais recente e o limite de crédito, que tem quatro vezes mais sinal que qualquer variável de saldo original.

**Feature engineering.** Três variáveis adicionais foram criadas: `n_meses_atraso` — quantos meses o cliente ficou em atraso nos seis meses, que se tornou o preditor mais forte do modelo; `total_pago` — soma dos valores pagos no período; e `tendencia_pay` — diferença entre o status do mês mais recente e o mais antigo, capturando se o comportamento melhorou ou piorou ao longo do semestre.

Divisão final: 80% treino, 20% teste, estratificado pelo target. Resultado: 21 preditores. Escalonamento não foi aplicado — árvores são invariantes a essa transformação.

---

## Modelo

O modelo é uma árvore de decisão CART. Em vez de uma equação, ela segmenta os clientes por regras binárias sucessivas — "se o cliente esteve em atraso em dois ou mais meses, vai para o ramo de alto risco; dentro desse ramo, se o mês mais recente também está em atraso, vai para a folha de risco máximo" — e cada folha entrega uma probabilidade de default. O resultado é um conjunto de regras que um analista de crédito consegue ler e validar sem precisar de score numérico.

Dois controles foram aplicados para evitar que a árvore decore o treino. O primeiro é uma pré-poda que impede partições com menos de 20 registros. O segundo é uma pós-poda por custo-complexidade: a árvore cresce completa e depois é progressivamente simplificada, com validação cruzada de 5 folds escolhendo o ponto de corte ótimo. Para compensar o desequilíbrio de classes, inadimplentes receberam peso 3,5 vezes maior no treino. O modelo final tem profundidade 5 e 11 folhas.

---

## Métricas de validação

No conjunto de teste — 6 mil clientes que o modelo nunca viu:

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,7685 |
| Gini | 0,537 |
| KS | 0,410 |
| F1-Score | 0,526 |
| Gap treino–teste | 0,0086 |

O **gap de 0,0086** é o primeiro número a olhar: a diferença entre o desempenho no treino e no teste é mínima, o que confirma que o modelo generalizou e não decorou os dados.

O **KS** — estatística de Kolmogorov-Smirnov — mede a maior separação possível entre dois grupos: de um lado, a distribuição dos scores dos adimplentes; do outro, a dos inadimplentes. Quanto maior o KS, mais o modelo consegue afastar esses dois grupos, e melhor ele ranqueia clientes por risco. Aqui ficou em 0,41. O mercado de crédito usa como referência: abaixo de 0,20 é fraco, entre 0,20 e 0,40 é aceitável, acima de 0,40 é bom. Estamos no limiar inferior da faixa "bom". O **Gini de 0,537** é a mesma informação do AUC em outra escala — a preferida em comitês de risco.

Na prática operacional: com o corte padrão de 0,5, o modelo captura **57% dos inadimplentes** antes de qualquer perda. Reduzindo o corte para 0,38, esse número sobe para **73%** — ao custo de acionar mais clientes que acabariam pagando. A escolha entre esses pontos depende da capacidade da equipe de cobrança e do custo relativo de cada erro.

Para confirmar que esse AUC de 0,77 é o teto do modelo e não falta de ajuste, rodei uma otimização automática com 150 tentativas variando seis parâmetros simultaneamente. Toda configuração mais complexa aumentou a diferença entre treino e teste sem melhorar o resultado no conjunto de teste. O modelo com 11 folhas permaneceu como o melhor ponto. Para ir além desse número, seria preciso mudar de tipo de modelo.
