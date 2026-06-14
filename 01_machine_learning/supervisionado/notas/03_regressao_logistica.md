# Regressão Logística

**Módulo:** 01 — Machine Learning  
**Data:** 2026-05-24  
**Fonte:** notas conceituais

---

A regressão linear prevê valores contínuos minimizando resíduos quadráticos — uma solução fechada com garantias teóricas do Teorema de Gauss-Markov (ver `01_regressao_linear.md`). Quando o alvo é binário — inadimplente ou não, fraude ou transação legítima, default ou não — o problema muda de natureza: queremos modelar uma **probabilidade**, e probabilidades têm uma restrição que a reta ignora, precisam ficar entre 0 e 1. A regressão logística resolve isso substituindo o critério quadrático pela maximização de verossimilhança e a previsão contínua pela função sigmoide — mesma estrutura linear, estimação fundamentalmente diferente. O modelo que classifica e-mail como spam ou detecta transações fraudulentas é, em sua última camada, uma regressão logística.

## Intuição

A regressão linear produz valores contínuos — e isso é exatamente o que queremos quando o alvo é um número como retorno, preço ou temperatura. Mas quando o alvo é binário — inadimplente ou não, fraude ou transação legítima, default ou não — o problema muda de natureza. Queremos modelar uma **probabilidade**, e probabilidades têm uma restrição que a reta linear ignora: precisam ficar entre 0 e 1.

Aplicar a regressão linear diretamente a um target binário produz dois problemas simultâneos. Primeiro, o modelo gera previsões fora do intervalo $[0, 1]$: para valores altos do preditor, a reta ultrapassa 1; para valores baixos, cai abaixo de 0. Isso não tem interpretação como probabilidade. Segundo, o relacionamento entre um preditor contínuo e a probabilidade de um evento raramente é linear — a probabilidade muda lentamente nas extremas e acelera no meio, seguindo uma forma em S.

![Regressão linear vs. logística em target binário](assets/01_logistica_intuicao.png)

Os dois painéis mostram os mesmos dados: pontos em 0 (classe negativa) e 1 (classe positiva) distribuídos ao longo de um preditor contínuo. À esquerda, a reta linear de OLS: ultrapassa 1 nos valores altos do preditor e cai abaixo de 0 nos baixos, tornando as previsões não interpretáveis como probabilidade. À direita, a curva logística: permanece dentro de $[0, 1]$ para qualquer valor do preditor e captura a aceleração central da probabilidade — a região onde o modelo tem mais incerteza.

O que resolve os dois problemas é a **função sigmoide** (ou logística). Ela toma qualquer valor real — positivo, negativo, muito grande, muito pequeno — e o comprime suavemente para o intervalo $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

A ideia da regressão logística é combinar a estrutura linear que já conhecemos com a sigmoide: calculamos o combinador linear $z = \beta_0 + \beta_1 x$ e passamos o resultado pela sigmoide para obter uma probabilidade. O modelo não prevê diretamente o valor 0 ou 1 — prevê a **probabilidade de pertencer à classe 1**.

---

## Definição formal

Dados $n$ pares $(x_i, y_i)$ com $y_i \in \{0, 1\}$, a regressão logística modela a probabilidade condicional de $y_i = 1$ dado $x_i$:

$$P(y_i = 1 \mid x_i) = \sigma(\beta_0 + \beta_1 x_i) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_i)}}$$

| Símbolo | Nome | Papel |
|---|---|---|
| $y_i$ | variável dependente binária | 0 ou 1 — o que queremos prever |
| $x_i$ | variável preditora | o que usamos para prever |
| $\beta_0$ | intercepto | parâmetro a estimar — desloca a curva horizontalmente |
| $\beta_1$ | coeficiente angular | parâmetro a estimar — controla a inclinação da curva |
| $\sigma(\cdot)$ | função sigmoide | transforma qualquer real em probabilidade em $(0, 1)$ |

Para entender o que o modelo está fazendo internamente, é útil reorganizar a equação. Se $p = P(y = 1 \mid x)$, então:

$$\frac{p}{1-p} = e^{\beta_0 + \beta_1 x}$$

O termo $p/(1-p)$ é a **odds** — a razão entre a probabilidade de ocorrência e a de não-ocorrência. Aplicando logaritmo em ambos os lados:

$$\log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x$$

O lado esquerdo é o **log-odds** (também chamado de **logit**): ele transforma uma probabilidade em $(0, 1)$ em um valor no intervalo $(-\infty, +\infty)$. O lado direito é exatamente o combinador linear da regressão linear. Isso revela a estrutura do modelo: a regressão logística é uma regressão linear dos log-odds — o que é linear não é a probabilidade, mas o log-odds.

---

## Fronteira de decisão

A equação do log-odds cria um objeto geométrico direto: a **fronteira de decisão**. Classificamos com $\hat{y} = 1$ quando $\hat{p} \geq 0.5$, o que equivale a log-odds $\geq 0$, portanto:

$$\hat{y} = 1 \iff \beta_0 + \beta_1 x \geq 0$$

O conjunto de pontos onde o log-odds é exatamente zero — $\beta_0 + \beta_1 x = 0$ — é a fronteira. Em uma dimensão é um ponto $x^* = -\beta_0/\beta_1$; com dois preditores é uma reta; com $p$ preditores é um **hiperplano** em $\mathbb{R}^p$.

![Fronteira de decisão em 1D e 2D](assets/01_logistica_fronteira.png)

À esquerda, a curva sigmoide com a fronteira marcada em laranja: o ponto $x^* = -\beta_0/\beta_1$ divide o eixo em duas regiões de classificação. À direita, o mesmo conceito em dois preditores: a fronteira deixa de ser um ponto e passa a ser uma reta — o hiperplano $\mathbf{x}^\top\boldsymbol{\beta} = 0$ — que separa as duas classes no espaço de $x_1$ e $x_2$. O gradiente de cores é a superfície de probabilidade; a linha laranja é onde ela cruza $\hat{p} = 0.5$.

Essa estrutura reaparece em SVMs (que também encontram um hiperplano separador) e é o que cada neurônio de uma rede neural implementa individualmente. É o motivo pelo qual a regressão logística é dita um **classificador linear**: não porque a probabilidade seja linear em $x$, mas porque a fronteira de decisão é.

---

## Estimação: por que não OLS e como funciona o MLE

O OLS (ver `01_regressao_linear.md`) minimiza a soma dos quadrados dos resíduos — uma função quadrática convexa com solução fechada. Quando o target é binário, aplicar esse critério não é apenas inconveniente: é conceitualmente errado. O OLS trata os resíduos como simétricos e de variância constante, mas quando $y \in \{0, 1\}$, a variância do erro depende de $p$ — ela é máxima quando $p = 0.5$ e cai a zero quando $p \to 0$ ou $p \to 1$. Além disso, minimizar quadrados sobre um target binário não leva à solução sigmoide — a estrutura da sigmoide não emerge naturalmente do critério quadrático.

A abordagem correta é a **Estimação por Máxima Verossimilhança (MLE)**. A ideia é perguntar: quais valores de $\beta_0$ e $\beta_1$ tornam os dados observados mais prováveis?

Para cada observação $i$, a probabilidade de observar $y_i$ dado o modelo é:

$$P(y_i \mid x_i, \beta_0, \beta_1) = p_i^{y_i}(1 - p_i)^{1 - y_i}$$

onde $p_i = \sigma(\beta_0 + \beta_1 x_i)$. Quando $y_i = 1$, o fator $(1-p_i)^0 = 1$ e sobra $p_i$ — a probabilidade de acerto. Quando $y_i = 0$, o fator $p_i^0 = 1$ e sobra $(1-p_i)$ — a probabilidade de acerto no negativo. A **verossimilhança total** supõe independência entre as observações e multiplica as probabilidades individuais:

$$\mathcal{L}(\beta_0, \beta_1) = \prod_{i=1}^n p_i^{y_i}(1 - p_i)^{1 - y_i}$$

Maximizar um produto de termos fracionários é numericamente instável. Aplicando o logaritmo — uma transformação monotônica que preserva o máximo — obtemos a **log-verossimilhança**:

$$\ell(\beta_0, \beta_1) = \sum_{i=1}^n \left[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \right]$$

onde $p_i = \sigma(\beta_0 + \beta_1 x_i)$. Maximizar $\ell$ é equivalente a minimizar o negativo dela — a **log-loss** (ou entropia cruzada binária), que reaparece na seção de avaliação como métrica de desempenho.

Diferentemente do OLS, essa função não tem solução fechada: a sigmoide dentro do logaritmo cria uma equação transcendental sem forma analítica. Os parâmetros são encontrados por otimização iterativa — tipicamente **gradiente descendente** ou métodos de segunda ordem como **Newton-Raphson**, que além do gradiente usam a curvatura da função (a hessiana) para dar passos maiores e mais certeiros, convergindo em poucas iterações. A convergência é garantida porque a log-verossimilhança é côncava — existe uma única solução global.

Para ver como a otimização funciona concretamente, vale calcular o gradiente da log-verossimilhança em relação ao vetor de parâmetros. Derivando $\ell$ em relação a $\boldsymbol{\beta}$ e usando a propriedade da sigmoide $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

$$\nabla_{\boldsymbol{\beta}}\,\ell = X^\top(y - \hat{p})$$

onde $y$ é o vetor de rótulos binários e $\hat{p}$ é o vetor de probabilidades previstas. O resultado espelha o que acontece no OLS — onde o gradiente do SSR também tem a forma $X^\top e$ (ver `01_regressao_linear.md`) — mas agora os "resíduos" são diferenças entre rótulos e probabilidades, não entre valores contínuos. Cada passo do gradiente descendente empurra os parâmetros na direção que reduz esses resíduos probabilísticos.

---

## Interpretando os coeficientes e avaliando sua significância

Com os parâmetros estimados em mãos, a curva ajustada é:

$$\hat{p}(x) = \frac{1}{1 + e^{-(\hat{\beta}_0 + \hat{\beta}_1 x)}}$$

Os coeficientes $\hat{\beta}_0$ e $\hat{\beta}_1$ não se interpretam diretamente como na regressão linear — a relação entre $x$ e $\hat{p}$ é não-linear. A interpretação natural passa pelos log-odds.

**$\hat{\beta}_1$ — o efeito do preditor:** para cada aumento de +1 unidade em $x$, o log-odds de $y = 1$ aumenta em $\hat{\beta}_1$. Em termos de odds, isso equivale a multiplicar a odds por $e^{\hat{\beta}_1}$: se $\hat{\beta}_1 = 0.5$, a odds aumenta em $e^{0.5} \approx 1.65$ — isto é, 65%. Esse fator $e^{\hat{\beta}_1}$ é o **odds ratio**, a forma mais natural de reportar o efeito em modelos logísticos.

**$\hat{\beta}_0$ — o intercepto:** controla o log-odds quando $x = 0$, ou seja, a posição da curva no eixo horizontal — onde a probabilidade de 0.5 ocorre. Mudá-lo desloca a curva para a esquerda ou direita sem alterar sua forma.

![Efeito de β₁ e β₀ na curva logística](assets/01_logistica_sigmoid_parametros.png)

Os dois painéis isolam os efeitos dos parâmetros. À esquerda, três curvas com $\beta_0 = 0$ e $\beta_1$ variando: curvas mais inclinadas correspondem a $\beta_1$ maior em valor absoluto — o modelo passa mais abruptamente de probabilidade baixa para alta. $\beta_1 < 0$ inverte a curva (probabilidade decresce com $x$). À direita, três curvas com $\beta_1$ fixo e $\beta_0$ variando: o ponto de cruzamento em $\hat{p} = 0.5$ se desloca horizontalmente — $\beta_0$ controla o limiar de decisão sem alterar a taxa de transição.

Uma observação importante sobre a conversão de coeficientes em probabilidade: o efeito de uma variação unitária em $x$ sobre $\hat{p}$ **não é constante** — ele depende de onde estamos na curva. O impacto é máximo na região central (em torno do ponto de inflexão, onde $\hat{p} \approx 0.5$) e diminui nas extremas. Por isso, reportar odds ratios é mais informativo do que tentar descrever efeitos em probabilidade sem especificar o ponto de avaliação.

Saber o valor do odds ratio é uma coisa — saber se ele é estatisticamente diferente de 1 é outra. Dois testes respondem a essa pergunta, e os dois testam a hipótese nula $H_0\text{: }\beta_j = 0$ (equivalentemente, odds ratio = 1, nenhum efeito do preditor):

O **teste de Wald** divide o estimador pelo seu erro padrão:

$$z = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}$$

Sob $H_0$, essa estatística segue aproximadamente uma normal padrão — a mesma lógica do teste $t$ da regressão linear (ver `01_regressao_linear.md`), mas usando a distribuição normal porque o MLE é um estimador de grandes amostras. Valores $|z| > 1.96$ correspondem a $p < 0.05$.

O **Teste da Razão de Verossimilhança (LRT)** compara diretamente a log-verossimilhança do modelo completo com a do modelo sem o preditor em questão:

$$\text{LRT} = 2\,(\ell_{\text{completo}} - \ell_{\text{reduzido}}) \;\sim\; \chi^2(1)$$

A multiplicação por 2 garante que a estatística siga uma qui-quadrado com 1 grau de liberdade. O LRT é preferível ao Wald quando a amostra é pequena ou quando os coeficientes são grandes em magnitude — nesses casos, o Wald perde precisão porque calcula o erro padrão localmente, enquanto o LRT compara o ajuste global dos dois modelos. Em amostras razoáveis, os dois costumam concordar; a diferença aparece nos limites: separação quase perfeita, preditores com alta correlação ou poucos eventos por preditor.

---

## Generalização: múltiplos preditores e mais de duas classes

A extensão para $p$ preditores segue a mesma lógica da regressão linear múltipla (ver `01_regressao_linear.md`). O combinador linear cresce para incluir todos os preditores:

$$\log\!\left(\frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$$

Em notação matricial: $\log\text{-odds} = \mathbf{x}^\top \boldsymbol{\beta}$, e a probabilidade é $\hat{p} = \sigma(\mathbf{x}^\top \boldsymbol{\beta})$. Cada $\hat{\beta}_j$ continua sendo um efeito parcial — a variação no log-odds associada a +1 unidade em $x_j$, mantendo os demais preditores fixos.

Quando o problema tem mais de duas classes ($k > 2$), a regressão logística binária não se aplica diretamente. A generalização natural é a **regressão logística multinomial**, que estima um vetor de coeficientes por classe e usa a função **softmax** para converter os $k$ combinadores lineares em probabilidades que somam 1:

$$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{x}^\top \boldsymbol{\beta}_c}}{\sum_{j=1}^k e^{\mathbf{x}^\top \boldsymbol{\beta}_j}}, \quad c = 1, \dots, k$$

A sigmoide binária é um caso especial do softmax com $k = 2$.

Uma abordagem alternativa ao softmax é o **One-vs-Rest (OvR)**: em vez de um modelo conjunto, treinam-se $k$ classificadores binários independentes — cada um aprende a separar uma classe das demais. Para classificar um novo ponto, aplica-se cada classificador e atribui-se a classe com maior $\hat{p}$.

![OvR vs Softmax — fronteiras de decisão em três classes](assets/01_logistica_ovr_softmax.png)

Os dois painéis mostram o mesmo conjunto de três classes. À esquerda (OvR): cada fronteira foi treinada de forma isolada; como os $k$ classificadores não se enxergam, as probabilidades das classes não somam 1 por construção — em regiões ambíguas, nenhum modelo tem confiança clara. À direita (Softmax): uma única otimização garante $\sum_c \hat{p}_c = 1$ para todo ponto, tornando as fronteiras geometricamente coerentes.

A diferença prática: OvR é o default do scikit-learn (`multi_class='ovr'`) e funciona com qualquer classificador binário, não apenas o logístico — o que o torna útil quando se quer usar SVMs ou Lasso em problemas de muitas classes. Softmax é preferível quando a calibração das probabilidades importa ou quando as classes não são mutuamente independentes, pois a normalização conjunta é garantida por construção.

---

## Medindo o ajuste

O $R^2$ da regressão linear mede a proporção da variância explicada — um conceito que não se transfere naturalmente para targets binários. Aqui o diagnóstico se faz em dois planos complementares: avaliação das **classificações** (depois de aplicar um limiar sobre $\hat{p}$) e avaliação das **probabilidades** diretamente.

**Matriz de confusão e métricas derivadas**

Para classificar, escolhemos um limiar $\tau$ (tipicamente 0.5) e definimos: $\hat{y} = 1$ se $\hat{p} \geq \tau$, e $\hat{y} = 0$ caso contrário. A **matriz de confusão** tabula as quatro combinações possíveis de predição e realidade:

|  | $\hat{y} = 1$ | $\hat{y} = 0$ |
|---|---|---|
| $y = 1$ | Verdadeiro Positivo (VP) | Falso Negativo (FN) |
| $y = 0$ | Falso Positivo (FP) | Verdadeiro Negativo (VN) |

Dela derivam as principais métricas:

$$\text{Precisão} = \frac{VP}{VP + FP}, \qquad \text{Recall} = \frac{VP}{VP + FN}, \qquad F_1 = \frac{2 \cdot \text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

A **precisão** responde: dos casos que o modelo sinalizou como positivo, quantos eram de fato positivos? O **recall** (ou sensibilidade) responde: dos casos que são realmente positivos, quantos o modelo capturou? Os dois estão em conflito — aumentar o recall (capturar mais positivos) geralmente reduz a precisão (sinalizar mais falsos positivos). O **F₁** é a média harmônica dos dois, penalizando desbalanços extremos.

A acurácia — proporção de classificações corretas — é uma métrica intuitiva, mas enganosa quando as classes são desbalanceadas. Um dataset em que 95% dos casos são negativos permite um modelo que sempre prevê 0 atingir 95% de acurácia sem aprender nada. Nesses cenários, precisão, recall e F₁ são mais informativos.

**Curva ROC e AUC**

A escolha do limiar $\tau$ é arbitrária — e diferentes problemas têm diferentes tolerâncias a falsos positivos e falsos negativos. A **curva ROC** (Receiver Operating Characteristic) avalia o modelo em **todos os limiares possíveis** simultaneamente, traçando o par (Taxa de Falsos Positivos, Recall) para cada valor de $\tau$:

$$\text{Taxa de FP} = \frac{FP}{FP + VN}, \qquad \text{Recall} = \frac{VP}{VP + FN}$$

O ponto $(0, 0)$ corresponde a $\tau = 1$ — o modelo nunca classifica ninguém como positivo. O ponto $(1, 1)$ corresponde a $\tau = 0$ — o modelo classifica todos como positivos. Um modelo aleatório caminha pela diagonal: para cada falso positivo capturado, captura um verdadeiro positivo na mesma proporção — sem nenhum poder discriminativo. Um modelo perfeito sobe diretamente para $(0, 1)$ e fica lá.

A **AUC** (Área sob a Curva ROC) resume a curva inteira em um único número entre 0 e 1. Interpretação: AUC é a probabilidade de que, dado um par aleatório (positivo, negativo), o modelo atribua maior $\hat{p}$ ao positivo. AUC = 0.5 equivale ao modelo aleatório; AUC = 1 é perfeição; valores acima de 0.7 são geralmente considerados úteis.

![Matriz de confusão e curva ROC](assets/01_logistica_avaliacao.png)

Os dois painéis mostram um mesmo modelo avaliado de formas complementares. À esquerda, a matriz de confusão com os quatro quadrantes coloridos: verde para acertos (VP e VN), vermelho para erros (FP e FN) — uma visualização imediata do tipo de erro predominante. À direita, a curva ROC azul acima da diagonal cinza (modelo aleatório): a área sombreada é a AUC. Quanto maior a área, mais o modelo discrimina positivos de negativos independentemente do limiar escolhido.

**Escolha do limiar**

O limiar padrão $\tau = 0.5$ é adequado quando as classes são equilibradas e os dois tipos de erro têm o mesmo custo. Em outros cenários, ele precisa ser ajustado.

Quando as **classes são desbalanceadas** — 5% de fraudes e 95% de transações legítimas, por exemplo — um modelo que prevê sempre 0 acerta 95% dos casos sem aprender nada, e $\tau = 0.5$ tende a reforçar esse comportamento. Nesses casos, a **curva Precisão–Recall** é mais informativa do que a ROC: ela amplifica a diferença entre modelos na região relevante (os poucos positivos) e ajuda a identificar o limiar que equilibra precisão e recall de acordo com a aplicação.

Ajustar $\tau$ modifica onde se corta a previsão, mas não muda o que o modelo aprendeu. Uma abordagem complementar — que atua durante o treino — é atribuir **pesos de classe** à log-loss, penalizando mais os erros na classe minoritária:

$$\ell_w = -\frac{1}{n}\sum_{i=1}^n \left[ w_1\, y_i \log \hat{p}_i + w_0\,(1 - y_i)\log(1 - \hat{p}_i) \right]$$

onde $w_1$ e $w_0$ são os pesos das classes positiva e negativa. Uma escolha comum é $w_c = n\,/\,(k \cdot n_c)$ — inversamente proporcional à frequência de cada classe — o `class_weight='balanced'` do scikit-learn. Ao contrário do ajuste de $\tau$, os pesos alteram os próprios $\hat{\boldsymbol{\beta}}$ estimados: o modelo aprende uma fronteira de decisão diferente, não apenas aplica um limiar diferente sobre a mesma curva.

![Fronteira de decisão sem e com pesos de classe](assets/01_logistica_pesos_classe.png)

Os dois painéis mostram o mesmo conjunto desbalanceado (90% negativos, 10% positivos). À esquerda, sem pesos: a fronteira (laranja) é empurrada para dentro da região da classe minoritária — o modelo "prefere" prever negativo para minimizar a perda. À direita, com `class_weight='balanced'`: a fronteira desloca-se em direção aos positivos, capturando mais da classe minoritária ao custo de mais falsos positivos. A diferença não está só no limiar — a posição da fronteira de decisão muda porque os próprios coeficientes mudam.

As duas abordagens são complementares: pesos de classe corrigem o modelo durante o treino; o limiar $\tau$ afina a decisão final sobre o modelo já treinado. Em desbalanceamento severo, combinar os dois tende a produzir os melhores resultados.

Quando os **custos de FP e FN são assimétricos** — não detectar uma fraude é mais custoso do que alertar um cliente legítimo — o limiar ótimo minimiza o custo esperado total. A condição de equilíbrio é classificar como positivo sempre que o custo esperado de errar for menor do que o de acertar: $\hat{p} \cdot C_{FN} \geq (1 - \hat{p}) \cdot C_{FP}$. Resolvendo para $\hat{p}$, o limiar que iguala os dois custos é:

$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

Um $C_{FN}$ alto empurra $\tau^*$ para baixo — o modelo aceita mais falsos positivos para não perder nenhum positivo real. Um $C_{FP}$ alto faz o oposto. Quando os custos não são conhecidos com precisão, a curva ROC permite visualizar o trade-off e escolher um ponto de operação de forma informada.

**Log-loss**

As métricas anteriores avaliam classificações — decisões binárias após aplicar um limiar. Mas às vezes o que importa é a qualidade das probabilidades em si, não das classificações. A **log-loss** avalia exatamente isso: penaliza previsões de probabilidade com alta confiança que se revelam erradas. É o negativo da log-verossimilhança normalizado pelo número de observações:

$$\text{log-loss} = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Quanto menor a log-loss, melhor. Um modelo que prevê $\hat{p} = 0.99$ para uma observação onde $y = 0$ é punido muito mais duramente do que um que prevê $\hat{p} = 0.6$ — a confiança equivocada é custosa. Log-loss igual a zero significa previsões perfeitas com probabilidade 1 ou 0 para cada classe.

**Calibração**

Uma propriedade que completa a avaliação probabilística é a **calibração**: um modelo bem calibrado produz $\hat{p} = 0.7$ para observações em que, empiricamente, cerca de 70% são positivas — as probabilidades numéricas correspondem às frequências reais. Log-loss e calibração são conceitos relacionados mas distintos: a log-loss penaliza a confiança equivocada de forma global; a calibração avalia especificamente se a escala de probabilidades está correta. É possível ter log-loss baixa sem calibração perfeita, e um modelo calibrado pode ter log-loss alta se suas previsões forem pouco decisivas.

A regressão logística é, em geral, bem calibrada quando suas premissas valem — consequência direta do MLE, que maximiza a log-verossimilhança e, por construção, ajusta as probabilidades às frequências observadas. Outros classificadores como SVM e árvores de decisão não têm essa propriedade e frequentemente requerem ajuste pós-treino. O *reliability diagram* verifica isso visualmente: divide as observações em faixas de $\hat{p}$ e plota a proporção observada de positivos em cada faixa — uma diagonal perfeita indica calibração ideal.

AUC, log-loss e calibração olham para facetas distintas das previsões, mas nenhuma responde à pergunta mais direta: "quão melhor o modelo é do que simplesmente prever sempre a taxa base de positivos?" Para isso existe o **pseudo-R² de McFadden**, que cria um paralelo com o $R^2$ da regressão linear:

$$R^2_{\text{McFadden}} = 1 - \frac{\ell_{\text{completo}}}{\ell_{\text{nulo}}}$$

onde $\ell_{\text{nulo}}$ é a log-verossimilhança de um modelo com apenas o intercepto — equivalente a prever sempre a proporção amostral de positivos, sem usar nenhum preditor. Quando o modelo não acrescenta nada, $\ell_{\text{completo}} = \ell_{\text{nulo}}$ e $R^2_{\text{McFadden}} = 0$. Quando o modelo é perfeito, $\ell_{\text{completo}} \to 0$ e $R^2_{\text{McFadden}} \to 1$.

Uma diferença importante em relação ao $R^2$ linear: a escala não é a mesma. Valores entre 0.2 e 0.4 já são considerados bons em modelos logísticos — não espere ver 0.9 como seria natural na regressão linear. Isso reflete a natureza do problema: prever uma probabilidade a partir de dados ruidosos é intrinsecamente mais difícil do que ajustar uma reta a dados contínuos.

**Diagnóstico de resíduos**

Na regressão linear, os resíduos $e_i = y_i - \hat{y}_i$ são a ferramenta central de diagnóstico — os gráficos de premissas giram em torno deles. Na logística, o resíduo bruto $(y_i - \hat{p}_i)$ existe, mas não tem as mesmas propriedades: sua variância não é constante (é máxima quando $\hat{p}_i = 0.5$), então comparar resíduos diretos entre observações é enganoso.

O equivalente padronizado é o **resíduo de deviance**:

$$d_i = \text{sign}(y_i - \hat{p}_i)\sqrt{-2\left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]}$$

O sinal preserva a direção do erro: positivo se o modelo subestimou $y_i$, negativo se superestimou. O interior do colchete — $y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)$ — é sempre negativo (logaritmo de probabilidade), então o sinal negativo na frente garante que a expressão sob a raiz seja sempre não negativa. Esse interior é exatamente a contribuição individual à log-loss total — somando $d_i^2$ sobre todas as observações recuperamos a **deviance** do modelo, análoga ao SSR da regressão linear. Observações com $|d_i|$ grande são potencialmente mal ajustadas ou influentes e merecem investigação.

---

## Premissas

A regressão logística é mais robusta do que a linear — não exige normalidade dos resíduos nem homocedasticidade — mas tem suas próprias condições de validade. Quando violadas, as estimativas podem ser viesadas, os erros padrão incorretos ou o modelo incapaz de convergir.

**1. Linearidade dos log-odds**

A relação entre cada preditor e o log-odds de $y = 1$ deve ser linear. Isso não significa que a relação com a probabilidade seja linear — ela nunca é — mas sim que a função $\log[p/(1-p)]$ é linear em $x$. Quando a relação verdadeira é curvilínea, o modelo sistematicamente subestima ou superestima a probabilidade em determinadas regiões do preditor.

Como detectar: gráfico do log-odds empírico (calculado em bins do preditor) contra os valores do preditor. Uma tendência curvilínea indica violação. Mitigação: adicionar termos polinomiais ou transformar o preditor.

![Linearidade dos log-odds — diagnóstico por bins](assets/01_logistica_premissa_logodds.png)

Em cada painel, os pontos azuis são os log-odds empíricos calculados em decis do preditor — a proporção de positivos em cada faixa convertida em log-odds. À esquerda, os pontos seguem a linha tracejada verde: o log-odds cresce linearmente com $x$, premissa satisfeita. À direita, os pontos descrevem uma curva em U: a linha tracejada vermelha ajusta um polinômio quadrático, evidenciando que o log-odds não é linear em $x$ — o modelo logístico simples estará mal especificado nesse cenário.

**2. Independência das observações**

Os erros devem ser não correlacionados entre observações — a mesma premissa da regressão linear. Em séries temporais ou dados agrupados (e.g., múltiplas observações por cliente), a violação infla a significância aparente dos coeficientes. Nesses casos, modelos de efeitos mistos ou erros padrão clusterizados são alternativas.

**3. Ausência de multicolinearidade**

Preditores fortemente correlacionados criam os mesmos problemas que na regressão linear múltipla (ver `01_regressao_linear.md`): erros padrão inflados, coeficientes instáveis e interpretação comprometida. O diagnóstico é idêntico — VIF calculado com os mesmos preditores em uma regressão linear auxiliar. A mesma escala de referência se aplica: VIF acima de 5 merece atenção.

**4. Tamanho amostral suficiente**

O MLE é um estimador de grandes amostras: suas propriedades teóricas (não-viés, eficiência, normalidade assintótica) só valem assintoticamente. A regra prática mais citada é de **pelo menos 10 eventos por preditor** — sendo "evento" a classe menos frequente. Com amostras pequenas, os coeficientes tendem a ser superestimados em magnitude, e os intervalos de confiança, enganosamente estreitos.

**5. Ausência de separação perfeita**

A separação perfeita ocorre quando existe um valor de $x$ (ou combinação de preditores) que separa as duas classes sem nenhum erro — todos os positivos acima de um limiar, todos os negativos abaixo. Nesse cenário, o MLE não converge: os coeficientes tendem ao infinito porque a log-verossimilhança continua aumentando conforme $|\hat{\beta}|$ cresce, sem atingir máximo finito. O sintoma típico são coeficientes absurdamente grandes com erros padrão enormes. A mitigação mais comum é **regularização L2** (equivalente a um prior normal sobre os coeficientes), que limita o crescimento dos parâmetros e restaura a convergência.

![Separação perfeita — sigmoide degenerando e dados separáveis](assets/01_logistica_premissa_separacao.png)

À esquerda, quatro sigmoides com $\beta_1$ crescente: conforme $\beta_1 \to \infty$, a curva se aproxima de um degrau — a transição de probabilidade 0 para 1 ocorre de forma instantânea em $x = 0$. Não há máximo finito da log-verossimilhança porque $\beta_1$ pode sempre crescer mais. À direita, um conjunto de dados com separação perfeita: todos os negativos (azul) estão à esquerda de $x = 0$ e todos os positivos (vermelho) à direita — a fronteira verde separa as classes sem nenhum erro. A sigmoide amarela mostra o modelo tentando se ajustar com $\beta_1$ muito grande: ele acerta perfeitamente os dados de treino, mas os coeficientes divergem e os erros padrão tornam-se inúteis.

---

## Parâmetros e hiperparâmetros

Os parâmetros do modelo logístico são exclusivamente os coeficientes $\hat{\boldsymbol{\beta}}$, estimados pelo MLE. Hiperparâmetros são valores que o praticante fixa antes do treino — não aprendidos dos dados. Estão distribuídos em seções anteriores desta nota:

| Hiperparâmetro | Seção | O que controla |
|---|---|---|
| $\tau$ | Escolha do limiar | ponto de corte sobre $\hat{p}$ para produzir $\hat{y} \in \{0,1\}$ |
| $w_0, w_1$ | Pesos de classe | reponderação da log-loss por classe durante o treino — altera os próprios $\hat{\boldsymbol{\beta}}$ |
| OvR vs Softmax | Generalização | estratégia de treinamento para $k > 2$ classes |
| $\lambda$ | Variações (abaixo) | intensidade da penalidade sobre $\lVert\boldsymbol{\beta}\rVert$ |

## Variações

Cada limitação do modelo base deu origem a uma extensão:

- **Regularização L2 (Ridge logístico)** — adiciona a penalidade $\lambda\lVert\boldsymbol{\beta}\rVert^2$ à função de custo, equivalente a impor um prior normal sobre os coeficientes. É a mitigação padrão para separação perfeita, mas também controla overfitting quando há muitos preditores ou amostras pequenas. Os coeficientes são contraídos em direção a zero, mas nenhum é exatamente zerado.
- **Regularização L1 (Lasso logístico)** — penaliza com $\lambda\lVert\boldsymbol{\beta}\rVert_1$ e produz coeficientes exatamente iguais a zero, fazendo seleção automática de variáveis. Útil quando a expectativa é de que poucos preditores sejam relevantes.
- **Elastic Net** — combina L1 e L2, controlando esparsidade e estabilidade ao mesmo tempo. Prático quando há grupos de preditores correlacionados.
- **Regressão logística com efeitos mistos** — quando as observações estão agrupadas (clientes dentro de agências, pacientes dentro de hospitais), adiciona termos aleatórios para capturar a estrutura hierárquica sem violar a premissa de independência.

Os detalhes de L1, L2 e Elastic Net são desenvolvidos em `02_regularizacao.md`.

---

## Conexão com outros tópicos

- **Regressão linear:** a logística compartilha a estrutura do combinador linear $\mathbf{x}^\top\boldsymbol{\beta}$ com a regressão linear — a diferença está no target (binário vs contínuo) e na estimação (MLE vs OLS). Nota `01_regressao_linear.md`.
- **Árvores e Gradient Boosting:** modelos não-lineares que não impõem a premissa de linearidade dos log-odds. Preferir logística quando interpretabilidade e calibração são essenciais (crédito, medicina, contextos regulatórios), quando os efeitos dos preditores são aproximadamente lineares nos log-odds, ou quando a amostra é pequena e a simplicidade ajuda a controlar o overfitting. Preferir árvores ou GBM quando as relações são não-lineares, há interações importantes entre variáveis, ou o conjunto inclui muitas features categóricas com muitos níveis. Em problemas com exigência de explicabilidade regulatória, a regressão logística ainda é o padrão dominante no mercado.
- **Regularização (Ridge, Lasso):** adicionam penalidade aos coeficientes para controlar overfitting — aplicável à logística, onde L2 também resolve separação perfeita. Nota `02_regularizacao.md`.
