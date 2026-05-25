# Instruções para este repositório

## Padrão de escrita para notas de estudo (`/notas/*.md`)

Toda nota segue um **arco narrativo linear**: cada seção prepara a próxima. O leitor nunca deve se perguntar "por que estou vendo isso agora?".

### Estrutura obrigatória

```
Intuição → Definição formal → Estimação/Mecanismo → Interpretação → Generalização → Avaliação → Premissas
```

Adapte os títulos ao conteúdo, mas mantenha essa progressão. Exemplo para regressão linear:

| Seção | Pergunta que responde |
|---|---|
| Intuição | O que esse modelo faz, geometricamente? |
| Definição formal | Como isso é escrito matematicamente? |
| Estimação | Como encontramos os parâmetros? |
| Interpretação | O que os parâmetros significam na prática? |
| Generalização | Como isso se estende a casos mais complexos? |
| Avaliação | Como sabemos se o modelo está bom? |
| Premissas | Quando isso vale — e quando deixa de valer? |

### Regras de narrativa

- **Um conceito, um lugar.** Nunca explique o mesmo conceito em duas seções diferentes. Se precisou voltar a um conceito, a estrutura está errada — reorganize.
- **Costure as seções.** A última frase de uma seção deve criar a pergunta que a próxima responde. Evite seções que começam do zero sem conectar com o que veio antes.
- **Explique antes de afirmar.** Se uma propriedade é afirmada (ex: "o SSR é convexo"), a justificativa vem no mesmo parágrafo — não num header separado depois.
- **Defina imediatamente após formalizar.** Todo símbolo novo é definido na linha seguinte à equação em que aparece pela primeira vez — nunca antes, nunca parágrafos depois.
- **Headers não substituem transições.** Um header não é uma frase de transição. O texto sob cada header deve conectar com o anterior em prosa, não apenas listar fatos.
- **Profundidade progressiva.** Intuição em linguagem simples → formalização → detalhes. Nunca o inverso.

### Gráficos

- Todo gráfico tem legenda de leitura em texto logo abaixo, explicando o que cada elemento representa e **qual conclusão o leitor deve tirar**.
- Gráficos da seção de intuição: sem equações sobrepostas.
- Paleta escura padrão: fundo `#0d1117`, pontos `#58a6ff`, destaque `#f78166`, auxiliar `#3fb950`.

### O que não fazer

- Não crie headers no formato "Por que X?" seguidos de "Por que Y?" sobre o mesmo tema — isso fragmenta o raciocínio em FAQs em vez de construir uma narrativa.
- Não adicione conteúdo em patches sem reler o fluxo completo da seção afetada.
- Não repita na seção "Conexão com outros tópicos" o que já foi dito no corpo da nota.
