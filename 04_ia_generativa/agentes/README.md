# Sistemas de Agentes

Agentes são sistemas onde um LLM não apenas gera texto, mas decide ações, usa ferramentas e persiste estado entre passos. Em vez de responder a uma pergunta, um agente raciocina sobre um objetivo, executa ações (buscas, chamadas de API, código), observa os resultados e itera. Este submódulo cobre os padrões arquiteturais que tornam isso possível, do ciclo básico de raciocínio-ação à orquestração de múltiplos agentes.

## Progressão

| Camada | Tópico | Conceitos-chave | Nível |
|--------|--------|-----------------|-------|
| 1 | ReAct | Ciclo raciocínio-ação-observação, prompting estruturado | Fundamento |
| 2 | Tool use e function calling | Definição de ferramentas, schema JSON, chamadas estruturadas | Fundamento |
| 3 | Memória episódica e semântica | Memória de curto e longo prazo, vetores, recuperação | Fundamento |
| 4 | Orquestração multi-agente | Agentes paralelos, passagem de contexto, coordenação | Aplicado |
| 5 | Frameworks | LangChain, LlamaIndex, AutoGen — abstrações e trade-offs | Cobertura mínima |
| 6 | Harness Engineering | Avaliação de agentes, scaffolding, rastreabilidade | Cobertura mínima |
| 7 | Avaliação e segurança | Métricas de agentes, prompt injection, limites de autonomia | Aplicado |
| 8 | Planejamento de longo horizonte | Tree of Thoughts, MCTS+LLM, decomposição de tarefas | Avançado |

## Notas

[ReAct](notas/01_react.md) · Tool use e function calling · Memória episódica e semântica · Orquestração multi-agente · Frameworks · Harness Engineering · Avaliação e segurança · Planejamento de longo horizonte

## Análises

| # | Título | Tema |
|---|--------|------|
| 01 | [Agente ReAct — Pricing de Opções ITUB4](analises/01_agent_pricing_opcoes_itub4.md) | Black-Scholes, Greeks, regime de volatilidade |
