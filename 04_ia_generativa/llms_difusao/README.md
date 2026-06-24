# LLMs e Difusão

Modelos generativos aprendem a distribuição dos dados em vez de aprender a fronteira entre classes. Isso os torna capazes de criar exemplos novos — textos, imagens, cenários sintéticos — por amostragem da distribuição aprendida. Este submódulo percorre as arquiteturas que tornaram isso possível, da base conceitual aos modelos modernos e suas formas de aplicação.

## Progressão

| Camada | Tópico | Conceitos-chave | Nível |
|--------|--------|-----------------|-------|
| 1 | Modelos generativos e discriminativos | P(x), P(x,y), P(y\|x), amostragem | Fundamento |
| 2 | Autoencoders | Encoder, decoder, espaço latente, reconstrução | Fundamento |
| 3 | VAEs | Prior, posterior, reparametrização, ELBO | Fundamento |
| 4 | GANs | Gerador, discriminador, jogo minimax | Fundamento |
| 5 | Mecanismo de atenção | Query, Key, Value, atenção escalar produto | Fundamento |
| 6 | Transformers | Multi-head attention, positional encoding, encoder-decoder | Fundamento |
| 7 | LLMs | Causal vs. mascarado, técnicas de decoding, alucinação | Fundamento |
| 8 | Modelos de difusão | Difusão direta, denoising, DDPM | Fundamento |
| 9 | Prompting e APIs | Zero-shot, few-shot, OpenAI, Anthropic, Google | Aplicado |
| 10 | RAG | Recuperação aumentada, chunking, reranking | Aplicado |
| 11 | GraphRAG | Grafos de conhecimento, recuperação estruturada | Cobertura mínima |

## Notas

[Modelos generativos e discriminativos](notas/01_modelos_generativos_discriminativos.md) · Autoencoders · VAEs · GANs · Mecanismo de atenção · Transformers · LLMs · Modelos de difusão · Prompting e APIs · RAG · GraphRAG
