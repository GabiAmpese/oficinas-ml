# Critérios de aceite — Retorno Inteligente

Propostos com base nas aulas de MVP, spec-driven development e qualidade. Situação **proposta para verificação**; nenhum teste adicional é declarado aprovado sem execução.

| ID | Necessidade / regra | Critério verificável | Evidência ou teste sugerido |
|---|---|---|---|
| CA01 | Usar históricos reais | Preparação aceita quatro planilhas de fontes identificadas e produz visitas e intervalos sem gerar registros sintéticos para treino | `python src/preparar_dados.py`; conferência da contagem e procedência |
| CA02 | Prever retorno, não vencimento técnico | Interface identifica a saída como retorno estimado e apresenta aviso de que não substitui a recomendação do mecânico | Inspeção de `app/app.py` e demonstração na Lista da semana |
| CA03 | Respeitar o passado de cada exemplo | Features de um intervalo são calculadas apenas com leituras disponíveis até `data_ini` | `tests/test_preparar.py` e testes manuais de datas |
| CA04 | Auditabilidade temporal | Checagem estrita usa apenas intervalos de treino que terminam antes do início do bloco validado | `tests/test_qualidade_adicional.py::test_dobras_estritas_nao_usam_resultados_futuros` |
| CA05 | Comparar ML e baseline | Aplicação permite alternar entre a régua escolhida e o modelo `modelo_ml` salvo | Interface e teste existente do artefato |
| CA06 | Não retreinar ao abrir o painel | `app/app.py` apenas carrega `joblib` e `metricas.json` | Inspeção e teste de carregamento |
| CA07 | Suportar veículo sem ritmo de uso | Régua personalizada utiliza fallback do treino ou limite em meses sem falhar | `test_sem_historico_usa_limite_meses` |
| CA08 | Evitar exposição indevida | Protótipo não inclui dados diretamente identificadores e não é publicado com acesso livre a dados históricos | Revisão manual dos arquivos e permissões de publicação |
| CA09 | Reproduzir resultados | Testes locais passam após instalar dependências e executar preparação/treino | `python executar_tudo.py` seguido de `python -m pytest -q` |
| CA10 | Comunicar incerteza | Tela exibe a faixa ±2s junto à previsão e explica a base pequena | Inspeção do Streamlit e relatório |

## Critérios de não-aceite
- Dizer que o modelo prevê **a necessidade mecânica** de troca de óleo.
- Afirmar que a validação por `data_fim` elimina todo risco temporal.
- Publicar dados brutos identificáveis ou históricos sem controlar acesso.
- Alterar seleção do modelo após inspecionar os 12 casos do teste final.

## Pendências de implantação (não implementadas)
Autenticação, autorização por oficina, auditoria de acesso, armazenamento e backup, atualização operacional de registros, gerenciamento de consentimentos/bases legais quando aplicável e monitoração em produção.
