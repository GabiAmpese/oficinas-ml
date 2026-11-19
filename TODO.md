# TODO.md — Retorno Inteligente (troca de óleo)

Ordem de execução. Referência: `specs.md`. Segue as 11 etapas da aula 4.

## Etapa 0 — Ambiente
- [x] Estrutura de pastas, `requirements.txt`, `README.md`, `.gitignore`
- [x] Copiar as 4 planilhas para `data/brutos/`

## Etapa 1 — Problema (specs seção 2)
- [x] Pergunta da oficina e custo de cada erro escritos no specs e no relatório

## Etapa 2 — Dados: `src/preparar_dados.py`
- [x] Ler as 4 planilhas (aba Visitas), unificar colunas, detectar unidade (km ou h)
- [x] Regra única de limpeza do serviço (troca de óleo sim/não)
- [x] Marcar `frota` a partir das observações (Cirocar) e do tipo de cliente
- [x] Gerar `data/visitas.csv`
- [x] Gerar `data/intervalos.csv` com features sem vazamento e `situacao`
- [x] Imprimir resumo: linhas, veículos, intervalos válidos/perdidos/erro por fonte, `isna().sum()`

## Etapa 3 — Separação (em `src/treinar.py`)
- [x] Ordenar por `data_fim`; teste = últimos 20%
- [x] `KFold(5, shuffle, random_state=42)` no treino

## Etapa 4 — Réguas
- [x] Mediana do treino
- [x] Regra do fabricante (unidades / taxa de uso, limitada pelos meses)
- [x] Repete o último intervalo
- [x] MAE/RMSE/MAPE das três na validação cruzada

## Etapa 5 — Preparo: `src/modelos.py`
- [x] `ColumnTransformer` com imputação + indicador, log1p nas colunas com valores enormes, padronização, one-hot
- [x] Conjuntos de features `basico` e `completo`

## Etapa 6 — Modelos
- [x] `LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`
- [x] Alvo alternativo `delta_unid` convertido em dias

## Etapa 7 — Ajuste fino
- [x] Optuna: alpha do Ridge, conjunto de features, alvo (dias ou log)
- [x] Gráfico tentativa × melhor até agora
- [x] Escolha do modelo final pelo MAE de validação (régua pode vencer)

## Etapa 8 — Prova final
- [x] Reajustar no treino inteiro; abrir o teste uma única vez
- [x] Placar (tabela + barras com a régua tracejada)
- [x] Faixa ±2s e cobertura no teste
- [x] Gráficos previsto × real e resíduo × previsto

## Etapa 9 — Explicação
- [x] `permutation_importance` no teste, gráfico de barras

## Etapa 10 — Decisão
- [x] `cross_val_predict` no treino; curva custo × antecedência; melhor antecedência
- [x] Mostrar a mudança com custos diferentes

## Etapa 11 — Simulação
- [x] Função `simular` e painel com controles no app

## Artefatos
- [x] `models/modelo_retorno.joblib` (pipeline completo) e `models/metricas.json`

## App Streamlit: `app/app.py`
- [x] Lista da semana com previsão, faixa, status e origem de cada número
- [x] Simulador "e se"
- [x] Sobre o modelo: placar, métricas, importâncias, decisão, limitações
- [x] Avisos fixos

## Testes: `tests/`
- [x] Preparação: detecção de unidade, regra de serviço, intervalo perdido, features sem vazamento (datas)
- [x] Modelos: réguas com resultado calculado à mão, split temporal sem vazamento, artefato carrega e prevê, categoria desconhecida não quebra, faixa cobre o esperado

## Execução ponta a ponta
- [x] `preparar_dados.py` → `treinar.py` → `pytest` → `streamlit run app/app.py`

## Relatório: `relatorio/relatorio.md`
- [x] As 11 etapas, com um parágrafo de texto respondendo à pergunta de cada uma
- [x] Tabela de objetivos alcançados e não alcançados
- [x] Roteiro da demonstração
- [ ] Revisão do grupo e autorização por escrito das oficinas e da prefeitura
