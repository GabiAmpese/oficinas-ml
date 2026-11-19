# Padrões do curso (Nanodegree IA & ML, Atitus) aplicados neste projeto

Extraído dos notebooks das aulas 1 a 5 e 9 a 13. Onde cada padrão aparece no código está entre parênteses.

## As 11 etapas da aula 4 (roteiro do specs e do relatório)
1. Problema: a pergunta da oficina e o custo de cada erro (specs §2).
2. Dados: contagens, `isna().sum()`, o que está faltando (`preparar_dados.py`).
3. Separação: prova final trancada; validação cruzada em 5 dobras no treino (`treinar.py`).
4. Baseline: a régua que o modelo precisa vencer (`ReguaMediana`, `ReguaFabricante`, `ReguaUltimoIntervalo`, `ReguaMediaDasReguas`).
5. Preparo: uma receita por tipo de coluna, `SimpleImputer(add_indicator=True)`, `log1p` em colunas com valores enormes, `OneHotEncoder(handle_unknown="ignore")` (`construir_preparo`).
6. Modelo: regressão linear e regularizadas.
7. Ajuste fino: Optuna com `TPESampler(seed=42)`, função que devolve a média da validação cruzada, `log=True` no alpha. Nunca olha o teste.
8. Prova final: uma única vez; comparar nota do teste com a da validação.
9. Explicação: `permutation_importance`, barras, "importância não é causa".
10. Decisão: regra de negócio escolhida pelo custo com previsões fora da dobra (`melhor_antecedencia`).
11. Simulação: função "e se" (`simular`) e painel com controles no app.
Fechamento: `joblib.dump` do pipeline inteiro; projeto em arquivos (`modelos.py`, `treinar.py`, `app.py`).

## Aula 5 (avaliação temporal)
- "O corte é no tempo. Nada de sortear linhas." Teste = últimos 20% por data.
- Várias réguas sem modelo antes de qualquer modelo; placar com a régua tracejada (`placar.png`).
- Lições repetidas no relatório: ter modelo não garante nada; o hábito tem que entrar no modelo (empilhamento das réguas); o que se sabe do negócio vale mais que o modelo.

## Aulas 1 a 3
- MAE e RMSE na unidade do negócio (dias), MAPE em %, R².
- Faixa de previsão com resíduo típico s e margem 2s, cobertura conferida (aula 1).
- Sobreajuste demonstrado: conjunto `completo` sem regularização estourou (aula 2).
- Padronizar antes de regularizar, só com o treino; `RidgeCV`, `LassoCV`, `ElasticNetCV` com `logspace(-2, 3, 60)`.
- Limiar/regra escolhida por custo (aula 3, exercício 10).
- Todo experimento termina com um parágrafo em texto respondendo à pergunta.

## Aulas 9 a 13 (LLMs)
- Não entram no modelo. Justificam a escolha por ML e a regra "uma demonstração que funciona não prova nada; o conjunto de casos com resposta conhecida é o mínimo" (nossos testes com valores calculados à mão).
