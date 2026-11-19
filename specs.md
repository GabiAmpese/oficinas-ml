# specs.md — Retorno Inteligente: previsão de retorno à oficina para troca de óleo

Nanodegree de IA & ML (Atitus). Especificação técnica da parte prática.
Substitui o tema anterior (motoristas de aplicativo), abandonado por falta de dados reais.

## 1. Tipo de solução

**Solução de Machine Learning**: um sistema que ingere o histórico de visitas de veículos a oficinas mecânicas, treina um modelo de regressão para prever **em quantos dias cada veículo volta para a próxima troca de óleo**, compara essa previsão com a regra fixa do fabricante e entrega à oficina uma lista semanal de clientes a contatar.

A estrutura segue as 11 etapas da aula 4 do curso (problema, dados, separação, baseline, preparo, modelo, ajuste fino, prova final, explicação, decisão, simulação) e a regra de ouro da aula 5: o corte entre treino e teste é no tempo.

## 2. Problema (etapa 1)

Pergunta da oficina: **"Quais clientes devo contatar esta semana para a troca de óleo?"**

Hoje o lembrete usa uma regra igual para todos ("10.000 km ou 12 meses"). Dois carros do mesmo modelo têm ritmos de uso muito diferentes. O sistema aprende o ritmo de cada veículo e prevê a data de retorno.

Custo de cada erro (usado na etapa de decisão):
- Lembrar **tarde demais** (o carro já voltou, ou trocou em outro lugar): perda do serviço.
- Lembrar **cedo demais**: contato à toa, cliente ignora.

## 3. Dados (etapa 2)

### 3.1 Fontes (todas reais, pseudonimizadas)

| Fonte | O que é | Como foi coletado |
|---|---|---|
| Cirocar | Oficina em Tapejara/RS e região | Extraído das NF-e emitidas (nov/2020 a nov/2026). Placa e cliente viraram códigos |
| Ampese | Oficina Mecânica Ampese, São Domingos do Sul/RS | Planilha preenchida pela oficina a partir de fichas e etiquetas |
| Lica | Oficina vizinha (Lica / Matheus Ampese) | Ordens de serviço impressas do sistema, transcritas |
| Prefeitura | Frota de máquinas da Prefeitura de São Domingos do Sul | Relatórios do sistema de frotas; medida em **horas de horímetro** |

Nenhum nome, placa, CPF ou telefone entra no projeto. Cada veículo tem um código (V001, F103...), único dentro da fonte.

### 3.2 Formato de entrada (`data/brutos/*.xlsx`, aba `Visitas`)

Uma linha por visita. Colunas obrigatórias: Oficina, Código do carro, Marca, Modelo, Ano do carro, Data da visita, Km no painel (ou horas), Serviço feito. Colunas úteis: Motor, Combustível, Tipo de óleo, Viscosidade, Intervalo recomendado (km e meses), Uso do carro, Carro de aplicativo?, Observações.

### 3.3 Tabela unificada de visitas (`data/visitas.csv`)

Gerada por `src/preparar_dados.py`. Colunas: `fonte, oficina, veiculo, marca, modelo, ano, motor, combustivel, data, leitura, unidade (km|h), servico, troca_oleo (bool), tipo_oleo, viscosidade, intervalo_rec_unid, intervalo_rec_meses, uso, aplicativo, frota, observacoes`.

Regra de limpeza do serviço: é troca de óleo toda linha cuja descrição contém "troca de óleo" (com ou sem filtros). "Troca de filtro de óleo" sozinha e demais serviços não são troca de óleo, mas suas leituras de km entram no cálculo do ritmo de uso.

### 3.4 Unidade de treino: o intervalo (`data/intervalos.csv`)

Um exemplo = duas trocas de óleo consecutivas do mesmo veículo, ambas com leitura. Para cada intervalo:

- `data_ini, leitura_ini, data_fim, leitura_fim, dias, delta_unid`
- `situacao`: `valido` (0 < delta ≤ 15.000 km, ou ≤ 600 h, e ≤ 730 dias), `perdida` (delta acima do limite ou mais de 2 anos sem registro: houve troca em outro lugar, regra informada pela Cirocar), `erro` (delta ≤ 0 ou dias ≤ 0). Só `valido` entra no treino.

Features, **calculadas só com informação disponível em `data_ini`** (sem vazamento):

| Feature | Como é calculada |
|---|---|
| `taxa_uso_ant` | unidades por dia entre a primeira leitura conhecida do veículo e `leitura_ini` (todas as visitas anteriores, inclusive não-óleo). Vazia se só há uma leitura |
| `intervalo_ant_dias`, `intervalo_ant_unid` | o intervalo válido imediatamente anterior do mesmo veículo. Vazios se não há |
| `n_trocas_ant` | quantas trocas de óleo anteriores o veículo tem na base |
| `idade_veic` | ano de `data_ini` menos ano do veículo |
| `dias_regra` | a conta da regra do fabricante com os dados do próprio veículo: `min(intervalo_rec_unid / taxa_uso_ant, intervalo_rec_meses × 30,4)`; vazia sem taxa de uso |
| `tipo_oleo`, `uso`, `aplicativo`, `frota`, `unidade`, `fonte` | categóricas |
| `intervalo_rec_unid`, `intervalo_rec_meses` | informados pela oficina; se vazios, padrão por tipo de óleo: mineral 5.000/6, semissintético 7.500/12, sintético 10.000/12; máquinas 250 h/6 |

Alvo principal: **`dias`** (dias até a próxima troca). Alvo alternativo testado: `delta_unid`, convertido em dias por `taxa_uso_ant`.

## 4. Separação (etapa 3)

Ordenar intervalos por `data_fim`. **Teste final = últimos 20%**, trancado até a etapa 8. No treino, validação cruzada `KFold(5, shuffle, random_state=42)` para escolher modelos e hiperparâmetros (aula 4). Pré-processamento aprendido só no treino, dentro do `Pipeline`.

O mesmo veículo pode aparecer no treino e no teste: na prática o sistema é usado com clientes conhecidos. Isso é dito no relatório.

## 5. Linhas de base (etapa 4)

Três réguas, todas avaliadas em dias:

1. **Mediana do treino**: todo intervalo dura a mediana.
2. **Regra do fabricante**: `dias = min(intervalo_rec_unid / taxa_uso, intervalo_rec_meses × 30,4)`; `taxa_uso` é `taxa_uso_ant` ou, se vazia, a mediana da fonte no treino.
3. **Repete o último intervalo**: `intervalo_ant_dias`; se vazio, mediana da fonte no treino. (Equivale à "sazonal ingênua" da aula 5 e é a régua mais difícil de bater.)

## 6. Preparo e modelos (etapas 5 e 6)

`ColumnTransformer`:
- numéricas: `SimpleImputer(median, add_indicator=True)` + `StandardScaler`; `taxa_uso_ant` e `intervalo_ant_unid` passam por `log1p` antes (poucos valores enormes, aula 4);
- categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.

Modelos: `LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV` (alphas `logspace(-2, 3, 60)`, aula 2) e **Ridge ajustado pelo Optuna**.

Três conjuntos de features, para a ablação da aula 2:
- `reguas`: só `prev_fabricante` e `prev_ultimo` (as previsões das duas réguas, ajustadas dentro do Pipeline); o modelo aprende o peso de cada uma
- `basico`: `taxa_uso_ant, dias_regra, intervalo_ant_dias, n_trocas_ant, intervalo_rec_unid, intervalo_rec_meses, tipo_oleo, unidade`
- `completo`: básico + `intervalo_ant_unid, idade_veic, uso, aplicativo, frota, fonte`

## 7. Ajuste fino (etapa 7)

Optuna (`TPESampler(seed=42)`, 40 tentativas) maximiza `−MAE` médio na validação cruzada do treino, escolhendo: `alpha` do Ridge (log, 1e-3 a 1e3), conjunto de features (`basico`/`completo`) e alvo (`dias` ou `log(dias)`).

Modelo final = menor MAE na validação cruzada entre todas as réguas e modelos. Uma régua pode vencer.

## 8. Prova final (etapa 8)

Modelo escolhido reajustado no treino inteiro; teste aberto **uma única vez**. Métricas: MAE (dias), RMSE, MAPE, R². Placar no formato da aula 5 (barras horizontais de MAE com a melhor régua tracejada). Comparar MAE de teste com o de validação.

Faixa de previsão: resíduo típico `s` da validação cruzada, faixa `± 2s`, cobertura conferida no teste (aula 1).

## 9. Explicação (etapa 9)

`permutation_importance` no teste (20 repetições, `neg_mean_absolute_error`), gráfico de barras. "Importância não é causa."

## 10. Decisão (etapa 10)

A oficina escolhe **com quantos dias de antecedência** mandar o lembrete. Com `cross_val_predict` no treino (nunca no teste), para cada antecedência `a` de 0 a 60 dias:
- custo_tarde: carros que voltaram antes de `prevista − a` (lembrete chegou depois) × R$ custo_perda;
- custo_cedo: lembretes enviados mais de 45 dias antes do retorno real × R$ custo_contato.

Melhor `a` = menor custo total. Mostrar que `a` muda quando os custos mudam.

## 11. Simulação (etapa 11)

Função `simular(intervalo, **mudancas)` que copia um caso, altera uma coluna (taxa de uso, tipo de óleo, aplicativo) e pede nova previsão. No app, controles deslizantes. Associação não é causa.

## 12. Aplicação (Streamlit)

1. **Lista da semana**: para cada veículo, última troca conhecida, previsão de retorno (data e faixa), dias até vencer, status (vence em X dias / vencido), origem de cada número (previsão do modelo / regra cadastrada / informado).
2. **Simulador "e se"**: escolhe um veículo, mexe nos controles, vê a previsão mudar.
3. **Sobre o modelo**: placar, métricas de validação e teste, importâncias, gráficos, decisão por custo e limitações.

Avisos fixos: base pequena de quatro fontes; a previsão é apoio à decisão e não substitui a inspeção do mecânico; o modelo prevê o retorno de clientes que voltam.

## 13. Estrutura

```
oficinas-ml/
  specs.md  TODO.md  README.md  requirements.txt
  data/brutos/{CIROCAR,AMPESE,LICA,PREFEITURA}.xlsx
  data/visitas.csv  data/intervalos.csv       (gerados)
  src/preparar_dados.py   src/modelos.py   src/treinar.py
  models/modelo_retorno.joblib  models/metricas.json
  app/app.py
  tests/test_preparar.py  tests/test_modelos.py
  relatorio/relatorio.md  relatorio/*.png  relatorio/padroes_do_curso.md
```

## 14. Execução

```
pip install -r requirements.txt
python src/preparar_dados.py
python src/treinar.py
pytest
streamlit run app/app.py
```

## 15. Limitações declaradas

- Base pequena (dezenas de intervalos), quatro fontes, uma delas em horas; resultado é um estudo dessas oficinas, sem generalizar.
- Só entram veículos que voltaram: o modelo prevê o retorno de clientes que voltam, não quando o óleo precisa ser trocado.
- Km informado pode conter erro; leituras impossíveis são descartadas.
- Variável gênero descartada por decisão do grupo (sem base causal e dado sensível); usam-se variáveis de comportamento.
