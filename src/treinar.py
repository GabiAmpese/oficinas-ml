"""
treinar.py — Separação temporal, réguas, modelos, ajuste fino (Optuna), prova final,
explicação, decisão e artefato salvo. Segue as etapas 3 a 10 da aula 4.

Execução:
    python src/preparar_dados.py   (antes)
    python src/treinar.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos import (ALVO, ModeloLogDias, ReguaCega, ReguaFabricante, ReguaMediaDasReguas, ReguaMediana, ReguaUltimoIntervalo,  # noqa: E402
                     colunas_de, construir_modelo, melhor_antecedencia, metricas, preencher_taxa, residuo_tipico)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_INTERVALOS = RAIZ / "data" / "intervalos.csv"
CAMINHO_MODELO = RAIZ / "models" / "modelo_retorno.joblib"
CAMINHO_METRICAS = RAIZ / "models" / "metricas.json"
PASTA_RELATORIO = RAIZ / "relatorio"

FRACAO_TESTE = 0.20
SEMENTE = 42
N_TENTATIVAS_OPTUNA = 40
CUSTO_TARDE = 150.0   # R$: serviço perdido (troca feita em outro lugar)
CUSTO_CEDO = 15.0     # R$: contato à toa
COR = "#6f4e37"


# ---------------------------------------------------------------------------
# Fábricas: cada nome vira um objeto novo com fit/predict (para a validação cruzada)
# ---------------------------------------------------------------------------
def fabrica(nome: str, params: dict | None = None):
    params = params or {}
    if nome == "regua_cega":
        return ReguaCega()
    if nome == "regua_mediana":
        return ReguaMediana()
    if nome == "regua_fabricante":
        return ReguaFabricante()
    if nome == "regua_ultimo_intervalo":
        return ReguaUltimoIntervalo()
    if nome == "regua_media_reguas":
        return ReguaMediaDasReguas()
    tipo, conjunto = params["tipo"], params["conjunto"]
    pipeline = construir_modelo(tipo, conjunto, alpha=params.get("alpha", 1.0))
    if params.get("alvo") == "log_dias":
        return ModeloLogDias(pipeline)
    return pipeline


def prever_cv(nome: str, params: dict | None, X: pd.DataFrame, y: pd.Series, folds) -> np.ndarray:
    """Previsão fora da dobra para cada linha validada (como cross_val_predict). NaN onde a linha não foi validada."""
    oof = np.full(len(X), np.nan)
    for tr, va in folds:
        m = fabrica(nome, params).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    return oof


def dobras_temporais(X: pd.DataFrame, n_blocos: int = 5) -> list:
    """Validação cruzada com corte no tempo (aula 5): janela que expande.

    Os intervalos, já ordenados por data_fim, são divididos em n_blocos cronológicos. O bloco k é a
    validação e o treino são todos os blocos anteriores: nenhum intervalo de treino termina depois do
    fim de um intervalo de validação. É a mesma regra do teste final (treino até uma data, teste depois),
    então validação e teste medem a mesma coisa. O primeiro bloco só treina.
    """
    blocos = np.array_split(np.arange(len(X)), n_blocos)
    return [(np.concatenate(blocos[:k]), blocos[k]) for k in range(1, n_blocos)]


def dobras_temporais_estritas(X: pd.DataFrame, n_blocos: int = 5, minimo_treino: int = 8) -> list:
    """Variante mais conservadora, usada como checagem de robustez.

    O treino do bloco k são APENAS os intervalos cuja data_fim é anterior à MENOR data_ini do bloco:
    só o que já estava concluído quando a primeira previsão do bloco seria feita. Valida menos
    intervalos e deixa as primeiras dobras com pouquíssimo treino.
    """
    blocos = np.array_split(np.arange(len(X)), n_blocos)
    data_fim = X["data_fim"].to_numpy()
    folds = []
    for k in range(1, n_blocos):
        va = blocos[k]
        limite = X["data_ini"].iloc[va].min()
        tr = np.where(data_fim <= np.datetime64(limite))[0]
        if len(tr) >= minimo_treino:
            folds.append((tr, va))
    return folds


def metricas_cv(y: pd.Series, oof: np.ndarray) -> dict:
    """Métricas só sobre as linhas que receberam previsão fora da dobra."""
    mascara = ~np.isnan(oof)
    return metricas(np.asarray(y)[mascara], oof[mascara])


class ModeloDelta:
    """Alvo alternativo: prevê delta em unidades (km ou h) e converte em dias pela taxa de uso."""

    def __init__(self, conjunto: str):
        self.pipeline = construir_modelo("ridge_cv", conjunto)
        self.regua = ReguaFabricante()

    def fit(self, X, y):
        self.regua.fit(X)
        self.pipeline.fit(X, X["delta_unid"])
        return self

    def predict(self, X):
        taxa = preencher_taxa(X, self.regua.taxa_por_fonte_, self.regua.taxa_global_, self.regua.taxa_por_unidade_)
        taxa = np.where(np.isnan(taxa), self.regua.taxa_global_, taxa)
        return np.maximum(self.pipeline.predict(X), 1.0) / np.maximum(taxa, 1e-6)


# ---------------------------------------------------------------------------
def main() -> None:
    # ---------------- Etapa 2/3: dados e separação temporal ----------------
    todos = pd.read_csv(CAMINHO_INTERVALOS, parse_dates=["data_ini", "data_fim"])
    dados = todos[todos["situacao"] == "valido"].sort_values("data_fim").reset_index(drop=True)
    n_teste = max(int(round(len(dados) * FRACAO_TESTE)), 5)
    treino, teste = dados.iloc[:-n_teste].copy(), dados.iloc[-n_teste:].copy()
    X_tr, y_tr = treino, treino[ALVO]
    X_te, y_te = teste, teste[ALVO]

    print("Dados reais de 4 fontes (Cirocar, Ampese, Lica, Prefeitura). Base pequena: estudo dessas oficinas.\n")
    print(f"Intervalos válidos: {len(dados)} | treino: {len(treino)} (até {treino['data_fim'].max().date()}) | "
          f"teste: {len(teste)} (de {teste['data_fim'].min().date()} a {teste['data_fim'].max().date()})")
    print(f"Mediana de dias no treino: {y_tr.median():.0f}\n")

    folds = dobras_temporais(X_tr)
    folds_estritos = dobras_temporais_estritas(X_tr)
    folds_embaralhados = list(KFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(X_tr))
    for tr_i, va_i in folds:  # garantia: nenhum intervalo de treino termina depois de um de validação
        assert X_tr["data_fim"].iloc[tr_i].max() <= X_tr["data_fim"].iloc[va_i].min()
    n_validados = int(sum(len(va) for _, va in folds))
    print("Validação cruzada TEMPORAL (janela que expande): "
          + " | ".join(f"dobra {i + 1}: treino {len(tr)} → valida {len(va)}" for i, (tr, va) in enumerate(folds))
          + f" | {n_validados} intervalos validados de {len(X_tr)}")
    print("Checagem estrita: " + " | ".join(f"{len(tr)} → {len(va)}" for tr, va in folds_estritos) + "\n")

    # ---------------- Etapas 4 a 6: réguas e modelos na validação cruzada ----------------
    candidatos = {
        "regua_cega": None,
        "regua_mediana": None,
        "regua_fabricante": None,
        "regua_ultimo_intervalo": None,
        "regua_media_reguas": None,
    }
    for conjunto in ["reguas", "basico", "completo"]:
        for tipo in ["linear", "ridge_cv", "lasso_cv", "elasticnet_cv"]:
            candidatos[f"{tipo}_{conjunto}"] = {"tipo": tipo, "conjunto": conjunto, "alvo": "dias"}

    resultados_cv, previsoes_cv, resultados_estrita, resultados_embaralhado = {}, {}, {}, {}
    for nome, params in candidatos.items():
        oof = prever_cv(nome, params, X_tr, y_tr, folds)
        resultados_cv[nome] = metricas_cv(y_tr, oof)
        previsoes_cv[nome] = oof
        # as outras duas validações ficam só para comparação no relatório; não decidem nada
        resultados_estrita[nome] = metricas_cv(y_tr, prever_cv(nome, params, X_tr, y_tr, folds_estritos))
        resultados_embaralhado[nome] = metricas_cv(y_tr, prever_cv(nome, params, X_tr, y_tr, folds_embaralhados))

    # alvo alternativo: delta em unidades convertido em dias
    oof = np.full(len(X_tr), np.nan)
    for tr, va in folds:
        oof[va] = ModeloDelta("basico").fit(X_tr.iloc[tr], y_tr.iloc[tr]).predict(X_tr.iloc[va])
    resultados_cv["ridge_cv_delta_basico"] = metricas_cv(y_tr, oof)
    previsoes_cv["ridge_cv_delta_basico"] = oof

    # ---------------- Etapa 7: ajuste fino com Optuna ----------------
    def avaliar(tentativa):
        params = {
            "tipo": "ridge",
            "alpha": tentativa.suggest_float("alpha", 1e-3, 1e3, log=True),
            "conjunto": tentativa.suggest_categorical("conjunto", ["reguas", "basico", "completo"]),
            "alvo": tentativa.suggest_categorical("alvo", ["dias", "log_dias"]),
        }
        oof = prever_cv("ridge_optuna", params, X_tr, y_tr, folds)
        return metricas_cv(y_tr, oof)["mae_dias"]

    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMENTE))
    estudo.optimize(avaliar, n_trials=N_TENTATIVAS_OPTUNA)
    params_optuna = {"tipo": "ridge", **estudo.best_params}
    oof = prever_cv("ridge_optuna", params_optuna, X_tr, y_tr, folds)
    resultados_cv["ridge_optuna"] = metricas_cv(y_tr, oof)
    previsoes_cv["ridge_optuna"] = oof
    candidatos["ridge_optuna"] = params_optuna

    print("VALIDAÇÃO CRUZADA TEMPORAL no treino (MAE em dias) | colunas: expandindo [estrita] (embaralhada antiga):")
    for nome, m in sorted(resultados_cv.items(), key=lambda kv: kv[1]["mae_dias"]):
        e = resultados_estrita.get(nome, {}).get("mae_dias"); a = resultados_embaralhado.get(nome, {}).get("mae_dias")
        extra = (f"[{e:5.1f}]" if e is not None else "       ") + " " + (f"({a:5.1f})" if a is not None else "       ")
        print(f"  {nome:28s} MAE {m['mae_dias']:7.1f} {extra} | RMSE {m['rmse_dias']:7.1f} | MAPE {m['mape_pct']:5.1f}% | R² {m['r2']:6.3f}")
    print(f"\nOptuna: melhor alpha {estudo.best_params['alpha']:.4f}, conjunto {estudo.best_params['conjunto']}, "
          f"alvo {estudo.best_params['alvo']} (MAE {estudo.best_value:.1f})")

    # ---------------- Escolha do modelo final ----------------
    # menor MAE na validação; em empate (diferença menor que 0,5 dia), fica a opção mais simples,
    # que é a que aparece antes na lista (réguas antes de modelos)
    melhor_mae = min(m["mae_dias"] for m in resultados_cv.values())
    nome_final = next(k for k, m in resultados_cv.items() if m["mae_dias"] <= melhor_mae + 0.5)
    melhor_regua = min([k for k in resultados_cv if k.startswith("regua")], key=lambda k: resultados_cv[k]["mae_dias"])
    melhor_ml = min([k for k in resultados_cv if not k.startswith("regua")], key=lambda k: resultados_cv[k]["mae_dias"])
    print(f"\nModelo escolhido pela validação: {nome_final}  |  melhor régua: {melhor_regua}  |  melhor ML: {melhor_ml}")

    # ---------------- Etapa 8: prova final (uma única vez) ----------------
    def ajustar_final(nome):
        if nome == "ridge_cv_delta_basico":
            return ModeloDelta("basico").fit(X_tr, y_tr)
        return fabrica(nome, candidatos.get(nome)).fit(X_tr, y_tr)

    modelo_final = ajustar_final(nome_final)
    prev_te = modelo_final.predict(X_te)
    metricas_teste = metricas(y_te, prev_te)

    # tabela caso a caso do teste (para análise, nunca para reescolher modelo)
    regua_fab_te = fabrica("regua_fabricante").fit(X_tr, y_tr).predict(X_te)
    regua_ult_te = fabrica("regua_ultimo_intervalo").fit(X_tr, y_tr).predict(X_te)
    modelo_ml_prev = ajustar_final(melhor_ml).predict(X_te)
    detalhe = pd.DataFrame({
        "veiculo": X_te["veiculo"].values, "fonte": X_te["fonte"].values,
        "data_ini": X_te["data_ini"].dt.date.values, "data_fim": X_te["data_fim"].dt.date.values,
        "dias_reais": y_te.values,
        "prev_escolhido": np.round(prev_te, 1), "erro_escolhido": np.round(np.abs(y_te.values - prev_te), 1),
        "prev_fabricante_personalizado": np.round(regua_fab_te, 1), "erro_fabricante_personalizado": np.round(np.abs(y_te.values - regua_fab_te), 1),
        "prev_ultimo_intervalo": np.round(regua_ult_te, 1), "erro_ultimo_intervalo": np.round(np.abs(y_te.values - regua_ult_te), 1),
        "prev_melhor_ml": np.round(modelo_ml_prev, 1), "erro_melhor_ml": np.round(np.abs(y_te.values - modelo_ml_prev), 1),
    })
    PASTA_RELATORIO.mkdir(exist_ok=True)
    detalhe.to_csv(PASTA_RELATORIO / "teste_detalhado.csv", index=False)

    # réguas e melhor ML também no teste, para o placar
    placar_teste = {nome_final: metricas_teste}
    modelo_ml = ajustar_final(melhor_ml)
    prev_te_ml = modelo_ml.predict(X_te)
    for nome in ["regua_cega", "regua_mediana", "regua_fabricante", "regua_ultimo_intervalo", "regua_media_reguas"]:
        if nome != nome_final:
            placar_teste[nome] = metricas(y_te, fabrica(nome).fit(X_tr, y_tr).predict(X_te))
    if melhor_ml != nome_final:
        placar_teste[melhor_ml] = metricas(y_te, prev_te_ml)

    print("\nPROVA FINAL (teste, aberto uma única vez):")
    for nome, m in sorted(placar_teste.items(), key=lambda kv: kv[1]["mae_dias"]):
        print(f"  {nome:28s} MAE {m['mae_dias']:7.1f} | RMSE {m['rmse_dias']:7.1f} | MAPE {m['mape_pct']:5.1f}% | R² {m['r2']:6.3f}")

    # faixa de previsão (aula 1): s da validação cruzada, cobertura conferida no teste
    mascara_cv = ~np.isnan(previsoes_cv[nome_final])
    y_cv = y_tr.to_numpy()[mascara_cv]
    s = residuo_tipico(y_cv, previsoes_cv[nome_final][mascara_cv])
    margem = 2 * s
    cobertura_teste = float(np.mean(np.abs(y_te - prev_te) <= margem))
    print(f"\nResíduo típico s (validação): {s:.1f} dias | faixa ±{margem:.0f} dias cobre {100 * cobertura_teste:.0f}% do teste")

    # ---------------- Etapa 9: explicação ----------------
    importancias = {}
    conjunto_final = candidatos.get(nome_final, {}).get("conjunto") if isinstance(candidatos.get(nome_final), dict) else None
    if nome_final == "ridge_cv_delta_basico":
        conjunto_final = "basico"
    # explicação: do modelo final se for ML; se a régua venceu, do melhor modelo de ML
    nome_explicado = nome_final if conjunto_final else melhor_ml
    modelo_explicado = modelo_final if conjunto_final else modelo_ml
    conjunto_explicado = conjunto_final or (candidatos[melhor_ml]["conjunto"] if isinstance(candidatos.get(melhor_ml), dict) else "basico")
    # colunas brutas que alimentam as réguas empilhadas também contam
    cols = [c for c in colunas_de(conjunto_explicado) if c not in ("prev_fabricante", "prev_ultimo")]
    cols += ["taxa_uso_ant", "intervalo_ant_dias", "intervalo_rec_unid", "intervalo_rec_meses", "fonte"]
    cols = list(dict.fromkeys(cols))
    if True:
        imp = permutation_importance(modelo_explicado, X_te, y_te, scoring="neg_mean_absolute_error",
                                     n_repeats=20, random_state=SEMENTE)
        # permutation_importance embaralha todas as colunas do DataFrame; ficamos só com as usadas
        serie = pd.Series(imp.importances_mean, index=X_te.columns)[cols].sort_values()
        importancias = {k: round(float(v), 2) for k, v in serie.items()}
        print(f"\nImportância por permutação no {nome_explicado} (aumento do MAE em dias ao embaralhar):")
        for k, v in sorted(importancias.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v:+6.1f}")

    # ---------------- Etapa 10: decisão ----------------
    oof_final = previsoes_cv[nome_final][mascara_cv]
    ant_padrao, custos_padrao = melhor_antecedencia(y_cv, oof_final, CUSTO_TARDE, CUSTO_CEDO)
    ant_caro, custos_caro = melhor_antecedencia(y_cv, oof_final, CUSTO_TARDE, CUSTO_CEDO * 5)
    print(f"\nDecisão: lembrar {ant_padrao} dias antes da data prevista (perda R$ {CUSTO_TARDE:.0f}, contato R$ {CUSTO_CEDO:.0f}); "
          f"se o contato custar R$ {CUSTO_CEDO * 5:.0f}, {ant_caro} dias.")

    # ---------------- Gráficos ----------------
    PASTA_RELATORIO.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130})

    # placar (aula 5)
    nomes = sorted(placar_teste, key=lambda k: placar_teste[k]["mae_dias"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(nomes, [placar_teste[n]["mae_dias"] for n in nomes], color=COR)
    ax.axvline(placar_teste[melhor_regua]["mae_dias"], color="black", linestyle="--", label="régua escolhida na validação")
    ax.invert_yaxis(); ax.set_xlabel("MAE no teste (dias)"); ax.set_title("Quem bate a régua (linha tracejada)?"); ax.legend()
    fig.tight_layout(); fig.savefig(PASTA_RELATORIO / "placar.png"); plt.close(fig)

    # validação cruzada: todos os candidatos
    nomes_cv = sorted(resultados_cv, key=lambda k: resultados_cv[k]["mae_dias"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(nomes_cv, [resultados_cv[n]["mae_dias"] for n in nomes_cv],
            color=[("black" if n.startswith("regua") else COR) for n in nomes_cv])
    ax.axvline(resultados_cv[melhor_regua]["mae_dias"], color="black", linestyle="--")
    ax.invert_yaxis(); ax.set_xlabel("MAE na validação cruzada temporal (dias)"); ax.set_title("Réguas (preto) e modelos (marrom)")
    fig.tight_layout(); fig.savefig(PASTA_RELATORIO / "validacao_cruzada.png"); plt.close(fig)

    # previsto x real e resíduos
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = [0, max(y_te.max(), prev_te.max()) * 1.05]
    eixos[0].scatter(y_te, prev_te, color=COR); eixos[0].plot(lim, lim, "--", color="gray")
    eixos[0].fill_between(lim, [lim[0] - margem, lim[1] - margem], [lim[0] + margem, lim[1] + margem], alpha=0.15, color=COR)
    eixos[0].set_xlabel("dias reais"); eixos[0].set_ylabel("dias previstos"); eixos[0].set_title(f"Teste: previsto x real ({nome_final})")
    eixos[1].scatter(prev_te, y_te - prev_te, color=COR); eixos[1].axhline(0, color="gray")
    eixos[1].set_xlabel("dias previstos"); eixos[1].set_ylabel("resíduo (dias)"); eixos[1].set_title("Resíduo x previsto")
    fig.tight_layout(); fig.savefig(PASTA_RELATORIO / "previsto_vs_real.png"); plt.close(fig)

    # optuna
    valores = [t.value for t in estudo.trials]
    melhores = np.minimum.accumulate(valores)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(range(1, len(valores) + 1), valores, color=COR, label="tentativa"); ax.plot(range(1, len(valores) + 1), melhores, color="black", label="melhor até agora")
    ax.set_xlabel("tentativa"); ax.set_ylabel("MAE na validação (dias)"); ax.legend(); ax.set_title("Ajuste fino com Optuna")
    fig.tight_layout(); fig.savefig(PASTA_RELATORIO / "optuna.png"); plt.close(fig)

    # importâncias
    if importancias:
        serie = pd.Series(importancias).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        serie.plot(kind="barh", color=COR, ax=ax); ax.set_xlabel("aumento do MAE (dias) ao embaralhar a coluna"); ax.set_title("Em que o modelo presta atenção")
        fig.tight_layout(); fig.savefig(PASTA_RELATORIO / "importancias.png"); plt.close(fig)

    # custo x antecedência
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(range(0, 61), custos_padrao, color=COR, label=f"contato R$ {CUSTO_CEDO:.0f}")
    ax.plot(range(0, 61), custos_caro, color="gray", label=f"contato R$ {CUSTO_CEDO * 5:.0f}")
    ax.axvline(ant_padrao, color=COR, linestyle="--"); ax.axvline(ant_caro, color="gray", linestyle="--")
    ax.set_xlabel("antecedência do lembrete (dias antes da data prevista)"); ax.set_ylabel("custo total dos erros (R$)"); ax.legend()
    ax.set_title("O modelo estima a data. A antecedência é decisão da oficina.")
    fig.tight_layout(); fig.savefig(PASTA_RELATORIO / "decisao_custo.png"); plt.close(fig)

    # ---------------- Artefato ----------------
    CAMINHO_MODELO.parent.mkdir(exist_ok=True)
    s_ml = residuo_tipico(y_cv, previsoes_cv[melhor_ml][mascara_cv])
    joblib.dump({"modelo": modelo_final, "nome": nome_final, "conjunto": conjunto_final, "s": s,
                 "modelo_ml": modelo_ml, "nome_ml": melhor_ml, "s_ml": s_ml,
                 "regua_fabricante": ReguaFabricante().fit(X_tr), "regua_ultimo": ReguaUltimoIntervalo().fit(X_tr, y_tr)},
                CAMINHO_MODELO)
    registro = {
        "aviso": "Dados reais de quatro fontes, base pequena. Estudo dessas oficinas, sem generalização.",
        "data_treino": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modelo_escolhido": nome_final,
        "params_modelo": candidatos.get(nome_final) if isinstance(candidatos.get(nome_final), dict) else None,
        "melhor_regua": melhor_regua,
        "melhor_ml": melhor_ml,
        "params_melhor_ml": candidatos.get(melhor_ml) if isinstance(candidatos.get(melhor_ml), dict) else None,
        "explicacao_de": nome_explicado,
        "tamanhos": {"treino": len(treino), "teste": len(teste), "veiculos_treino": int(treino["veiculo"].nunique()),
                     "veiculos_teste": int(teste["veiculo"].nunique())},
        "periodos": {"treino": [str(treino["data_fim"].min().date()), str(treino["data_fim"].max().date())],
                     "teste": [str(teste["data_fim"].min().date()), str(teste["data_fim"].max().date())]},
        "por_fonte": dados.groupby("fonte").size().to_dict(),
        "validacao": {"tipo": "temporal, janela que expande (aula 5)",
                      "dobras": [{"treino": int(len(tr)), "validacao": int(len(va))} for tr, va in folds],
                      "intervalos_validados": n_validados,
                      "dobras_estritas": [{"treino": int(len(tr)), "validacao": int(len(va))} for tr, va in folds_estritos]},
        "validacao_cruzada": resultados_cv,
        "validacao_cruzada_estrita_checagem": resultados_estrita,
        "validacao_cruzada_embaralhada_antes": resultados_embaralhado,
        "teste_final": placar_teste,
        "optuna": {"melhores_params": estudo.best_params, "melhor_mae": round(estudo.best_value, 2), "tentativas": N_TENTATIVAS_OPTUNA},
        "faixa": {"s_dias": round(s, 1), "margem_dias": round(margem, 1), "cobertura_teste": round(cobertura_teste, 3),
                  "s_ml_dias": round(s_ml, 1)},
        "importancias": importancias,
        "decisao": {"custo_tarde": CUSTO_TARDE, "custo_cedo": CUSTO_CEDO, "antecedencia_dias": int(ant_padrao),
                    "antecedencia_contato_caro": int(ant_caro), "custos_por_antecedencia": [round(c) for c in custos_padrao]},
    }
    with open(CAMINHO_METRICAS, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nModelo salvo em {CAMINHO_MODELO}\nMétricas em    {CAMINHO_METRICAS}\nGráficos em    {PASTA_RELATORIO}\nTeste caso a caso em {PASTA_RELATORIO / 'teste_detalhado.csv'}")


if __name__ == "__main__":
    main()
