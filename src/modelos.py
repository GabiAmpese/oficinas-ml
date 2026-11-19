"""
modelos.py — Peças reutilizáveis: features, réguas (baselines), preparo, métricas, decisão e simulação.

Usado por treinar.py, app/app.py e pelos testes.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DIAS_POR_MES = 30.4
ALVO = "dias"

# ---------------------------------------------------------------------------
# Conjuntos de features (specs seção 6)
# ---------------------------------------------------------------------------
FEATURES = {
    "reguas": {   # só as previsões das réguas: o modelo aprende o peso de cada uma
        "log": [],
        "num": ["prev_fabricante", "prev_ultimo"],
        "cat": [],
    },
    "basico": {
        "log": ["taxa_uso_ant"],
        "num": ["prev_fabricante", "prev_ultimo", "dias_regra", "intervalo_ant_dias", "n_trocas_ant",
                "intervalo_rec_unid", "intervalo_rec_meses"],
        "cat": ["tipo_oleo", "unidade"],
    },
    "completo": {
        "log": ["taxa_uso_ant", "intervalo_ant_unid"],
        "num": ["prev_fabricante", "prev_ultimo", "dias_regra", "intervalo_ant_dias", "n_trocas_ant",
                "intervalo_rec_unid", "intervalo_rec_meses", "idade_veic"],
        "cat": ["tipo_oleo", "unidade", "uso", "aplicativo", "frota", "fonte"],
    },
}


def colunas_de(conjunto: str) -> list:
    f = FEATURES[conjunto]
    return f["log"] + f["num"] + f["cat"]


# ---------------------------------------------------------------------------
# Preparo (aula 4, etapa 5): uma receita por tipo de coluna
# ---------------------------------------------------------------------------
def construir_preparo(conjunto: str) -> ColumnTransformer:
    f = FEATURES[conjunto]
    receita_log = Pipeline([
        ("preencher", SimpleImputer(strategy="median", add_indicator=True)),
        ("logaritmo", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("padronizar", StandardScaler()),
    ])
    receita_num = Pipeline([
        ("preencher", SimpleImputer(strategy="median", add_indicator=True)),
        ("padronizar", StandardScaler()),
    ])
    receita_cat = Pipeline([
        ("preencher", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("log", receita_log, f["log"]),
        ("num", receita_num, f["num"]),
        ("cat", receita_cat, f["cat"]),
    ])


class FeaturesDeRegua(BaseEstimator, TransformerMixin):
    """Acrescenta ao DataFrame as previsões das réguas, ajustadas só no treino (dentro do Pipeline).

    Assim o modelo aprende a CORREÇÃO sobre a regra do fabricante e sobre o hábito do cliente,
    em vez de precisar redescobri-los a partir de 47 exemplos.
    """

    def fit(self, X, y=None):
        self.fabricante_ = ReguaFabricante().fit(X)
        self.ultimo_ = ReguaUltimoIntervalo().fit(X, y)
        return self

    def transform(self, X):
        X = X.copy()
        X["prev_fabricante"] = self.fabricante_.predict(X)
        X["prev_ultimo"] = self.ultimo_.predict(X)
        return X

    def get_feature_names_out(self, input_features=None):
        return list(input_features) + ["prev_fabricante", "prev_ultimo"]


def construir_modelo(nome: str, conjunto: str, alpha: float = 1.0) -> Pipeline:
    forcas = np.logspace(-2, 3, 60)
    modelos = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=alpha),
        "ridge_cv": RidgeCV(alphas=forcas),
        "lasso_cv": LassoCV(alphas=forcas, cv=5, max_iter=50000, random_state=0),
        "elasticnet_cv": ElasticNetCV(alphas=forcas, l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=50000, random_state=0),
    }
    return Pipeline([
        ("reguas", FeaturesDeRegua()),              # previsões das réguas viram entradas (empilhamento)
        ("preparo", construir_preparo(conjunto)),
        ("modelo", modelos[nome]),
    ])


class ModeloLogDias(RegressorMixin, BaseEstimator):
    """Envolve um Pipeline para treinar em log(dias) e devolver dias. Mesma interface fit/predict."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, np.log1p(np.asarray(y, dtype=float)))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.expm1(self.pipeline.predict(X))


# ---------------------------------------------------------------------------
# Réguas (aula 4 etapa 4, aula 5 seção 5)
# ---------------------------------------------------------------------------
def preencher_taxa(X: pd.DataFrame, taxa_por_fonte: dict, taxa_global: float, taxa_por_unidade: dict | None = None) -> np.ndarray:
    """Taxa de uso do próprio veículo; se faltar, mediana da fonte; depois da unidade (km ou h).
    Se nem a unidade tem mediana (ex.: máquinas em horas sem histórico), devolve NaN: a regra usa só os meses."""
    taxa = X["taxa_uso_ant"].copy()
    faltando = taxa.isna()
    por_fonte = X.loc[faltando, "fonte"].map(taxa_por_fonte)
    por_unid = X.loc[faltando, "unidade"].map(taxa_por_unidade or {})
    taxa[faltando] = por_fonte.fillna(por_unid)
    return taxa.to_numpy(dtype=float)


class ReguaMediana:
    """Todo intervalo dura a mediana do treino."""

    def fit(self, X, y):
        self.mediana_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mediana_)


class ReguaFabricante:
    """dias = min(intervalo recomendado / taxa de uso, meses recomendados x 30,4)."""

    def fit(self, X, y=None):
        com_taxa = X.dropna(subset=["taxa_uso_ant"])
        self.taxa_por_fonte_ = com_taxa.groupby("fonte")["taxa_uso_ant"].median().to_dict()
        self.taxa_por_unidade_ = com_taxa.groupby("unidade")["taxa_uso_ant"].median().to_dict()
        self.taxa_global_ = float(com_taxa["taxa_uso_ant"].median()) if len(com_taxa) else 1.0
        return self

    def predict(self, X):
        taxa = preencher_taxa(X, self.taxa_por_fonte_, self.taxa_global_, self.taxa_por_unidade_)
        with np.errstate(divide="ignore", invalid="ignore"):
            por_unidades = X["intervalo_rec_unid"].to_numpy(dtype=float) / taxa
        por_unidades = np.where(np.isnan(por_unidades), np.inf, por_unidades)   # sem taxa: vale só os meses
        por_meses = X["intervalo_rec_meses"].to_numpy(dtype=float) * DIAS_POR_MES
        return np.minimum(por_unidades, por_meses)


class ReguaUltimoIntervalo:
    """Repete o último intervalo do veículo; sem histórico, usa a mediana da fonte no treino."""

    def fit(self, X, y):
        dados = X.assign(y=np.asarray(y))
        self.mediana_por_fonte_ = dados.groupby("fonte")["y"].median().to_dict()
        self.mediana_global_ = float(np.median(y))
        return self

    def predict(self, X):
        prev = X["intervalo_ant_dias"].copy()
        faltando = prev.isna()
        prev[faltando] = X.loc[faltando, "fonte"].map(self.mediana_por_fonte_).fillna(self.mediana_global_)
        return prev.to_numpy(dtype=float)


class ReguaMediaDasReguas:
    """Média simples entre a regra do fabricante e o último intervalo."""

    def fit(self, X, y):
        self.fabricante_ = ReguaFabricante().fit(X)
        self.ultimo_ = ReguaUltimoIntervalo().fit(X, y)
        return self

    def predict(self, X):
        return (self.fabricante_.predict(X) + self.ultimo_.predict(X)) / 2


# ---------------------------------------------------------------------------
# Métricas (aulas 1 e 5)
# ---------------------------------------------------------------------------
def metricas(y_real, y_prev) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    erro = y_real - y_prev
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    return {
        "mae_dias": round(float(np.mean(np.abs(erro))), 2),
        "rmse_dias": round(float(np.sqrt(np.mean(erro ** 2))), 2),
        "mape_pct": round(float(100 * np.mean(np.abs(erro / np.maximum(y_real, 1)))), 1),
        "r2": round(float(1 - np.sum(erro ** 2) / ss_tot), 3) if ss_tot > 0 else float("nan"),
        "n": int(len(y_real)),
    }


def residuo_tipico(y_real, y_prev, n_parametros: int = 2) -> float:
    """s = sqrt(soma(residuo^2) / (n - p)), como na aula 1."""
    erro = np.asarray(y_real, dtype=float) - np.asarray(y_prev, dtype=float)
    n = len(erro)
    return float(np.sqrt(np.sum(erro ** 2) / max(n - n_parametros, 1)))


# ---------------------------------------------------------------------------
# Decisão (aula 4, etapa 10): antecedência do lembrete escolhida pelo custo
# ---------------------------------------------------------------------------
def custo_antecedencia(dias_real, dias_prev, antecedencia: float, custo_tarde: float, custo_cedo: float,
                       janela_cedo: int = 45) -> float:
    """Custo total de mandar o lembrete `antecedencia` dias antes da data prevista.

    tarde: o carro voltou antes do lembrete (lembrete inútil, risco de perder o serviço).
    cedo:  o lembrete chegou mais de `janela_cedo` dias antes do retorno real (cliente ignora).
    """
    dias_real = np.asarray(dias_real, dtype=float)
    dia_lembrete = np.asarray(dias_prev, dtype=float) - antecedencia
    tarde = (dias_real < dia_lembrete).sum()
    cedo = ((dias_real - dia_lembrete) > janela_cedo).sum()
    return float(tarde * custo_tarde + cedo * custo_cedo)


def melhor_antecedencia(dias_real, dias_prev, custo_tarde: float, custo_cedo: float,
                        antecedencias=range(0, 61)) -> tuple:
    custos = [custo_antecedencia(dias_real, dias_prev, a, custo_tarde, custo_cedo) for a in antecedencias]
    i = int(np.argmin(custos))
    return list(antecedencias)[i], custos


# ---------------------------------------------------------------------------
# Simulação (aula 4, etapa 11)
# ---------------------------------------------------------------------------
def simular(modelo, caso: pd.DataFrame, **mudancas) -> float:
    """Copia o caso, troca colunas e devolve a nova previsão em dias."""
    cenario = caso.copy()
    for coluna, valor in mudancas.items():
        cenario[coluna] = valor
    return float(modelo.predict(cenario)[0])
