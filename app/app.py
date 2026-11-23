"""
app.py — Painel da oficina: lista da semana, simulador "e se" e página sobre o modelo.

Nunca treina. Carrega models/modelo_retorno.joblib e models/metricas.json gerados por src/treinar.py.

Execução:
    streamlit run app/app.py
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from modelos import simular  # noqa: E402
from preparar_dados import PADRAO_RECOMENDADO  # noqa: E402

CAMINHO_MODELO = RAIZ / "models" / "modelo_retorno.joblib"
CAMINHO_METRICAS = RAIZ / "models" / "metricas.json"
CAMINHO_VISITAS = RAIZ / "data" / "visitas.csv"
CAMINHO_INTERVALOS = RAIZ / "data" / "intervalos.csv"
PASTA_RELATORIO = RAIZ / "relatorio"

PREVISAO = "🔮 previsão do modelo"
REGRA = "📋 regra cadastrada"
INFORMADO = "✏️ informado pela oficina"

NOMES = {
    "regua_media_reguas": "Média das réguas (fabricante + último intervalo)",
    "regua_fabricante": "Regra do fabricante",
    "regua_ultimo_intervalo": "Repete o último intervalo",
    "regua_mediana": "Mediana do treino",
    "regua_cega": "Regra cega do fabricante (como a oficina faz hoje)",
    "ridge_cv_basico": "Ridge (features básicas)",
    "ridge_optuna": "Ridge ajustado por Optuna",
}


@st.cache_resource
def carregar_artefato():
    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def carregar_metricas():
    with open(CAMINHO_METRICAS, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def carregar_dados():
    visitas = pd.read_csv(CAMINHO_VISITAS, parse_dates=["data"])
    intervalos = pd.read_csv(CAMINHO_INTERVALOS, parse_dates=["data_ini", "data_fim"])
    return visitas, intervalos


def nome_bonito(chave: str) -> str:
    return NOMES.get(chave, chave.replace("_", " "))


# ---------------------------------------------------------------------------
# Situação atual de cada veículo: última troca conhecida e features para prever a próxima
# ---------------------------------------------------------------------------
def situacao_atual(visitas: pd.DataFrame, intervalos: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for veiculo, g in visitas.groupby("veiculo"):
        g = g.sort_values("data")
        trocas = g[g["troca_oleo"] & g["leitura"].notna()]
        if trocas.empty:
            continue
        ultima = trocas.iloc[-1]
        com_leitura = g[g["leitura"].notna() & (g["data"] <= ultima["data"])]
        primeira = com_leitura.iloc[0]
        dias_hist = (ultima["data"] - primeira["data"]).days
        unid_hist = ultima["leitura"] - primeira["leitura"]
        taxa = unid_hist / dias_hist if dias_hist > 0 and unid_hist > 0 else float("nan")

        validos = intervalos[(intervalos["veiculo"] == veiculo) & (intervalos["situacao"] == "valido")].sort_values("data_fim")
        ant_dias = validos.iloc[-1]["dias"] if len(validos) else float("nan")
        ant_unid = validos.iloc[-1]["delta_unid"] if len(validos) else float("nan")

        unidade = ultima["unidade"]
        tipo = ultima["tipo_oleo"] if pd.notna(ultima["tipo_oleo"]) else None
        tabela = PADRAO_RECOMENDADO[unidade]
        padrao_unid, padrao_meses = tabela.get(tipo, tabela[None])
        rec_unid = ultima["intervalo_rec_unid"] if pd.notna(ultima["intervalo_rec_unid"]) else padrao_unid
        rec_meses = ultima["intervalo_rec_meses"] if pd.notna(ultima["intervalo_rec_meses"]) else padrao_meses
        dias_regra = min(rec_unid / taxa, rec_meses * 30.4) if taxa == taxa and taxa > 0 else float("nan")

        linhas.append({
            "fonte": ultima["fonte"], "oficina": ultima["oficina"], "veiculo": veiculo,
            "marca": ultima["marca"], "modelo": ultima["modelo"], "unidade": unidade,
            "data_ini": ultima["data"], "leitura_ini": ultima["leitura"],
            "taxa_uso_ant": taxa, "intervalo_ant_dias": ant_dias, "intervalo_ant_unid": ant_unid,
            "n_trocas_ant": int(len(trocas)), "idade_veic": (ultima["data"].year - ultima["ano"]) if pd.notna(ultima["ano"]) else float("nan"),
            "dias_regra": dias_regra, "tipo_oleo": tipo, "intervalo_rec_unid": float(rec_unid), "intervalo_rec_meses": float(rec_meses),
            "rec_informado": bool(pd.notna(ultima["intervalo_rec_unid"])),
            "uso": ultima["uso"], "aplicativo": ultima["aplicativo"], "frota": ultima["frota"],
            "delta_unid": float("nan"),
        })
    return pd.DataFrame(linhas)


def prever(art, X: pd.DataFrame, usar_ml: bool):
    modelo = art["modelo_ml"] if usar_ml else art["modelo"]
    s = art["s_ml"] if usar_ml else art["s"]
    return modelo.predict(X), s


def aviso_rodape():
    st.divider()
    st.caption("⚠️ Base pequena (dezenas de intervalos reais de quatro fontes). A previsão é apoio à decisão e não "
               "substitui a inspeção do mecânico. O sistema prevê o retorno de clientes que voltam à oficina.")


# ---------------------------------------------------------------------------
def pagina_lista(art, met, visitas, intervalos):
    st.title("Lista da semana: quem contatar para a troca de óleo")
    st.caption("O sistema prevê quando o cliente costuma voltar, com base no histórico. Quando o óleo deve ser trocado "
               "tecnicamente é a recomendação do fabricante; a decisão final é do mecânico.")
    hoje = st.date_input("Data de referência", value=date(2026, 11, 19))
    col1, col2 = st.columns(2)
    usar_ml = col1.radio("Previsão com", ["Regra escolhida na validação", "Melhor modelo de ML"], horizontal=True) == "Melhor modelo de ML"
    antecedencia = col2.slider("Antecedência do lembrete (dias antes da data prevista)", 0, 60, int(met["decisao"]["antecedencia_dias"]),
                               help="Escolhida pelo custo dos erros (página Sobre o modelo). Pode ajustar.")
    col3, col4 = st.columns(2)
    fontes = col3.multiselect("Oficinas", sorted(visitas["fonte"].unique()), default=sorted(visitas["fonte"].unique()))
    limite_sumido = col4.number_input("Ocultar veículos sem troca há mais de (dias)", 30, 3000, 540, 30,
                                      help="Cliente que não volta há muito tempo provavelmente trocou em outro lugar.")

    atual = situacao_atual(visitas, intervalos)
    atual = atual[atual["fonte"].isin(fontes)].reset_index(drop=True)
    sumidos = int(((pd.Timestamp(hoje) - atual["data_ini"]).dt.days > limite_sumido).sum())
    atual = atual[(pd.Timestamp(hoje) - atual["data_ini"]).dt.days <= limite_sumido].reset_index(drop=True)
    if atual.empty:
        st.info("Nenhum veículo nas oficinas selecionadas."); return

    prev, s = prever(art, atual, usar_ml)
    atual["dias_previstos"] = prev.round(0)
    atual["data_prevista"] = atual["data_ini"] + pd.to_timedelta(atual["dias_previstos"], unit="D")
    atual["data_lembrete"] = atual["data_prevista"] - timedelta(days=int(antecedencia))
    atual["faixa_ini"] = atual["data_prevista"] - timedelta(days=int(2 * s))
    atual["faixa_fim"] = atual["data_prevista"] + timedelta(days=int(2 * s))
    hoje_ts = pd.Timestamp(hoje)
    atual["dias_para_vencer"] = (atual["data_prevista"] - hoje_ts).dt.days

    def status(row):
        if row["data_prevista"] < hoje_ts:
            return "🔴 retorno previsto ultrapassado"
        if row["data_lembrete"] <= hoje_ts:
            return "🟡 contatar agora"
        return "🟢 no prazo"
    atual["status"] = atual.apply(status, axis=1)
    ordem = {"🟡 contatar agora": 0, "🔴 retorno previsto ultrapassado": 1, "🟢 no prazo": 2}
    atual = atual.sort_values(["status", "data_prevista"], key=lambda c: c.map(ordem) if c.name == "status" else c)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Veículos acompanhados", len(atual))
    c2.metric("Contatar agora", int((atual["status"] == "🟡 contatar agora").sum()))
    c3.metric("Retorno previsto ultrapassado", int((atual["status"] == "🔴 retorno previsto ultrapassado").sum()))
    c4.metric("Ocultos (sumidos)", sumidos)

    exibir = atual[["status", "oficina", "veiculo", "marca", "modelo", "data_ini", "leitura_ini", "unidade",
                    "dias_previstos", "data_prevista", "faixa_ini", "faixa_fim", "data_lembrete", "dias_para_vencer"]].copy()
    for c in ["data_ini", "data_prevista", "faixa_ini", "faixa_fim", "data_lembrete"]:
        exibir[c] = exibir[c].dt.strftime("%d/%m/%Y")
    exibir.columns = ["Status", "Oficina", "Veículo", "Marca", "Modelo", "Última troca", "Leitura", "Un.",
                      "Dias previstos", "Retorno previsto", "Faixa de", "Faixa até", "Lembrete em", "Dias p/ vencer"]
    st.dataframe(exibir, width="stretch", hide_index=True)
    st.caption(f"Retorno previsto: {PREVISAO} ({nome_bonito(art['nome_ml'] if usar_ml else art['nome'])}). "
               f"Faixa: ±{2 * s:.0f} dias (2 × resíduo típico da validação). Última troca e leitura: {INFORMADO}. "
               f"Intervalo recomendado: {REGRA} ou padrão por tipo de óleo.")
    aviso_rodape()


def pagina_simulador(art, met, visitas, intervalos):
    st.title("Simulador: e se?")
    st.info("Escolha um veículo e mexa nos controles. A previsão muda porque o modelo aprendeu associações, não causas.")
    atual = situacao_atual(visitas, intervalos)
    usar_ml = st.radio("Previsão com", ["Regra escolhida na validação", "Melhor modelo de ML"], horizontal=True) == "Melhor modelo de ML"
    escolha = st.selectbox("Veículo", atual["veiculo"] + "  (" + atual["marca"].fillna("") + " " + atual["modelo"].fillna("") + ")")
    caso = atual[atual["veiculo"] == escolha.split("  (")[0]].reset_index(drop=True)
    unidade = caso.loc[0, "unidade"]

    base, s = prever(art, caso, usar_ml)
    st.markdown(f"**Previsão atual:** {base[0]:.0f} dias após a última troca ({caso.loc[0, 'data_ini'].strftime('%d/%m/%Y')}) "
                f"→ {(caso.loc[0, 'data_ini'] + timedelta(days=float(base[0]))).strftime('%d/%m/%Y')}, faixa ±{2 * s:.0f} dias")

    taxa_atual = float(caso.loc[0, "taxa_uso_ant"]) if pd.notna(caso.loc[0, "taxa_uso_ant"]) else 20.0
    c1, c2, c3 = st.columns(3)
    taxa = c1.slider(f"Ritmo de uso ({unidade}/dia)", 0.5, max(200.0, taxa_atual * 2), float(round(taxa_atual, 1)))
    rec_unid = c2.number_input(f"Intervalo recomendado ({unidade})", 100.0, 30000.0, float(caso.loc[0, "intervalo_rec_unid"]), 100.0)
    rec_meses = c3.number_input("Intervalo recomendado (meses)", 1.0, 24.0, float(caso.loc[0, "intervalo_rec_meses"]), 1.0)
    c4, c5 = st.columns(2)
    tipo = c4.selectbox("Tipo de óleo", ["Mineral", "Semissintético", "Sintético"],
                        index=["Mineral", "Semissintético", "Sintético"].index(caso.loc[0, "tipo_oleo"]) if caso.loc[0, "tipo_oleo"] in ["Mineral", "Semissintético", "Sintético"] else 2)
    app = c5.selectbox("Carro de aplicativo?", ["Não", "Sim"], index=1 if caso.loc[0, "aplicativo"] == "Sim" else 0)

    modelo = art["modelo_ml"] if usar_ml else art["modelo"]
    dias_regra = min(rec_unid / taxa, rec_meses * 30.4)
    novo = simular(modelo, caso, taxa_uso_ant=taxa, intervalo_rec_unid=rec_unid, intervalo_rec_meses=rec_meses,
                   tipo_oleo=tipo, aplicativo=app, dias_regra=dias_regra)
    st.markdown(f"### Nova previsão: {novo:.0f} dias ({novo - base[0]:+.0f} em relação à atual)")
    st.caption(f"Regra do fabricante com esses valores: {dias_regra:.0f} dias. {PREVISAO}: aprendeu com {met['tamanhos']['treino']} intervalos reais.")
    aviso_rodape()


def pagina_modelo(met):
    st.title("Sobre o modelo")
    st.warning(met["aviso"])
    t = met["tamanhos"]
    st.markdown(f"""
- **Escolhido na validação:** `{met['modelo_escolhido']}` ({nome_bonito(met['modelo_escolhido'])})
- **Melhor modelo de ML:** `{met['melhor_ml']}` (params: `{met.get('params_melhor_ml')}`)
- **Treino:** {t['treino']} intervalos de {t['veiculos_treino']} veículos, até {met['periodos']['treino'][1]} · **Teste:** {t['teste']} intervalos, de {met['periodos']['teste'][0]} a {met['periodos']['teste'][1]}
- **Por fonte:** {met['por_fonte']}
""")

    v = met.get("validacao", {})
    st.markdown(f"#### Validação cruzada temporal no treino ({v.get('tipo', '')})")
    st.caption("Dobras (treino → validação): " + " | ".join(f"{d['treino']} → {d['validacao']}" for d in v.get("dobras", []))
               + f". Nenhum intervalo de treino termina depois de um de validação. {v.get('intervalos_validados', '')} intervalos validados.")
    cv = pd.DataFrame(met["validacao_cruzada"]).T.sort_values("mae_dias")
    cv.index = [nome_bonito(i) for i in cv.index]
    st.dataframe(cv.rename(columns={"mae_dias": "MAE (dias)", "rmse_dias": "RMSE", "mape_pct": "MAPE %", "r2": "R²", "n": "n"}), width="stretch")

    st.markdown("#### Prova final (teste, aberto uma única vez)")
    te = pd.DataFrame(met["teste_final"]).T.sort_values("mae_dias")
    te.index = [nome_bonito(i) for i in te.index]
    st.dataframe(te.rename(columns={"mae_dias": "MAE (dias)", "rmse_dias": "RMSE", "mape_pct": "MAPE %", "r2": "R²", "n": "n"}), width="stretch")
    f = met["faixa"]
    st.caption(f"Faixa de previsão: resíduo típico s = {f['s_dias']} dias, faixa ±{f['margem_dias']} dias, cobre {100 * f['cobertura_teste']:.0f}% do teste.")

    c1, c2 = st.columns(2)
    for col, arq, leg in [(c1, "placar.png", "Placar no teste"), (c2, "validacao_cruzada.png", "Validação cruzada"),
                          (c1, "previsto_vs_real.png", "Previsto × real e resíduos"), (c2, "importancias.png", f"Importância por permutação ({met['explicacao_de']})"),
                          (c1, "optuna.png", "Ajuste fino com Optuna"), (c2, "decisao_custo.png", "Decisão: custo × antecedência")]:
        if (PASTA_RELATORIO / arq).exists():
            col.image(str(PASTA_RELATORIO / arq), caption=leg)

    d = met["decisao"]
    st.markdown(f"""
#### Decisão
Os custos são **hipotéticos**, escolhidos para demonstrar a decisão: perda do serviço a R$ {d['custo_tarde']:.0f} e contato a R$ {d['custo_cedo']:.0f}. Com eles, a menor soma de erros é lembrar **{d['antecedencia_dias']} dias antes** da data prevista.
Se o contato custar 5 vezes mais, passa para {d['antecedencia_contato_caro']} dias. O que importa é o mecanismo: a oficina coloca os custos dela e a antecedência sai sozinha. O modelo estima a data; a antecedência é decisão da oficina.

#### Limitações
- Base pequena, de quatro fontes; uma delas (máquinas da prefeitura) medida em horas.
- Só entram veículos que voltaram: o sistema prevê o retorno de clientes que voltam, não quando o óleo precisa ser trocado.
- Com poucos exemplos, regras simples do negócio empataram ou venceram a regressão. Isso é um resultado, não um defeito.
- Importância não é causa; o simulador mostra o que o modelo prevê, não o que acontece se a oficina mudar algo.
""")
    aviso_rodape()


def main():
    st.set_page_config(page_title="Retorno Inteligente", page_icon="🛢️", layout="wide")
    if not CAMINHO_MODELO.exists():
        st.error("Modelo não encontrado. Rode `python src/preparar_dados.py` e `python src/treinar.py`."); st.stop()
    art, met = carregar_artefato(), carregar_metricas()
    visitas, intervalos = carregar_dados()
    st.sidebar.title("🛢️ Retorno Inteligente")
    pagina = st.sidebar.radio("Página", ["Lista da semana", "Simulador", "Sobre o modelo"])
    st.sidebar.caption("Nanodegree IA & ML, Atitus · dados reais de 4 fontes, pseudonimizados")
    if pagina == "Lista da semana":
        pagina_lista(art, met, visitas, intervalos)
    elif pagina == "Simulador":
        pagina_simulador(art, met, visitas, intervalos)
    else:
        pagina_modelo(met)


if __name__ == "__main__":
    main()
