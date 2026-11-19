"""
preparar_dados.py — Lê as planilhas das oficinas, unifica, limpa e gera os intervalos de treino.

Entrada:  data/brutos/*.xlsx (aba "Visitas", formato da planilha de coleta)
Saída:    data/visitas.csv    (uma linha por visita, colunas padronizadas)
          data/intervalos.csv (uma linha por intervalo entre duas trocas de óleo do mesmo veículo)

Execução:
    python src/preparar_dados.py

Todos os dados são reais e pseudonimizados.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA_BRUTOS = RAIZ / "data" / "brutos"
CAMINHO_VISITAS = RAIZ / "data" / "visitas.csv"
CAMINHO_INTERVALOS = RAIZ / "data" / "intervalos.csv"

# Veículos a excluir do estudo, se algum dia for preciso: {"FONTE_CODIGO": "motivo"}
EXCLUIR: dict = {}

# Fontes medidas em horas de horímetro (máquinas), não em km
FONTES_EM_HORAS = {"PREFEITURA"}

# Limite acima do qual o intervalo é considerado "troca perdida" (trocou em outro lugar no meio)
LIMITE_PERDIDA = {"km": 15000, "h": 600}
# Mais de 2 anos sem troca registrada também é registro faltante (nenhum fabricante recomenda mais de 12 meses)
LIMITE_PERDIDA_DIAS = 730

# Intervalo recomendado padrão quando a oficina não informou (por tipo de óleo)
PADRAO_RECOMENDADO = {
    "km": {"Mineral": (5000, 6), "Semissintético": (7500, 12), "Sintético": (10000, 12), None: (10000, 12)},
    "h": {None: (250, 6)},
}

MAPA_COLUNAS = {
    "Oficina": "oficina",
    "Código do carro": "codigo",
    "Marca": "marca",
    "Modelo": "modelo",
    "Ano do carro": "ano",
    "Motor (cilindrada)": "motor",
    "Combustível": "combustivel",
    "Data da visita": "data",
    "Km no painel": "leitura",
    "Serviço feito": "servico",
    "Tipo de óleo": "tipo_oleo",
    "Viscosidade do óleo": "viscosidade",
    "Intervalo recomendado (km)": "intervalo_rec_unid",
    "Intervalo recomendado (meses)": "intervalo_rec_meses",
    "Uso do carro": "uso",
    "Carro de aplicativo?": "aplicativo",
    "Dorme na garagem?": "garagem",
    "Cidade do cliente": "cidade",
    "Voltou no prazo?": "voltou_no_prazo",
    "Observações": "observacoes",
}


# ---------------------------------------------------------------------------
# Leitura e unificação
# ---------------------------------------------------------------------------
def ler_planilha(caminho: Path) -> pd.DataFrame:
    df = pd.read_excel(caminho, sheet_name="Visitas")
    # a segunda linha do modelo é a descrição das colunas
    if len(df) and str(df.iloc[0, 0]).startswith("Nome da oficina"):
        df = df.iloc[1:]
    df = df.rename(columns=MAPA_COLUNAS)
    df["fonte"] = caminho.stem.upper()
    return df


def eh_troca_oleo(servico) -> bool:
    """Regra única de limpeza: troca de óleo (com ou sem filtros). 'Troca de filtro de óleo' não conta."""
    s = str(servico).strip().lower()
    s = s.replace("ó", "o")
    return s.startswith("troca de oleo")


def unificar(brutos: pd.DataFrame) -> pd.DataFrame:
    df = brutos.copy()
    df["veiculo"] = df["fonte"] + "_" + df["codigo"].astype(str).str.strip()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["leitura"] = pd.to_numeric(df["leitura"], errors="coerce")
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["intervalo_rec_unid"] = pd.to_numeric(df["intervalo_rec_unid"], errors="coerce")
    df["intervalo_rec_meses"] = pd.to_numeric(df["intervalo_rec_meses"], errors="coerce")
    df["unidade"] = np.where(df["fonte"].isin(FONTES_EM_HORAS), "h", "km")
    df["troca_oleo"] = df["servico"].apply(eh_troca_oleo)

    for col in ["tipo_oleo", "viscosidade", "uso", "aplicativo", "garagem", "cidade", "marca", "modelo", "motor", "combustivel", "oficina"]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA

    # frota: observação da Cirocar ("Carro de frota de empresa") ou máquinas
    obs = df["observacoes"].astype("string").fillna("").str.lower()
    df["frota"] = np.where(obs.str.contains("frota") | (df["unidade"] == "h"), "Sim", "Não")

    # Exclusões
    excluidos = df["veiculo"].isin(EXCLUIR)
    df = df[~excluidos].copy()

    df = df.dropna(subset=["data"])
    df = df.sort_values(["veiculo", "data"]).reset_index(drop=True)
    colunas = ["fonte", "oficina", "veiculo", "marca", "modelo", "ano", "motor", "combustivel", "data", "leitura",
               "unidade", "servico", "troca_oleo", "tipo_oleo", "viscosidade", "intervalo_rec_unid",
               "intervalo_rec_meses", "uso", "aplicativo", "frota", "garagem", "cidade", "voltou_no_prazo", "observacoes"]
    return df[colunas]


# ---------------------------------------------------------------------------
# Intervalos
# ---------------------------------------------------------------------------
def recomendado_padrao(unidade: str, tipo_oleo) -> tuple:
    tabela = PADRAO_RECOMENDADO[unidade]
    chave = tipo_oleo if tipo_oleo in tabela else None
    return tabela[chave]


def gerar_intervalos(visitas: pd.DataFrame) -> pd.DataFrame:
    """Um intervalo por par de trocas de óleo consecutivas com leitura, por veículo.

    Features usam apenas visitas com data <= data_ini (sem vazamento).
    """
    linhas = []
    for veiculo, g in visitas.groupby("veiculo", sort=False):
        g = g.sort_values("data").reset_index(drop=True)
        trocas = g[g["troca_oleo"] & g["leitura"].notna()].reset_index(drop=True)
        if len(trocas) < 2:
            continue
        unidade = g["unidade"].iloc[0]
        limite = LIMITE_PERDIDA[unidade]
        historico_intervalos = []  # intervalos válidos anteriores (dias, unid)

        for k in range(1, len(trocas)):
            ini, fim = trocas.iloc[k - 1], trocas.iloc[k]
            dias = (fim["data"] - ini["data"]).days
            delta = fim["leitura"] - ini["leitura"]

            if delta <= 0 or dias <= 0:
                situacao = "erro"
            elif delta > limite or dias > LIMITE_PERDIDA_DIAS:
                situacao = "perdida"
            else:
                situacao = "valido"

            # ritmo de uso com tudo que se sabia até data_ini (inclusive visitas não-óleo)
            ate_ini = g[(g["data"] <= ini["data"]) & g["leitura"].notna()]
            primeira = ate_ini.iloc[0]
            dias_hist = (ini["data"] - primeira["data"]).days
            unid_hist = ini["leitura"] - primeira["leitura"]
            taxa_uso_ant = unid_hist / dias_hist if dias_hist > 0 and unid_hist > 0 else np.nan

            n_trocas_ant = int((trocas["data"] < ini["data"]).sum()) + 1  # inclui a troca de data_ini
            if historico_intervalos:
                intervalo_ant_dias, intervalo_ant_unid = historico_intervalos[-1]
            else:
                intervalo_ant_dias, intervalo_ant_unid = np.nan, np.nan

            tipo_oleo = ini["tipo_oleo"] if pd.notna(ini["tipo_oleo"]) else None
            rec_unid, rec_meses = ini["intervalo_rec_unid"], ini["intervalo_rec_meses"]
            padrao_unid, padrao_meses = recomendado_padrao(unidade, tipo_oleo)
            rec_informado = pd.notna(rec_unid) or pd.notna(rec_meses)
            if pd.isna(rec_unid):
                rec_unid = padrao_unid
            if pd.isna(rec_meses):
                rec_meses = padrao_meses

            idade = ini["data"].year - ini["ano"] if pd.notna(ini["ano"]) else np.nan

            # a conta da regra do fabricante com os dados do próprio veículo (vazia se não há taxa de uso)
            if pd.notna(taxa_uso_ant) and taxa_uso_ant > 0:
                dias_regra = min(rec_unid / taxa_uso_ant, rec_meses * 30.4)
            else:
                dias_regra = np.nan

            linhas.append({
                "fonte": ini["fonte"], "oficina": ini["oficina"], "veiculo": veiculo,
                "marca": ini["marca"], "modelo": ini["modelo"], "unidade": unidade,
                "data_ini": ini["data"], "leitura_ini": ini["leitura"],
                "data_fim": fim["data"], "leitura_fim": fim["leitura"],
                "dias": dias, "delta_unid": delta, "situacao": situacao,
                "taxa_uso_ant": taxa_uso_ant,
                "intervalo_ant_dias": intervalo_ant_dias, "intervalo_ant_unid": intervalo_ant_unid,
                "n_trocas_ant": n_trocas_ant, "idade_veic": idade, "dias_regra": dias_regra,
                "tipo_oleo": tipo_oleo, "viscosidade": ini["viscosidade"],
                "intervalo_rec_unid": float(rec_unid), "intervalo_rec_meses": float(rec_meses),
                "rec_informado": bool(rec_informado),
                "uso": ini["uso"], "aplicativo": ini["aplicativo"], "frota": ini["frota"],
            })
            if situacao == "valido":
                historico_intervalos.append((dias, delta))

    iv = pd.DataFrame(linhas)
    return iv.sort_values(["data_fim", "veiculo"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def main() -> None:
    arquivos = sorted(PASTA_BRUTOS.glob("*.xlsx"))
    if not arquivos:
        raise SystemExit(f"Nenhuma planilha em {PASTA_BRUTOS}")
    brutos = pd.concat([ler_planilha(a) for a in arquivos], ignore_index=True)
    print(f"Planilhas lidas: {[a.name for a in arquivos]} | {len(brutos)} linhas brutas")

    visitas = unificar(brutos)
    for v, motivo in EXCLUIR.items():
        print(f"Excluído {v}: {motivo}")
    visitas.to_csv(CAMINHO_VISITAS, index=False)

    print(f"\nVisitas válidas: {len(visitas)} | veículos: {visitas['veiculo'].nunique()}")
    print(f"Trocas de óleo: {visitas['troca_oleo'].sum()} | com leitura: {(visitas['troca_oleo'] & visitas['leitura'].notna()).sum()}")
    print("\nCélulas vazias por coluna (visitas):")
    print(visitas[["ano", "leitura", "tipo_oleo", "viscosidade", "uso", "aplicativo", "intervalo_rec_unid"]].isna().sum().to_string())

    intervalos = gerar_intervalos(visitas)
    intervalos.to_csv(CAMINHO_INTERVALOS, index=False)

    print(f"\nIntervalos gerados: {len(intervalos)}")
    print(intervalos.groupby(["fonte", "situacao"]).size().unstack(fill_value=0).to_string())
    validos = intervalos[intervalos["situacao"] == "valido"]
    print(f"\nVÁLIDOS: {len(validos)} intervalos de {validos['veiculo'].nunique()} veículos | "
          f"com intervalo anterior: {validos['intervalo_ant_dias'].notna().sum()} | "
          f"com taxa de uso anterior: {validos['taxa_uso_ant'].notna().sum()}")
    print(f"Dias até a próxima troca (válidos): mediana {validos['dias'].median():.0f}, "
          f"mín {validos['dias'].min()}, máx {validos['dias'].max()}")
    print(f"\nArquivos: {CAMINHO_VISITAS}\n          {CAMINHO_INTERVALOS}")


if __name__ == "__main__":
    main()
