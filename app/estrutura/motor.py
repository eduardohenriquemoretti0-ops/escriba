"""Motor de estrutura: quem fala cada trecho, guiado pela ESTRUTURA da reunião.

Princípio (regra do Eduardo): a estrutura manda, a voz só confirma.
  - Quem fala vem do que é DITO: o presidente anuncia, o dirigente chama,
    a fala seguinte é de quem foi chamado; o agradecimento confirma.
  - O agrupamento de voz (cluster do motor de transcrição) só desempata:
    separa o condutor dos comentaristas e escolhe entre dois homônimos.
  - Trecho que nenhuma regra explica sai "[?]" com o tempo, nunca inventado.

ENTRADA  analisar(segmentos, lista_json, papeis=None, tipo=None)
         segmentos: [{"inicio": s, "fim": s, "texto": "...", "voz": "A"}, ...]
         (qualquer motor: AssemblyAI, local, o que vier depois — só texto+tempo+voz)
SAÍDA    Resultado: falas (um item por trecho, trechos podem ser divididos),
         blocos (a estrutura: partes e quem fez), pendencias (o que ouvir, com tempo).

Este arquivo é ESTRUTURAL. Não corrija aqui uma frase ou um nome específico:
isso vai em regras/religiosa.json ou na lista de irmãos.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

from .eventos import NOME_RX, Evento, Regras
from .nomes import Lista, Pessoa, Resolucao, fold, fonetica

COND = ("chamada", "agradecimento", "paragrafo", "pergunta")
# eventos que mostram que é FALA (não letra de cântico)
FALA_NAO_LETRA = ["designacao", "fecha_parte", "paragrafo", "agradecimento", "encerramento",
                  "recapitulacao", "tema_discurso", "oracao_designada", "pergunta", "convite_dirigente"]
CORO = "Em coro"
FORA = "(fora da reunião)"
VOZ_CORO = "CORO"          # rótulo que o motor de transcrição dá ao canto da assistência


def hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def npal(texto: str) -> int:
    return len(texto.split())


@dataclass
class Fala:
    ini: float
    fim: float
    texto: str
    voz: str = ""
    seg: int = -1                  # índice do segmento original
    quem: str = ""
    papel: str = ""
    parte: str = ""
    confianca: str = ""
    duvida: bool = False
    motivo: str = ""

    def marca(self, quem: str, papel: str, parte: str, confianca: str, motivo: str, duvida: bool = False):
        self.quem, self.papel, self.parte, self.confianca, self.motivo, self.duvida = quem, papel, parte, confianca, motivo, duvida


@dataclass
class Bloco:
    parte: str
    papel: str
    quem: str
    ini: float
    fim: float
    duvida: bool = False


@dataclass
class Resultado:
    tipo: str
    falas: list
    blocos: list
    pendencias: list = field(default_factory=list)

    def como_dict(self) -> dict:
        return {"tipo": self.tipo, "falas": [asdict(f) for f in self.falas],
                "blocos": [asdict(b) for b in self.blocos], "pendencias": self.pendencias}


class Contexto:
    """Estado compartilhado: falas, regras, lista, e utilidades de divisão/resolução."""

    def __init__(self, falas: list[Fala], regras: Regras, lista: Lista):
        self.F = falas
        self.R = regras
        self.lista = lista
        self.blocos: list[Bloco] = []
        self.vozes_pessoa: dict[str, Counter] = {}   # pessoa -> vozes já atribuídas com segurança
        self._cache: dict[str, list[Evento]] = {}
        # quem o Eduardo digitou em "participaram da reunião": vence a lista geral
        self.participantes: Optional[Lista] = None

    # ------------------------------------------------------------ eventos
    def ev(self, f: Fala, tipos=None) -> list[Evento]:
        if f.texto not in self._cache:
            self._cache[f.texto] = self.R.eventos(f.texto)
        evs = self._cache[f.texto]
        return [e for e in evs if tipos is None or e.tipo in tipos]

    # ------------------------------------------------------------ divisão
    def dividir(self, i: int, pos: int) -> bool:
        """Divide a fala i no caractere pos (ajustado ao espaço). Tempo proporcional ao texto."""
        f = self.F[i]
        t = f.texto
        while 0 < pos < len(t) and not t[pos - 1].isspace() and not t[pos].isspace():
            pos += 1
        a, b = t[:pos].strip(), t[pos:].strip()
        if not a or not b:
            return False
        meio = f.ini + (f.fim - f.ini) * (len(a) / max(1, len(t)))
        # as metades herdam a marcação (quem dividiu remarca a parte que muda de dono)
        marca = dict(quem=f.quem, papel=f.papel, parte=f.parte, confianca=f.confianca,
                     duvida=f.duvida, motivo=f.motivo)
        f1 = Fala(f.ini, meio, a, f.voz, f.seg, **marca)
        f2 = Fala(meio, f.fim, b, f.voz, f.seg, **marca)
        self.F[i:i + 1] = [f1, f2]
        return True

    # ------------------------------------------------------------ nomes
    def resolver(self, ouvido: Optional[str], genero: Optional[str] = None,
                 vozes: Optional[Counter] = None) -> Optional[Resolucao]:
        if not ouvido:
            return None
        r = self.lista.resolver(ouvido, genero)
        if self.participantes is not None:
            r = self._pelos_participantes(ouvido, genero, r)
        # voz confirma: entre homônimos, fica quem já falou com essa voz nesta reunião
        if r.duvida and r.candidatos and vozes:
            def afinidade(p: Pessoa) -> float:
                vp = self.vozes_pessoa.get(p.nome)
                if not vp:
                    return 0.0
                tot = sum(vozes.values())
                return sum(min(vozes[v], 1) * vozes[v] for v in vozes if v in vp) / max(1, tot)
            notas = sorted(((afinidade(p), p) for p in r.candidatos), key=lambda x: -x[0])
            if notas[0][0] >= 0.5 and (len(notas) == 1 or notas[1][0] < 0.2):
                return Resolucao(ouvido, notas[0][1], [], "media",
                                 f"{r.motivo}; voz igual à de {notas[0][1].nome} nesta reunião")
        return r

    def _pelos_participantes(self, ouvido: str, genero: Optional[str], r: Resolucao) -> Resolucao:
        """Nome ouvido casado primeiro com os participantes informados da reunião.

        Um só participante casa -> é ele (mesmo que a lista geral diga outro ou fique em
        dúvida). Dois casam ("Arlete" com Arlete Hoenne e Arlete Oliveira) -> a dúvida
        fica só entre eles. Nenhum casa -> vale a lista geral, como antes."""
        rp = self.participantes.resolver(ouvido, genero)
        if rp.pessoa is None and not rp.candidatos:
            # apelido/nome cortado só entre os participantes: "Will" William, "Rafa" Rafael
            t = fold(ouvido.split()[0]) if ouvido.split() else ""
            pref = [p for p in self.participantes.pessoas
                    if len(t) >= 4 and fold(p.tokens[0]).startswith(t) and fold(p.tokens[0]) != t]
            if len(pref) == 1:
                rp = Resolucao(ouvido, pref[0], [], "media", f"'{ouvido}' é o começo de {pref[0].nome}")
        if rp.pessoa is None and not rp.candidatos and len(ouvido.split()) >= 2:
            # "Zé Eduardo": uma palavra do ouvido bate com uma palavra de um único participante
            achou = [p for p in self.participantes.pessoas
                     if any(len(o) >= 4 and any(fonetica(o) == fonetica(t) for t in p.tokens)
                            for o in ouvido.split())]
            if len(achou) == 1:
                rp = Resolucao(ouvido, achou[0], [], "media", f"'{ouvido}' casa com {achou[0].nome}")
        if rp.pessoa is not None:
            if r.pessoa is not None and fold(r.pessoa.nome) == fold(rp.pessoa.nome):
                return r
            conf = "media" if rp.confianca != "baixa" else "baixa"
            return Resolucao(ouvido, rp.pessoa, [], conf, f"participante informado na reunião ({rp.motivo})")
        if len(rp.candidatos) > 1 and rp.motivo != "fora da lista (parecido)":
            return Resolucao(ouvido, None, rp.candidatos, "baixa", f"{rp.motivo} entre os participantes")
        return r

    def aprende_voz(self, pessoa: str, vozes: Counter):
        self.vozes_pessoa.setdefault(pessoa, Counter()).update(vozes)

    def bloco(self, parte, papel, quem, i0, i1, duvida=False):
        if i0 < i1 <= len(self.F):
            self.blocos.append(Bloco(parte, papel, quem, self.F[i0].ini, self.F[i1 - 1].fim, duvida))


def genero_do_titulo(texto_evento: str) -> Optional[str]:
    t = fold(texto_evento)
    if re.search(r"\b(irma|irmas|dona|senhora)\b", t):
        return "f"
    if re.search(r"\b(irmao|senhor)\b", t):
        return "m"
    return None


def marca_pessoa(f: Fala, r: Optional[Resolucao], papel: str, parte: str, motivo: str, conf_max: str = "alta"):
    if r is None:
        f.marca("[?]", papel, parte, "baixa", motivo + " · ninguém foi chamado pelo nome: ouça e diga quem é", True)
        return
    ordem = {"baixa": 0, "media": 1, "alta": 2}
    conf = min(r.confianca, conf_max, key=lambda c: ordem[c])
    detalhe = motivo if not r.duvida else f"{motivo} · ouvido '{r.ouvido}': {r.motivo}"
    f.marca(r.nome, papel, parte, conf, detalhe, r.duvida)


# =====================================================================
#  CÂNTICO e ORAÇÃO
# =====================================================================
def cantar(ctx: Contexto, i: int, parte: str) -> int:
    """i = fala que anuncia o cântico. Marca a letra cantada como 'Em coro'.
    Devolve o índice da primeira fala depois do cântico."""
    F, R = ctx.F, ctx.R
    evs = [e for e in ctx.ev(F[i], ["cantico"])]
    if evs:
        e = evs[-1]
        t = F[i].texto
        # fim do anúncio: fim da frase, mais "Tem como base ..." / "E tem por tema ..." / "Os que puderem ..."
        m = re.search(r"[.!?](\s|$)", t[e.fim:])
        corte = e.fim + (m.end() if m else len(t) - e.fim)
        while True:
            m2 = re.match(r"\s*(?i:(?:e\s+)?tem\s+(?:como\s+)?(?:por\s+)?(?:base|tema)|com\s+base|os\s+que\s+puderem|se\s+puderem|aqueles\s+que\s+puderem|ap[óo]s\s+o\s+c[âa]ntico|e\s+no\s+final)[^.!?]*[.!?]?", t[corte:])
            if not m2 or not m2.group(0).strip():
                break
            corte += m2.end()
        resto = t[corte:]
        if npal(resto) >= 8 and not R.tem("oracao_designada", resto) and not R.tem("designacao", resto):
            if ctx.dividir(i, corte):
                F[i + 1].marca(CORO, "cântico", parte, "alta", "letra cantada depois do anúncio do cântico")
                i += 1
    j = i + 1
    if j >= len(F):
        return j
    ini_canto = F[j].ini
    while j < len(F):
        f = F[j]
        if j > i + 1:
            pausa = f.ini - F[j - 1].fim
            if R.tem("oracao_invocacao", f.texto) or ctx.ev(f, FALA_NAO_LETRA):
                break
            if (f.ini - ini_canto) >= R.lim("cantico_min_seg", 40) and pausa >= R.lim("cantico_pausa_fim_seg", 4.0):
                break
            if (f.ini - ini_canto) > R.lim("cantico_max_seg", 330):
                break
        elif R.tem("oracao_invocacao", f.texto):
            break
        f.marca(CORO, "cântico", parte, "alta", "letra cantada depois do anúncio do cântico")
        j += 1
    return j


def orar(ctx: Contexto, i: int, quem: Optional[Resolucao], nome_fixo: Optional[str], papel: str, parte: str) -> int:
    """Marca a oração a partir de i até o 'Amém'. Devolve o índice seguinte."""
    F = ctx.F
    j = i
    vozes = Counter()
    while j < len(F):
        if j > i and npal(F[j].texto) <= 2 and ctx.R.tem("amem", F[j].texto):
            break      # "Amém." curto de outra voz = assistência (marcado depois como Em coro)
        vozes[F[j].voz] += 1
        termina = ctx.R.tem("amem", F[j].texto)
        j += 1
        if termina:
            break
    if quem is not None and quem.duvida:
        quem = ctx.resolver(quem.ouvido, None, vozes) or quem
    for k in range(i, j):
        if nome_fixo:
            F[k].marca(nome_fixo, papel, parte, "alta", "estrutura")
        else:
            marca_pessoa(F[k], quem, papel, parte, "anunciado para fazer a oração")
    ctx.bloco(parte, papel, F[i].quem if i < len(F) else "", i, j, F[i].duvida if i < len(F) else True)
    return j


# =====================================================================
#  PERGUNTAS E RESPOSTAS (Sentinela, Joias, consideração, estudo bíblico, recapitulação)
# =====================================================================
def perguntas_e_respostas(ctx: Contexto, i0: int, fim_t: float, condutor: str, cond_duvida: bool,
                          papel_cond: str, parte: str, leitor: Optional[Resolucao] = None,
                          suprimir_chamadas: bool = False) -> int:
    """Turnos guiados pelo condutor. Devolve o índice da primeira fala depois da janela.

    estado 'condutor' -> o condutor fala; quem fala sem ser chamado sai [?] com o tempo
    estado 'chamado'  -> a fala seguinte é de quem foi chamado (nome -> lista)
    estado 'leitura'  -> depois da dica de parágrafo, quem lê é o leitor
    O agradecimento ("Obrigado, X") confirma, completa ou corrige quem acabou de falar.
    """
    F, R = ctx.F, ctx.R
    lim = R.lim

    def dentro(i):
        return i < len(F) and F[i].ini < fim_t

    def valida(e: Evento) -> bool:
        """Chamada de padrão fraco/médio só vale se o nome existir (na lista ou parecido)."""
        if suprimir_chamadas and e.tipo == "chamada":
            return False
        if e.tipo != "chamada" or e.forca == "forte":
            return True
        r = ctx.resolver(e.nome)
        if r is None:
            return False
        return (not r.duvida) if e.forca == "fraca" else (not r.duvida or bool(r.candidatos))

    def cond_evs(f: Fala) -> list[Evento]:
        evs = [e for e in ctx.ev(f, COND) if valida(e)]
        if leitor is not None:      # "convidado para hoje, irmão X" = começa a leitura
            dicas = [Evento("paragrafo", e.ini, e.fim, None, None, "forte") for e in ctx.ev(f, ["leitor"])
                     if e.nome and ctx.resolver(e.nome) is not None and ctx.resolver(e.nome).nome == leitor.nome]
            if dicas:
                evs = [e for e in evs if not (e.tipo == "chamada" and any(d.ini <= e.ini <= d.fim for d in dicas))]
                evs += dicas
            # chamar o leitor pelo nome ("Por favor, Fernandinho") = começa a leitura
            prim_leitor = (leitor.ouvido if leitor.duvida else leitor.nome).split()[0]
            evs = [Evento("paragrafo", e.ini, e.fim, None, None, "forte")
                   if e.tipo == "chamada" and e.nome and _mesmo_nome(e.nome.split()[0], prim_leitor)
                   else e for e in evs]
        return sorted(evs, key=lambda e: e.ini)

    # --- vozes do condutor: as que mais carregam chamada/agradecimento/parágrafo
    tot, comev = Counter(), Counter()
    i = i0
    while dentro(i):
        tot[F[i].voz] += 1
        if cond_evs(F[i]):
            comev[F[i].voz] += 1
        i += 1
    vozes_cond = {v for v in tot if comev[v] >= lim("voz_condutor_min_segmentos", 3)
                  and comev[v] / tot[v] >= lim("voz_condutor_fracao", 0.30)}
    vozes_leitor: Counter = Counter()

    estado = "abertura"             # até a primeira chamada/dica, quem fala é o condutor
    alvo: Optional[Resolucao] = None
    bloco: list[int] = []           # falas do comentário em curso
    ultimo_era_condutor = False

    def marca_bloco(idx: list[int], r: Optional[Resolucao], motivo: str, conf_max: str):
        if not idx:
            return
        vz = Counter(F[k].voz for k in idx)
        if r is not None and r.duvida:
            r = ctx.resolver(r.ouvido, None, vz) or r
        for k in idx:
            marca_pessoa(F[k], r, "comentarista", parte, motivo, conf_max)
        if r is not None and not r.duvida:
            ctx.aprende_voz(r.pessoa.nome, vz)

    def fecha_bloco(confirmado: Optional[Resolucao] = None):
        nonlocal bloco, alvo
        idx, chamado = bloco, alvo
        bloco, alvo = [], None
        if not idx:
            return
        if confirmado is None:
            marca_bloco(idx, chamado, "chamado pelo condutor" if chamado else "comentário sem chamada ouvida", "alta")
            return
        if chamado is None:
            marca_bloco(idx, confirmado, "agradecido pelo condutor (chamada não ouvida)", "media")
            return
        mesmo = (not chamado.duvida and not confirmado.duvida and chamado.pessoa.nome == confirmado.pessoa.nome) \
            or (chamado.ouvido and confirmado.ouvido and fold(chamado.ouvido) == fold(confirmado.ouvido))
        if mesmo:
            marca_bloco(idx, chamado if not chamado.duvida else confirmado, "chamado e agradecido pelo condutor", "alta")
            return
        if confirmado.duvida and not chamado.duvida:
            marca_bloco(idx, chamado, f"chamado pelo condutor (agradecido como '{confirmado.ouvido}')", "media")
            return
        # simétrico: chamada em dúvida, agradecimento claro -> vale o agradecimento
        # (ex.: chamado "Ana Marinês?" vago, agradecido "irmã Marines Silveira")
        if chamado.duvida and not confirmado.duvida:
            marca_bloco(idx, confirmado, f"agradecido pelo condutor (chamada em dúvida, '{chamado.ouvido}')", "media")
            return
        # nomes diferentes: o agradecimento vale para quem falou por último (mesma voz no fim do bloco)
        v_fim = F[idx[-1]].voz
        corte = len(idx)
        while corte > 0 and F[idx[corte - 1]].voz == v_fim:
            corte -= 1
        if corte > 0:
            marca_bloco(idx[:corte], chamado, "chamado pelo condutor", "media")
            marca_bloco(idx[corte:], confirmado, "agradecido pelo condutor (outra voz depois do chamado)", "media")
            return
        for k in idx:
            F[k].marca(f"[?] {chamado.nome} ou {confirmado.nome}?", "comentarista", parte, "baixa",
                       f"chamado '{chamado.ouvido}', agradecido '{confirmado.ouvido}': ouça e confirme", True)

    def eh_condutor(f: Fala, conf: str, motivo: str):
        nonlocal ultimo_era_condutor
        f.marca(condutor, papel_cond, parte, conf, motivo, cond_duvida)
        ultimo_era_condutor = True

    def proximo_e_condutor(i) -> bool:
        return dentro(i + 1) and (F[i + 1].voz in vozes_cond or bool(cond_evs(F[i + 1])))

    i = i0
    while dentro(i):
        f = F[i]
        if f.quem == CORO:
            i += 1
            continue
        apres = ctx.ev(f, ["apresentacao"])
        cond = cond_evs(f)

        # 1) a pessoa diz o próprio nome ("É Gláucia.")
        if apres and not cond:
            fecha_bloco()
            alvo = ctx.resolver(apres[0].nome)
            estado = "chamado"
            bloco = [i]
            ultimo_era_condutor = False
            i += 1
            continue

        # 2·) comentário e agradecimento do condutor no mesmo trecho: divide antes do agradecimento
        if cond and cond[0].tipo == "agradecimento" and npal(f.texto[:cond[0].ini]) >= 12 and estado in ("chamado", "condutor"):
            ini_ag = cond[0].ini
            if ctx.dividir(i, ini_ag):
                if estado == "condutor" and not bloco:
                    alvo = None
                    estado = "chamado"
                bloco.append(i)
                ultimo_era_condutor = False
                i += 1
                continue

        # 2a) chamada/dica no FIM de um trecho que continua com a resposta/leitura: divide depois dela
        if cond and cond[-1].tipo in ("chamada", "paragrafo"):
            e1 = cond[-1]
            m = re.match(r"\s*[º°]?\s*(?:,?\s*(?i:por\s+favor|por\s+gentileza|pelo\s+zoom))?\s*[,.:?!]?", f.texto[e1.fim:])
            corte = e1.fim + (m.end() if m else 0)
            resto = f.texto[corte:]
            minimo = lim("resto_chamada_min_palavras", 15) if e1.tipo == "chamada" else lim("resto_min_palavras", 8)
            if (npal(f.texto[:e1.ini]) > lim("cabeca_max_palavras", 8) and npal(resto) >= minimo
                    and not [e for e in R.eventos(resto, list(COND)) if valida(e)]):
                ctx.dividir(i, corte)
                f = F[i]
                cond = cond_evs(f)

        # 2b) cabeça com chamada/dica de parágrafo e o resto é a resposta/leitura: divide o trecho
        if cond and cond[0].tipo in ("chamada", "paragrafo"):
            e0 = cond[0]
            antes = npal(f.texto[:e0.ini])
            m = re.match(r"\s*[º°]?\s*(?:,?\s*(?i:por\s+favor|por\s+gentileza|pelo\s+zoom))?\s*[,.:?!]?", f.texto[e0.fim:])
            corte = e0.fim + (m.end() if m else 0)
            resto = f.texto[corte:]
            ev_resto = [e for e in R.eventos(resto, list(COND)) if valida(e)]
            resto_ok = not ev_resto or (ev_resto[0].tipo == "agradecimento" and npal(resto[:ev_resto[0].ini]) >= lim("resto_min_palavras", 8))
            if (antes <= lim("cabeca_max_palavras", 8) and npal(resto) >= lim("resto_min_palavras", 8) and resto_ok):
                if ctx.dividir(i, corte):
                    fecha_bloco()
                    eh_condutor(F[i], "alta", "chamada/dica no começo do trecho")
                    if e0.tipo == "chamada":
                        alvo = ctx.resolver(e0.nome, genero_do_titulo(f.texto[max(0, e0.ini - 8):e0.fim]))
                        estado = "chamado"
                    else:
                        estado = "leitura"
                    i += 1
                    f = F[i]
                    cond = []

        # 3) trecho com evento do condutor -> condutor; atualiza o estado
        if cond:
            eh_condutor(f, "alta", "chama, agradece ou dá o parágrafo")
            for e in cond:
                if e.tipo == "agradecimento":
                    fecha_bloco(ctx.resolver(e.nome, genero_do_titulo(f.texto[max(0, e.ini - 4):e.fim])))
                    estado = "condutor"
                elif e.tipo == "chamada":
                    fecha_bloco()
                    alvo = ctx.resolver(e.nome, genero_do_titulo(f.texto[max(0, e.ini - 8):e.fim]))
                    estado = "chamado"
                elif e.tipo == "paragrafo":
                    fecha_bloco()
                    estado = "leitura"
                elif e.tipo == "pergunta":
                    fecha_bloco()
                    estado = "condutor"
            i += 1
            continue

        palavras = npal(f.texto)
        curta = palavras <= lim("fala_condutor_curta_max_palavras", 14)

        # 4) "Isso." "Muito bem." "Exatamente." -> condutor (não fecha o comentário: o agradecimento pode vir)
        if R.e_retorno(f.texto):
            eh_condutor(f, "media", "retorno curto do condutor")
            i += 1
            continue

        # 5) fala curtinha entre duas falas do condutor -> condutor
        if ultimo_era_condutor and palavras <= lim("entre_condutor_max_palavras", 5) and proximo_e_condutor(i) and estado != "chamado":
            eh_condutor(f, "media", "fala curta entre falas do condutor")
            i += 1
            continue

        if estado == "leitura":
            if f.voz in vozes_cond and f.voz not in vozes_leitor and (curta or ("?" in f.texto and palavras <= 40)):
                eh_condutor(f, "media", "pergunta do condutor depois da leitura")
                estado = "condutor"
            else:
                if leitor is not None:
                    marca_pessoa(f, leitor, "leitor", parte, "lê depois da dica de parágrafo")
                    if not leitor.duvida:          # a voz do leitor já serve de desempate na mesma reunião
                        ctx.aprende_voz(leitor.pessoa.nome, Counter({f.voz: 1}))
                else:
                    f.marca("[?] leitor", "leitor", parte, "baixa", "lê depois da dica de parágrafo; leitor não anunciado", True)
                vozes_leitor[f.voz] += 1
                ultimo_era_condutor = False
        elif estado == "chamado":
            bloco_vozes = {F[k].voz for k in bloco}
            if (not bloco and alvo is not None and palavras <= 8
                    and any(fold(t) in fold(f.texto) for t in alvo.ouvido.split() if len(t) > 2)):
                eh_condutor(f, "media", "condutor repete o nome de quem chamou")
            elif bloco and f.voz in vozes_cond and f.voz not in bloco_vozes and not R.e_retorno(f.texto):
                fecha_bloco()
                eh_condutor(f, "media", "voz do condutor depois do comentário")
                estado = "condutor"
            else:
                bloco.append(i)
                ultimo_era_condutor = False
        elif estado == "abertura":
            eh_condutor(f, "media", "o condutor abre a parte")
        else:  # condutor
            leitor_voz = (vozes_leitor[f.voz] >= 3 and palavras >= lim("leitura_por_voz_min_palavras", 50))
            if f.voz in vozes_cond and not leitor_voz:
                eh_condutor(f, "media", "voz do condutor")
            elif leitor_voz:
                if leitor is not None:
                    marca_pessoa(f, leitor, "leitor", parte, "voz do leitor, leitura longa (dica de parágrafo não ouvida)", "media")
                else:
                    f.marca("[?] leitor", "leitor", parte, "baixa", "voz do leitor; leitor não anunciado", True)
                ultimo_era_condutor = False
            else:
                estado = "chamado"
                alvo = None
                bloco = [i]
                ultimo_era_condutor = False
        i += 1
    fecha_bloco()
    # vozes do condutor e do leitor ficam registradas: servem para sugerir nome pela voz depois
    if not cond_duvida:
        ctx.aprende_voz(condutor, Counter({v: comev[v] for v in vozes_cond}))
    return i


# =====================================================================
#  REUNIÃO DO FIM DE SEMANA
# =====================================================================
def _indice_no_tempo(ctx, t: float) -> int:
    for i, f in enumerate(ctx.F):
        if f.ini >= t - 1e-6:
            return i
    return len(ctx.F)


def _primeiro(ctx, tipo, i0=0, i1=None, depois_t=None):
    for i in range(i0, i1 if i1 is not None else len(ctx.F)):
        if depois_t is not None and ctx.F[i].ini < depois_t:
            continue
        evs = ctx.ev(ctx.F[i], [tipo])
        if evs:
            return i, evs
    return None, []


def _nome_mais_citado(ctx, tipo, i0, i1) -> Optional[str]:
    c = Counter()
    for i in range(i0, min(i1, len(ctx.F))):
        for e in ctx.ev(ctx.F[i], [tipo]):
            if e.nome:
                c[e.nome] += 1
    return c.most_common(1)[0][0] if c else None


def janela_presidente(ctx: Contexto, i0: int, i1: int, pres: str, pres_duvida: bool, parte: str):
    """Abertura/volta do presidente: cântico -> Em coro, oração -> presidente, o resto -> presidente."""
    F = ctx.F
    i = i0
    while i < i1 and i < len(F):
        if F[i].quem:      # já marcado (ex.: letra de cântico dividida)
            i += 1
            continue
        F[i].marca(pres, "presidente", parte, "alta" if not pres_duvida else "baixa",
                   "abre a reunião, anuncia o cântico e a oração", pres_duvida)
        if ctx.ev(F[i], ["cantico"]):
            antes = len(F)
            j = cantar(ctx, i, parte)
            i1 += len(F) - antes
            i = j
            continue
        i += 1
    return i1


# ---------------------------------------------------------------------
#  ÂNCORAS: NOME + PAPEL (regra do Eduardo, 2026-10-18)
#  Não existe roteiro de frases. O que existe é: um NOME dito junto do que a
#  pessoa vai fazer (orador, dirigente, leitor, oração). O motor crava o nome
#  e o papel, e a ordem da reunião diz onde cada um entra.
# ---------------------------------------------------------------------
_MENCAO = re.compile(r"(?i:\b(?:irm[ãa]o?s?|dona)\s+(?:(?:o|a|nosso|nossa)\s+)?)" + NOME_RX)
_SUFIXO_APELIDO = ("inho", "inha", "zinho", "zinha")


def _frase(texto: str, pos: int) -> tuple[int, int]:
    """Começo e fim da frase que contém pos."""
    ini = max(texto.rfind(c, 0, pos) for c in ".!?") + 1
    m = re.search(r"[.!?]", texto[pos:])
    fim = pos + m.end() if m else len(texto)
    while ini < len(texto) and texto[ini].isspace():
        ini += 1
    return ini, fim


def _mesmo_nome(palavra: str, nome: str) -> bool:
    """'Henrique' = 'Henrique'; 'Fernandinho' = 'Fernando'; grafias que soam igual."""
    if fonetica(palavra) == fonetica(nome):
        return True
    fp, fn = fold(palavra), fold(nome)
    if len(fn) >= 4 and fp.endswith(_SUFIXO_APELIDO):
        raiz = fn[:-1] if fn[-1] in "aeo" else fn
        return fp.startswith(raiz)
    return False


def cita(f: Fala, nome: Optional[str], ultima: bool = False) -> Optional[int]:
    """Posição em que o PRIMEIRO nome da pessoa é dito no trecho (None se não é dito)."""
    if not nome:
        return None
    toks = [t for t in nome.replace("[?]", "").replace("?", "").split() if t[:1].isupper()]
    if not toks:
        return None
    achado = None
    for m in re.finditer(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+", f.texto):
        if _mesmo_nome(m.group(0), toks[0]):
            achado = m.start()
            if not ultima:
                break
    return achado


def papel_perto(ctx: Contexto, texto: str, ini: int, fim: int) -> Optional[str]:
    """Que papel é dito junto do nome (na mesma frase, até papel_raio_chars)."""
    raio = ctx.R.lim("papel_raio_chars", 140)
    a, b = _frase(texto, ini)
    jan_ini, jan_fim = max(a, ini - raio), min(b, fim + raio)
    janela = fold(texto[jan_ini:jan_fim])
    melhor, dist = None, None
    for papel, palavras in ctx.R.papel_palavras.items():
        for w in palavras:
            for m in re.finditer(r"\b" + re.escape(w), janela):
                p = jan_ini + m.start()
                d = (ini - p) if p < ini else max(0, p - fim)
                if dist is None or d < dist:
                    melhor, dist = papel, d
    return melhor


def mencoes(ctx: Contexto, i0: int, i1: int, papel: Optional[str] = None) -> list[tuple[int, str]]:
    """'irmão X' / 'irmã X' ditos em F[i0:i1] (com o papel dito junto, se pedido)."""
    achados = []
    for i in range(i0, min(i1, len(ctx.F))):
        t = ctx.F[i].texto
        for m in _MENCAO.finditer(t):
            nome = ctx.R.limpa_nome(m.group("nome"), t, m.end("nome"))
            if not nome:
                continue
            if papel is None or papel_perto(ctx, t, m.start(), m.end()) == papel:
                achados.append((i, nome))
    return achados


def _mais_citado_no_papel(ctx: Contexto, papel: str, i0: int, i1: int, ultimo: bool = False,
                          primeiro: bool = False) -> Optional[str]:
    achados = mencoes(ctx, i0, i1, papel)
    if not achados:
        return None
    if ultimo or primeiro:
        return achados[-1 if ultimo else 0][1]
    return Counter(n for _, n in achados).most_common(1)[0][0]


def _ancora_discurso(ctx: Contexto, orador: Optional[str], i1: int, d_min: float) -> Optional[int]:
    """Última fala que diz o nome do orador antes de um trecho longo (>= d_min) sem ele:
    é o presidente 'relembrando o tema e o nome do orador e chamando-o'."""
    F = ctx.F
    cits = [i for i in range(len(F)) if cita(F[i], orador) is not None]
    for k, i in enumerate(cits):
        if i >= i1:
            break
        prox = F[cits[k + 1]].ini if k + 1 < len(cits) else F[-1].fim
        if prox - F[i].fim >= d_min:
            return i
    return None


def _monologo(ctx: Contexto, i1: int, d_min: float) -> Optional[int]:
    """Sem nome nenhum: o discurso é o trecho longo de uma mesma voz na 1ª metade.
    Devolve a fala ANTERIOR a ele (a do presidente que chama o orador)."""
    F = ctx.F
    k = 0
    while k < min(i1, len(F)):
        if F[k].voz == VOZ_CORO:
            k += 1
            continue
        j = k
        while j + 1 < len(F) and (F[j + 1].voz == F[k].voz or npal(F[j + 1].texto) <= 3):
            j += 1
        if F[j].fim - F[k].ini >= d_min:
            return k - 1 if k > 0 else None
        k = j + 1
    return None


def _divide_depois_da_frase(ctx: Contexto, i: int, pos: int, min_resto: int = 30) -> bool:
    """Divide a fala i no fim da frase que contém pos, se o resto for longo (outra pessoa)."""
    _, fim = _frase(ctx.F[i].texto, pos)
    return npal(ctx.F[i].texto[fim:]) >= min_resto and ctx.dividir(i, fim)


def _divide_antes_da_frase(ctx: Contexto, i: int, pos: int, min_antes: int = 15) -> bool:
    """Divide a fala i no começo da frase que contém pos, se antes dela houver fala longa."""
    ini, _ = _frase(ctx.F[i].texto, pos)
    return npal(ctx.F[i].texto[:ini]) >= min_antes and ctx.dividir(i, ini)


def fim_de_semana(ctx: Contexto, papeis: dict) -> None:
    """Boas-vindas -> anuncia orador e tema -> cântico -> oração -> relembra o tema e
    chama o orador -> discurso -> agradece o orador e chama o dirigente (às vezes o
    leitor) -> dirigente anuncia o cântico e quem lê -> estudo -> quem faz a oração
    final (antes do estudo ou depois do cântico final) -> cântico -> oração.
    Cada passagem é achada pelo NOME de quem vai fazer, não por frase fixa."""
    F, R = ctx.F, ctx.R
    if not F:
        return
    d_min = R.lim("discurso_min_seg", 600)
    d_max = R.lim("discurso_max_seg", 3000)
    i_meio = _indice_no_tempo(ctx, F[0].ini + (F[-1].fim - F[0].ini) * 0.5)

    # --- presidente: quem abre. Nome só se informado (o presidente não diz o próprio nome).
    pres_r = ctx.resolver(papeis.get("presidente")) if papeis.get("presidente") else None
    pres = pres_r.nome if pres_r else "[?] Presidente"
    pres_duv = pres_r is None or pres_r.duvida

    # --- orador: o cadastro; senão o nome dito junto de orador/discurso/tema na 1ª metade
    orador_ouvido = (papeis.get("orador") or _mais_citado_no_papel(ctx, "orador", 0, i_meio)
                     or _nome_mais_citado(ctx, "anuncio_orador", 0, i_meio))
    orador_r = ctx.resolver(orador_ouvido)
    if orador_r is not None and orador_r.duvida:
        # visitante: entra na lista desta reunião COMO OUVIDO, marcado para confirmar
        visitante = Pessoa(orador_ouvido, "orador anunciado nesta reunião (fora da lista)", [], "reuniao")
        ctx.lista = ctx.lista.com_extras([visitante])
        orador_r = Resolucao(orador_ouvido, visitante, [], "media", "anunciado como orador; fora da lista — confirme a grafia")
    orador_nome = orador_r.nome if orador_r else "[?] Orador"
    orador_duv = orador_r is None or (orador_r.pessoa.origem == "reuniao" and not papeis.get("orador"))

    # --- começo do discurso: a última vez que o presidente diz o nome do orador
    i_anc = _ancora_discurso(ctx, orador_ouvido, i_meio, d_min)
    como = "o presidente chama o orador pelo nome"
    if i_anc is None:
        i_anc = _monologo(ctx, i_meio, d_min)
        como = "trecho longo de uma só voz (nome do orador não ouvido)"
        orador_duv = True
    if i_anc is None:
        return       # sem discurso: o resultado sai em dúvida e o worker cai no caminho antigo
    pos = cita(F[i_anc], orador_ouvido, ultima=True)
    if pos is not None:
        _divide_depois_da_frase(ctx, i_anc, pos)
    i_disc = i_anc + 1
    # o nome pode ter sido dito só no anúncio, antes do cântico e da oração de abertura:
    # enquanto só a voz de quem anunciou (ou o coro) fala, o discurso começa depois do "Amém"
    k, ult_amem = i_disc, None
    while k < len(F) and F[k].ini - F[i_anc].fim < d_min and F[k].voz in (F[i_anc].voz, VOZ_CORO):
        if F[k].voz != VOZ_CORO and R.tem("amem", F[k].texto):
            ult_amem = k
        k += 1
    if ult_amem is not None:
        i_disc = ult_amem + 1

    # --- fim do discurso: depois de d_min, a primeira fala que diz o nome do orador
    #     (agradecimento) ou o do dirigente (convite). Teto: d_max.
    dirig_cad = papeis.get("dirigente")
    t0 = F[i_disc].ini if i_disc < len(F) else F[-1].fim
    i_fim = None
    for j in range(i_disc, len(F)):
        if F[j].ini < t0 + d_min:
            continue
        if F[j].ini > t0 + d_max:
            break
        p = cita(F[j], orador_ouvido)
        if p is None:
            p = cita(F[j], dirig_cad)
        if p is None:
            mj = mencoes(ctx, j, j + 1, "dirigente")
            if mj:
                p = cita(F[j], mj[0][1])
        if p is None:
            evs = ctx.ev(F[j], ["fim_discurso", "convite_dirigente"])
            p = evs[0].ini if evs else None
        if p is not None:
            i_fim = j + 1 if _divide_antes_da_frase(ctx, j, p) else j
            break
    fim_incerto = i_fim is None
    if fim_incerto:      # nome não ouvido: corta no primeiro cântico depois de d_min (abre o estudo)
        for j in range(i_disc, len(F)):
            if F[j].ini >= t0 + d_min and (ctx.ev(F[j], ["cantico"]) or F[j].voz == VOZ_CORO):
                i_fim = j
                break
    if i_fim is None or F[i_fim].ini - t0 > d_max:
        i_fim = _indice_no_tempo(ctx, t0 + d_max)
        fim_incerto = True

    # --- janela 1 do presidente: abertura, cântico, oração, anúncio do orador
    fim_j1 = janela_presidente(ctx, 0, i_disc, pres, pres_duv, "abertura")
    delta = fim_j1 - i_disc
    i_disc += delta
    i_fim += delta
    ctx.bloco("abertura", "presidente", pres, 0, i_disc, pres_duv)

    # --- discurso: tudo entre a chamada do orador e o agradecimento do presidente
    vozes = Counter(F[k].voz for k in range(i_disc, i_fim))
    for k in range(i_disc, i_fim):
        if F[k].quem == CORO:
            continue
        F[k].marca(orador_nome, "orador", "discurso", "alta" if not (orador_duv or fim_incerto) else "media",
                   f"discurso: {como}", orador_duv)
    ctx.bloco("discurso", "orador", orador_nome, i_disc, i_fim, orador_duv)
    if orador_r is not None:
        principais = Counter({v: c for v, c in vozes.items() if c >= 0.2 * sum(vozes.values())})
        ctx.aprende_voz(orador_r.nome if orador_r.pessoa is None else orador_r.pessoa.nome, principais)
    if i_fim >= len(F):
        return

    # --- presidente volta: agradece o orador e chama o dirigente (às vezes o leitor)
    lim_conv = min(len(F), i_fim + 6)
    i_conv, dirig_ouvido = None, dirig_cad
    if dirig_ouvido:
        i_conv = next((k for k in range(i_fim, lim_conv) if cita(F[k], dirig_ouvido) is not None), None)
    if i_conv is None:
        mj = mencoes(ctx, i_fim, lim_conv, "dirigente")
        if mj:
            i_conv = mj[0][0]
            dirig_ouvido = dirig_ouvido or mj[0][1]
    if i_conv is None:
        # pela ordem: depois de agradecer o orador, o nome que o presidente diz (que não
        # é o do orador nem dito junto de leitura/oração) é o de quem vai dirigir
        # (quem recebe o "obrigado, irmão X" é o orador, mesmo mal ouvido; o convite vem por último)
        cands = []
        for k, nome in mencoes(ctx, i_fim, lim_conv):
            if cands and k != cands[0][0]:
                break
            if cita(Fala(0, 0, nome), orador_ouvido) is not None:
                continue
            if any(e.nome and fold(e.nome) == fold(nome) for e in ctx.ev(F[k], ["agradecimento"])):
                continue
            p = cita(F[k], nome)
            if p is not None and papel_perto(ctx, F[k].texto, p, p + len(nome)) in ("leitor", "oracao"):
                continue
            cands.append((k, nome))
        if cands:
            i_conv, dirig_ouvido = cands[-1][0], dirig_ouvido or cands[-1][1]
    if i_conv is None:
        k, evs_conv = _primeiro(ctx, "convite_dirigente", i_fim, lim_conv)
        if k is not None:
            i_conv = k
            dirig_ouvido = dirig_ouvido or (evs_conv[0].nome if evs_conv else None)
    if i_conv is None:
        i_conv = i_fim
    for k in range(i_fim, i_conv + 1):
        if F[k].quem == CORO:
            continue
        F[k].marca(pres, "presidente", "volta do presidente", "alta" if not pres_duv else "baixa",
                   "agradece o orador e chama o dirigente", pres_duv)
    ctx.bloco("volta do presidente", "presidente", pres, i_fim, i_conv + 1, pres_duv)

    dirig_r = ctx.resolver(dirig_ouvido)
    dirig = dirig_r.nome if dirig_r else "[?] Dirigente"
    dirig_duv = dirig_r is None or dirig_r.duvida
    # leitor: o cadastro; senão o nome dito junto de leitura/leitor depois do convite
    # (o primeiro: o dirigente anuncia quem lê ao abrir o estudo; depois vêm outros nomes perto de "parágrafo")
    leitor_ouvido = (papeis.get("leitor") or _mais_citado_no_papel(ctx, "leitor", i_fim, len(F), primeiro=True)
                     or _nome_mais_citado(ctx, "leitor", 0, len(F)))
    leitor_r = ctx.resolver(leitor_ouvido)
    # oração final: o último nome dito junto de oração depois do convite
    # (pode vir antes do estudo ou depois do anúncio do cântico final)
    oracao_ouvido = _mais_citado_no_papel(ctx, "oracao", i_fim, len(F), ultimo=True)

    # --- dirigente: cântico de abertura do estudo
    i = i_conv + 1
    lim_ini = i + 3
    while i < min(lim_ini, len(F)):
        if F[i].quem == CORO:
            i += 1
            continue
        F[i].marca(dirig, "dirigente", "estudo de A Sentinela", "alta" if not dirig_duv else "baixa",
                   "abre o estudo", dirig_duv)
        if ctx.ev(F[i], ["cantico"]):
            i = cantar(ctx, i, "estudo de A Sentinela")
            break
        i += 1
    while i < len(F) and F[i].quem == CORO:
        i += 1
    i_estudo = i

    # --- encerramento: último cântico (anunciado, ou o coro separado pelo motor:
    #     a fala logo antes do último coro é a do dirigente anunciando)
    i_fin = None
    for k in range(len(F) - 1, i_estudo, -1):
        if ctx.ev(F[k], ["cantico"]):
            i_fin = k
            break
    k_coro = next((k for k in range(len(F) - 1, i_estudo, -1) if F[k].voz == VOZ_CORO), None)
    if k_coro is not None and (i_fin is None or i_fin < k_coro - 2):
        i_fin = next((k for k in range(k_coro - 1, i_estudo, -1) if F[k].voz != VOZ_CORO), None)
    t_fim_qa = F[i_fin].ini if i_fin is not None else F[-1].fim + 1
    i = perguntas_e_respostas(ctx, i_estudo, t_fim_qa, dirig, dirig_duv, "dirigente",
                              "estudo de A Sentinela", leitor_r)
    if i_fin is not None:      # a lista cresceu com as divisões: reencontra pelo tempo
        i_fin = _indice_no_tempo(ctx, t_fim_qa)
    ctx.bloco("estudo de A Sentinela", "dirigente", dirig, i_conv + 1, i, dirig_duv)
    if leitor_r is not None:
        ctx.blocos.append(Bloco("estudo de A Sentinela", "leitor", leitor_r.nome,
                                F[i_estudo].ini, F[i - 1].fim if i > 0 else 0, leitor_r.duvida))
    if i_fin is None:
        return

    # --- final: anúncio (dirigente), cântico, oração designada, depois do Amém = fora
    F[i_fin].marca(dirig, "dirigente", "encerramento", "alta" if not dirig_duv else "baixa",
                   "encerra o estudo e anuncia o cântico final", dirig_duv)
    # quem ora: o nome dito DEPOIS do cântico na fala que anuncia o cântico final
    # ("cântico 149, e após o cântico, nosso irmão X"); senão o dito junto de "oração"
    ev_c = ctx.ev(F[i_fin], ["cantico"])
    depois = [m for m in _MENCAO.finditer(F[i_fin].texto) if m.start() >= (ev_c[-1].fim if ev_c else 0)]
    nome_final = ctx.R.limpa_nome(depois[-1].group("nome"), F[i_fin].texto, depois[-1].end("nome")) if depois else None
    oracao_ouvido = nome_final or oracao_ouvido
    oracao_r = ctx.resolver(oracao_ouvido, "m") if oracao_ouvido else None
    # o mesmo primeiro nome de quem já fez parte nesta reunião (o orador visitante costuma
    # orar no fim) é essa pessoa, não um homônimo da lista
    for ja in (orador_r, dirig_r, leitor_r):
        if (ja is not None and ja.pessoa is not None and oracao_ouvido
                and _mesmo_nome(oracao_ouvido.split()[0], ja.pessoa.tokens[0])):
            oracao_r = Resolucao(oracao_ouvido, ja.pessoa, [], "media",
                                 f"mesmo nome de quem já participou desta reunião ({ja.pessoa.nome})")
            break
    if oracao_r is None:
        ev_or = ctx.ev(F[i_fin], ["oracao_designada"])
        oracao_r = ctx.resolver(ev_or[0].nome, "m") if ev_or and ev_or[0].nome else None
    j = cantar(ctx, i_fin, "encerramento")
    while j < len(F) and F[j].quem == CORO:
        j += 1
    if j < len(F):
        j = orar(ctx, j, oracao_r, None, "oração final", "encerramento")
    while j < len(F):
        if npal(F[j].texto) <= 2 and ctx.R.tem("amem", F[j].texto):
            F[j].marca(CORO, "amém", "encerramento", "alta", "amém da assistência")
        else:
            F[j].marca(FORA, "fora", "depois do encerramento", "alta", "depois do Amém final")
        j += 1


# =====================================================================
#  REUNIÃO DO MEIO DE SEMANA
# =====================================================================
_TITULO_NOME = re.compile(r"(?i:\b(?:irm[ãa]o|irm[ãa]s?)\s+)" +
                          r"(?P<nome>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõçüñ]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõçüñ]+){0,2})")


def _modo_da_parte(ctx: Contexto, textos: str) -> tuple[str, str]:
    achado = ("parte", "discurso")
    pos_max = -1
    for pid, rx, modo in ctx.R.partes:
        for m in rx.finditer(textos):
            if m.start() > pos_max:
                pos_max, achado = m.start(), (pid, modo)
    return achado


def _designacao_por_nome(ctx: Contexto, i: int) -> list[Evento]:
    """O presidente anuncia a parte dizendo o NOME de quem vai fazer ('irmão X', 'irmãs X e Y')
    e em seguida outra voz fala por um tempo: é uma designação, seja qual for a frase."""
    F = ctx.F
    t = F[i].texto
    # o presidente costuma agradecer a parte anterior e, no fim, anunciar a próxima:
    # nome em frase de agradecimento/parabéns não é designação; vale o último nome
    nomes = []
    for m in _MENCAO.finditer(t):
        if not ctx.R.limpa_nome(m.group("nome"), t, m.end("nome")):
            continue
        a, b = _frase(t, m.start())
        if re.search(r"obrigad|agradec|parabens", fold(t[a:b])):
            continue
        nomes.append(m)
    if not nomes:
        return []
    # quem responde: a próxima fala de verdade (pula coro e "Obrigado." curtinho)
    j = i + 1
    while j < len(F) and (F[j].voz == VOZ_CORO or (npal(F[j].texto) <= 3 and j - i <= 3)):
        j += 1
    if j >= len(F) or F[j].voz == F[i].voz or npal(F[j].texto) < ctx.R.lim("designacao_resposta_min_palavras", 8):
        return []
    # a parte dura: outra voz segura a palavra até o presidente voltar (anúncio solto não é parte)
    k = j
    while k < len(F) and F[k].voz != F[i].voz:
        k += 1
    if (F[k].ini if k < len(F) else F[-1].fim) - F[j].ini < ctx.R.lim("designacao_min_seg", 45):
        return []
    # a frase do anúncio pode ter dois nomes ("irmã Ana Lídia, junto com a irmã Flávia")
    a, _ = _frase(t, nomes[-1].start())
    m = next(n for n in nomes if n.start() >= a)
    return [Evento("designacao", m.start(), m.end(), None, None, "media")]


def meio_de_semana(ctx: Contexto, papeis: dict) -> None:
    F, R = ctx.F, ctx.R
    pres_r = ctx.resolver(papeis.get("presidente")) if papeis.get("presidente") else None
    pres = pres_r.nome if pres_r else "[?] Presidente"
    pres_duv = pres_r is None or pres_r.duvida

    def presidente(k, motivo, parte="programa"):
        F[k].marca(pres, "presidente", parte, "alta" if not pres_duv else "baixa", motivo, pres_duv)

    # último cântico = encerramento
    i_fin = None
    for k in range(len(F) - 1, -1, -1):
        if ctx.ev(F[k], ["cantico"]):
            i_fin = k
            break
    if i_fin is not None and F[i_fin].ini < F[-1].fim * 0.7:
        i_fin = None
    t_fin = F[i_fin].ini if i_fin is not None else None

    i = 0
    texto_pres = ""          # o que o presidente disse desde a última parte (decide o tipo da parte)
    abertura_feita = False
    while i < len(F):
        if t_fin is not None:
            i_fin = _indice_no_tempo(ctx, t_fin)
            if i >= i_fin:
                break
        f = F[i]
        if f.quem:
            i += 1
            continue
        evs_des = ctx.ev(f, ["designacao"]) or _designacao_por_nome(ctx, i)
        # anúncios da congregação e recapitulação: presidente conduz
        if ctx.ev(f, ["recapitulacao"]) and abertura_feita:
            presidente(i, "abre a recapitulação")
            fim_t = F[i_fin].ini if i_fin is not None else F[-1].fim + 1
            # anúncios cortam a recapitulação
            for k in range(i + 1, len(F)):
                if F[k].ini >= fim_t:
                    break
                if len(ctx.ev(F[k], ["anuncios"])) >= 1 and len(re.findall(r"(?i:grupo\s+\d+|an[úu]ncio)", F[k].texto)) >= 1:
                    fim_t = F[k].ini
                    break
            i0 = i
            i = perguntas_e_respostas(ctx, i0, fim_t, pres, pres_duv, "presidente", "recapitulação")
            ctx.bloco("recapitulação", "presidente", pres, i0, i, pres_duv)
            continue
        if ctx.ev(f, ["anuncios"]) and abertura_feita and not evs_des:
            presidente(i, "anúncios da congregação (nomes ditos aqui não são falantes)", "anúncios")
            i += 1
            continue
        if ctx.ev(f, ["cantico"]) and not evs_des:
            presidente(i, "anuncia o cântico")
            i = cantar(ctx, i, "abertura" if not abertura_feita else "programa")
            if i < len(F) and R.tem("oracao_invocacao", F[i].texto) and not abertura_feita:
                i = orar(ctx, i, pres_r, pres, "presidente", "abertura")
            abertura_feita = True
            continue
        if not evs_des:
            presidente(i, "presidente entre as partes")
            texto_pres += " " + f.texto
            i += 1
            continue

        # ---------------- designação de parte
        presidente(i, "anuncia a parte e quem vai fazer")
        abertura_feita = True
        pos = evs_des[-1].ini
        texto_pres += " " + f.texto[:pos]
        pid, modo = _modo_da_parte(ctx, texto_pres + " " + f.texto[pos:])
        nomes_ouv = [m.group("nome") for m in _TITULO_NOME.finditer(f.texto[pos:pos + 220])]
        nomes_ouv = [R.limpa_nome(n, f.texto, 0) for n in nomes_ouv]
        nomes_ouv = [n for n in nomes_ouv if n]
        # "As irmãs Jaqueline, junto com a irmã Kelly"
        mj = re.search(r"(?i:irm[ãa]s?\s+)([A-ZÁÉÍÓÚÂÊÔÃÕÇ]\w+)\s*,?\s*(?i:junto\s+com)", f.texto[pos:])
        if mj and mj.group(1) not in nomes_ouv:
            nomes_ouv.insert(0, mj.group(1))
        texto_pres = ""
        designados = [ctx.resolver(n, genero_do_titulo(f.texto[pos:])) for n in nomes_ouv[:2]]
        # fim da parte: próxima volta do presidente
        j = i + 1
        while j < len(F):
            if i_fin is not None and j >= i_fin:
                break
            g = F[j]
            fecha = bool(ctx.ev(g, ["fecha_parte", "designacao", "recapitulacao"]))
            if not fecha:
                for e in ctx.ev(g, ["agradecimento"]):
                    r = ctx.resolver(e.nome)
                    if any(d is not None and r is not None and d.nome == r.nome for d in designados):
                        fecha = True
            if fecha and (modo in ("participacao", "estudo") and j - i < 3) is False:
                break
            j += 1
        # quem não teve nome no anúncio pode ter no agradecimento de volta
        if j < len(F) and (not designados or all(d is None for d in designados)):
            ag = ctx.ev(F[j], ["agradecimento"])
            if ag:
                designados = [ctx.resolver(ag[0].nome, genero_do_titulo(F[j].texto[max(0, ag[0].ini - 8):ag[0].fim]))]
        if j < len(F) and designados and designados[0] is not None and designados[0].duvida:
            for e in ctx.ev(F[j], ["agradecimento"]):
                r = ctx.resolver(e.nome)
                if r is not None and not r.duvida and (r.pessoa in designados[0].candidatos or
                                                       fold(r.pessoa.tokens[0]) == fold(designados[0].ouvido.split()[0])):
                    designados[0] = Resolucao(designados[0].ouvido, r.pessoa, [], "alta",
                                              f"anunciado '{designados[0].ouvido}', agradecido '{e.nome}'")
                    break
        d1 = designados[0] if designados else None
        d1_nome = d1.nome if d1 else f"[?] {pid}"
        d1_duv = d1 is None or d1.duvida
        if modo in ("participacao", "estudo"):
            leitor = None
            if modo == "estudo":
                lt = _nome_mais_citado(ctx, "leitor", i, j)
                leitor = ctx.resolver(lt) if lt else None
            fim_t = F[j].ini if j < len(F) else F[-1].fim + 1
            k = perguntas_e_respostas(ctx, i + 1, fim_t, d1_nome, d1_duv, "condutor", pid, leitor)
            ctx.bloco(pid, "condutor", d1_nome, i + 1, k, d1_duv)
            j = k      # a janela pode ter crescido com as divisões
        elif modo == "demonstracao" and len(designados) >= 2:
            ordem = []
            for k in range(i + 1, j):
                if F[k].voz not in ordem:
                    ordem.append(F[k].voz)
            for k in range(i + 1, j):
                idx = ordem.index(F[k].voz)
                d = designados[idx] if idx < 2 else None
                if d is not None:
                    marca_pessoa(F[k], d, "designado", pid, "demonstração: pela ordem de entrada da voz", "media")
                else:
                    F[k].marca(f"[?] {designados[0].nome if designados[0] else '?'} ou {designados[1].nome if designados[1] else '?'}",
                               "designado", pid, "baixa", "demonstração: voz não separou as duas", True)
            ctx.bloco(pid, "designado", " e ".join(d.nome for d in designados if d), i + 1, j,
                      any(d is None or d.duvida for d in designados))
        else:
            for k in range(i + 1, j):
                marca_pessoa(F[k], d1, "designado", pid, f"parte '{pid}' anunciada pelo presidente")
            ctx.bloco(pid, "designado", d1_nome, i + 1, j, d1_duv)
        i = j

    if t_fin is not None:
        i_fin = _indice_no_tempo(ctx, t_fin)
        presidente(i_fin, "anuncia o cântico final e quem faz a oração", "encerramento")
        ev_or = ctx.ev(F[i_fin], ["oracao_designada"])
        oracao_r = ctx.resolver(ev_or[0].nome, "m") if ev_or and ev_or[0].nome else None
        j = cantar(ctx, i_fin, "encerramento")
        if j < len(F):
            j = orar(ctx, j, oracao_r, None, "oração final", "encerramento")
        while j < len(F):
            if npal(F[j].texto) <= 2 and R.tem("amem", F[j].texto):
                F[j].marca(CORO, "amém", "encerramento", "alta", "amém da assistência")
            else:
                F[j].marca(FORA, "fora", "depois do encerramento", "alta", "depois do Amém final")
            j += 1


# =====================================================================
#  VOZ CONFIRMA: sugestão de nome para quem ficou sem nome
# =====================================================================
def sugere_pela_voz(ctx: Contexto) -> None:
    """Quem ficou sem nome ("[?]", "[?] Presidente") ganha SUGESTÃO pela voz, se essa voz,
    nesta reunião, só foi usada por uma pessoa já identificada pela estrutura.
    Continua em dúvida (o Eduardo confirma ouvindo); só ganha um palpite com motivo."""
    donos: dict[str, Counter] = {}
    for pessoa, vozes in ctx.vozes_pessoa.items():
        for v, c in vozes.items():
            donos.setdefault(v, Counter())[pessoa] += c

    def palpite(vozes: Counter) -> Optional[str]:
        if not vozes:
            return None
        v = vozes.most_common(1)[0][0]
        d = donos.get(v)
        if d and len(d) == 1:
            p, c = d.most_common(1)[0]
            if c >= 1:
                return p
        return None

    F = ctx.F
    # pessoas CRAVADAS nesta reunião (sem dúvida): base para cravar a oração final
    cravados = {f.quem for f in F if f.quem and not f.duvida
                and not f.quem.startswith("[?]") and f.quem != "Em coro"}
    # papéis inteiros (presidente etc.): todas as falas do papel juntas
    for papel_nome in ("[?] Presidente", "[?] Dirigente", "[?] Orador", "[?] leitor"):
        idx = [k for k, f in enumerate(F) if f.quem == papel_nome]
        p = palpite(Counter(F[k].voz for k in idx))
        if p:
            for k in idx:
                F[k].quem = f"[?] {p}? ({papel_nome.replace('[?] ', '').lower()}, pela voz)"
                F[k].motivo += f" · voz igual à de {p} nesta reunião"
            for b in ctx.blocos:
                if b.quem == papel_nome:
                    b.quem = f"[?] {p}? (pela voz)"
    # trechos soltos sem nome: por sequência
    k = 0
    while k < len(F):
        if F[k].quem != "[?]":
            k += 1
            continue
        j = k
        while j < len(F) and F[j].quem == "[?]" and F[j].voz == F[k].voz:
            j += 1
        p = palpite(Counter(F[x].voz for x in range(k, j)))
        if p:
            # oração final cuja voz é a de alguém JÁ cravado nesta reunião (o
            # orador, em geral): crava — não é palpite solto, é a mesma voz que
            # a estrutura já identificou com certeza. Demais trechos seguem [?].
            crava_final = F[k].papel == "oração final" and p in cravados
            for x in range(k, j):
                if crava_final:
                    F[x].marca(p, F[x].papel, F[x].parte, "media",
                               F[x].motivo + f" · voz igual à de {p} (já identificado nesta reunião)", False)
                else:
                    F[x].quem = f"[?] {p}? (pela voz)"
                    F[x].motivo += f" · voz igual à de {p} nesta reunião"
        k = j


# =====================================================================
#  ENTRADA
# =====================================================================
def _participantes(lista: Lista, nomes: list[str]) -> list[Pessoa]:
    """Participantes digitados -> pessoas: a da lista quando casa sem dúvida
    ("Fabio Vizignani" -> Fábio Vizignani); senão como foi digitado ("Rita", "Otavio")."""
    pessoas, vistos = [], set()
    for n in nomes:
        n = (n or "").strip()
        if not n:
            continue
        r = lista.resolver(n)
        p = r.pessoa if r.pessoa is not None else Pessoa(n, "participante informado na reunião", [], "reuniao")
        if fold(p.nome) not in vistos:
            vistos.add(fold(p.nome))
            pessoas.append(p)
    return pessoas


def analisar(segmentos: list[dict], lista_json: dict, papeis: Optional[dict] = None,
             tipo: Optional[str] = None, regras: Optional[Regras] = None,
             participantes: Optional[list[str]] = None) -> Resultado:
    R = regras or Regras.carregar("religiosa")
    falas = [Fala(float(s["inicio"]), float(s["fim"]), str(s["texto"]).strip(), str(s.get("voz", "")), k)
             for k, s in enumerate(segmentos) if str(s.get("texto", "")).strip()]
    ctx = Contexto(falas, R, Lista.de_json(lista_json))
    if participantes:
        part = _participantes(ctx.lista, participantes)
        ctx.lista = ctx.lista.com_extras([p for p in part if p.origem == "reuniao"])
        ctx.participantes = Lista(part)
    # canto que o motor de transcrição já separou como CORO: é da assistência, nunca de uma pessoa
    for f in falas:
        if f.voz == VOZ_CORO:
            f.marca(CORO, "cântico", "", "alta", "canto da assistência (separado pelo motor)")
    tipo = tipo or R.tipo_de_reuniao([f.texto for f in falas[:80]])
    papeis = {k: v for k, v in (papeis or {}).items() if v}
    # papel informado no cadastro da reunião vale como está: se o nome não está na lista,
    # entra como pessoa desta reunião (o app deve oferecer "adicionar à lista")
    extras = []
    for nome in papeis.values():
        r = ctx.lista.resolver(nome)
        if r.duvida:
            extras.append(Pessoa(nome, "informado no cadastro da reunião", [], "reuniao"))
    if extras:
        ctx.lista = ctx.lista.com_extras(extras)
    if tipo == "fim_de_semana":
        fim_de_semana(ctx, papeis)
    elif tipo == "meio_de_semana":
        meio_de_semana(ctx, papeis)
    for f in ctx.F:
        if not f.quem:
            f.marca("[?]", "", "sem estrutura", "baixa", "nenhuma regra da estrutura explica este trecho: ouça", True)
    sugere_pela_voz(ctx)
    # pendências: trechos em dúvida, agrupados, com tempo
    pend = []
    for f in ctx.F:
        if f.duvida or f.confianca == "media":
            if pend and pend[-1]["quem"] == f.quem and f.ini - pend[-1]["_fim"] < 5:
                pend[-1]["fim"] = hms(f.fim)
                pend[-1]["_fim"] = f.fim
                continue
            pend.append({"inicio": hms(f.ini), "fim": hms(f.fim), "_fim": f.fim, "quem": f.quem,
                         "papel": f.papel, "nivel": "ouvir" if f.duvida else "conferir",
                         "motivo": f.motivo, "trecho": f.texto[:90]})
    for p in pend:
        p.pop("_fim", None)
    return Resultado(tipo, ctx.F, ctx.blocos, pend)
