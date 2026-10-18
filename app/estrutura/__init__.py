"""Motor de estrutura de reunião (Escriba). Ver ESPECIFICACAO.md."""
from .motor import analisar, Resultado, Fala, Bloco, hms  # noqa: F401
from .eventos import Regras  # noqa: F401
from .nomes import Lista, Pessoa, fonetica  # noqa: F401

__all__ = ["analisar", "Resultado", "Fala", "Bloco", "hms", "Regras", "Lista", "Pessoa", "fonetica"]
