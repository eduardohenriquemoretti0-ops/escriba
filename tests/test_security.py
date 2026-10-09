"""Testes dos fixes de segurança: token, job_id, rate limit."""
from app.security import RateLimiter, token_matches, valid_job_id


def test_token_constant_time_match():
    assert token_matches("segredo", "segredo") is True
    assert token_matches("errado", "segredo") is False
    assert token_matches("", "segredo") is False
    assert token_matches("qualquer", "") is False  # sem token configurado


def test_job_id_whitelist_bloqueia_traversal():
    assert valid_job_id("abc123_-")
    assert not valid_job_id("../../etc/passwd")
    assert not valid_job_id("a/b")
    assert not valid_job_id("")
    assert not valid_job_id("x" * 65)


def test_rate_limit_por_chave():
    rl = RateLimiter(max_requests=3, window_sec=60)
    key = "1.2.3.4:/jobs"
    assert [rl.allow(key) for _ in range(4)] == [True, True, True, False]
    # chave diferente (outro IP) não é afetada
    assert rl.allow("5.6.7.8:/jobs") is True
