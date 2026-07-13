from hyskill.generator import HypotheticalGenerator, SKILL_TEMPLATE, PASSAGE_TEMPLATE


class FakeClient:
    def __init__(self, reply="---\nname: x\ndescription: y\n---\nbody"):
        self.calls = 0
        self.reply = reply

    def complete(self, prompt: str, temperature: float) -> str:
        self.calls += 1
        return self.reply


class FailingClient:
    def complete(self, prompt, temperature):
        raise RuntimeError("boom")


def test_generates_k_samples(tmp_path):
    fake = FakeClient()
    g = HypotheticalGenerator(client=fake, k_samples=3, cache_dir=tmp_path)
    docs = g.generate("task q")
    assert len(docs) == 3 and fake.calls == 3


def test_cache_hit_skips_client(tmp_path):
    fake = FakeClient()
    g = HypotheticalGenerator(client=fake, k_samples=2, cache_dir=tmp_path)
    g.generate("task q")
    g2 = HypotheticalGenerator(client=fake, k_samples=2, cache_dir=tmp_path)
    g2.generate("task q")
    assert fake.calls == 2  # second run fully cached


def test_failure_returns_empty(tmp_path):
    g = HypotheticalGenerator(client=FailingClient(), k_samples=2, cache_dir=tmp_path)
    assert g.generate("task q") == []


def test_templates_mention_task():
    assert "{q}" in SKILL_TEMPLATE and "{q}" in PASSAGE_TEMPLATE
