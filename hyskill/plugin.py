"""SR-Agents plugin: registers `hyskill` and `naive_hyde` retrievers.

Usage:
    sragents --plugin hyskill.plugin retrieve --retriever hyskill \
        --retriever-arg corpus_path=data/bench/corpus/corpus.json \
        --retriever-arg model=Qwen/Qwen3-4B-Instruct \
        --retriever-arg api_base=http://localhost:8000/v1 \
        --retriever-arg k_samples=4 \
        --retriever-arg cache_dir=results/hyp_cache ...

All --retriever-arg values arrive as strings; factories coerce types.
`mock_generator=1` swaps in an offline echo generator (integration tests).
"""

from sragents.retrieve.base import register

# --- upstream bugfix (versioned here so fresh clones reproduce it) ---------
# SR-Agents' llm_rerank hardcodes max_tokens=4096; with an 8192-ctx server,
# any rerank prompt >4096 input tokens gets a hard 400 (long queries in
# logicbench/medcalc/bigcode). The ranking output is ~150 tokens, so cap it.
# Applied when this plugin module is imported (sragents --plugin hyskill.plugin).
import sragents.retrieve.llm_rerank as _lr

_orig_chat = _lr.chat


def _capped_chat(*args, **kwargs):
    if kwargs.get("max_tokens", 0) > 1024:
        kwargs["max_tokens"] = 1024
    return _orig_chat(*args, **kwargs)


_lr.chat = _capped_chat
# ---------------------------------------------------------------------------

from hyskill.generator import (HypotheticalGenerator, OpenAIClient,
                               PASSAGE_TEMPLATE, SENTENCE_TEMPLATE,
                               SKILL_TEMPLATE)
from hyskill.naive_hyde import NaiveHydeRetriever
from hyskill.retriever import HySkillRetriever


class _MockClient:
    def complete(self, prompt: str, temperature: float) -> str:
        task = prompt.rsplit("Task:", 1)[-1].strip()[:120]
        return ("---\nname: hypothetical-skill\ndescription: " + task +
                "\n---\n1. analyse the task\n2. apply the method\n\n"
                "```python\nsolve()\n```")


def _generator(template, model="", api_base="", k_samples="4",
               temperature="0.7", cache_dir="", mock_generator="0",
               no_think="0"):
    client = (_MockClient() if str(mock_generator) == "1"
              else OpenAIClient(model=model, api_base=api_base,
                                no_think=str(no_think) == "1"))
    return HypotheticalGenerator(
        client=client, k_samples=int(k_samples), temperature=float(temperature),
        template=template, cache_dir=cache_dir or None,
        model_tag=f"{model}|{template[:20]}")


@register("hyskill")
def hyskill_factory(corpus_path, encoder_name="BAAI/bge-base-en-v1.5",
                    rrf_k="60", emb_cache_dir="", **gen_kwargs):
    return HySkillRetriever(
        corpus_path=corpus_path,
        generator=_generator(SKILL_TEMPLATE, **gen_kwargs),
        encoder_name=encoder_name, rrf_k=int(rrf_k),
        emb_cache_dir=emb_cache_dir or None)


@register("naive_hyde")
def naive_hyde_factory(encoder_name="BAAI/bge-base-en-v1.5",
                       emb_cache_dir="", template="passage", **gen_kwargs):
    """template=passage: imagine an answer-like passage (HyDE-faithful).
    template=skill: imagine a full SKILL.md, matched as ONE vector.
    template=sentence: imagine a one-sentence skill (name + what it does) —
    the imagination-granularity ablation, ~10x cheaper than full skill.
    Skill-template generations share the hyskill cache (same model_tag prefix)."""
    tpl = {"skill": SKILL_TEMPLATE, "sentence": SENTENCE_TEMPLATE}.get(
        template, PASSAGE_TEMPLATE)
    return NaiveHydeRetriever(
        generator=_generator(tpl, **gen_kwargs),
        encoder_name=encoder_name, emb_cache_dir=emb_cache_dir or None)


@register("dense")
def dense_factory(model_path="sentence-transformers/all-MiniLM-L6-v2",
                  batch_size="256"):
    """Plain dense retriever with NO query prefix — encoder-matched query-only
    baseline for hyskill/naive_hyde attribution."""
    from sragents.retrieve.dense import DenseRetriever
    return DenseRetriever(model_name_or_path=model_path,
                         batch_size=int(batch_size))


@register("two_stage")
def two_stage_factory(corpus_path, encoder_name="BAAI/bge-base-en-v1.5",
                      rrf_k="60", recall_k="50", emb_cache_dir="", **gen_kwargs):
    """Fusion recall (stage 1) + whole-hypothetical single-vector rerank (stage 2)."""
    from hyskill.two_stage import TwoStageRetriever
    return TwoStageRetriever(
        corpus_path=corpus_path,
        generator=_generator(SKILL_TEMPLATE, **gen_kwargs),
        encoder_name=encoder_name, rrf_k=int(rrf_k),
        emb_cache_dir=emb_cache_dir or None, recall_k=int(recall_k))
