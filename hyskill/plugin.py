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

from hyskill.generator import (HypotheticalGenerator, OpenAIClient,
                               PASSAGE_TEMPLATE, SKILL_TEMPLATE)
from hyskill.naive_hyde import NaiveHydeRetriever
from hyskill.retriever import HySkillRetriever


class _MockClient:
    def complete(self, prompt: str, temperature: float) -> str:
        task = prompt.rsplit("Task:", 1)[-1].strip()[:120]
        return ("---\nname: hypothetical-skill\ndescription: " + task +
                "\n---\n1. analyse the task\n2. apply the method\n\n"
                "```python\nsolve()\n```")


def _generator(template, model="", api_base="", k_samples="4",
               temperature="0.7", cache_dir="", mock_generator="0"):
    client = (_MockClient() if str(mock_generator) == "1"
              else OpenAIClient(model=model, api_base=api_base))
    return HypotheticalGenerator(
        client=client, k_samples=int(k_samples), temperature=float(temperature),
        template=template, cache_dir=cache_dir or None,
        model_tag=f"{model}|{template[:20]}")


@register("hyskill")
def hyskill_factory(corpus_path, encoder_name="BAAI/bge-base-en-v1.5",
                    rrf_k="60", **gen_kwargs):
    return HySkillRetriever(
        corpus_path=corpus_path,
        generator=_generator(SKILL_TEMPLATE, **gen_kwargs),
        encoder_name=encoder_name, rrf_k=int(rrf_k))


@register("naive_hyde")
def naive_hyde_factory(encoder_name="BAAI/bge-base-en-v1.5", **gen_kwargs):
    return NaiveHydeRetriever(
        generator=_generator(PASSAGE_TEMPLATE, **gen_kwargs),
        encoder_name=encoder_name)
