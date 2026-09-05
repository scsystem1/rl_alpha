from rlalpha.search.models import SearchContext, TrainPoolSummary
from rlalpha.search.prompts import PROMPT_VERSION, build_messages, prompt_contract


def test_prompt_is_train_only_and_identical_across_reward_objectives():
    context = SearchContext(3, ("Mean($return,20)",), (0.2,), 0.01, 40, 5000)
    messages = build_messages(context)
    rendered = str(messages).lower()
    assert "validation" not in rendered
    assert "test" not in rendered
    assert "<expr>" in rendered
    assert "mean($return,20)" in rendered


def test_compact_prompt_has_task_evidence_without_theme_hints():
    context = SearchContext(3, tuple(f"Mean($return,{window})" for window in (1, 5, 10, 20, 40, 60, 120, 252)), tuple(range(8)), 0.01, 40, 5000)
    rendered = str(build_messages(context)).lower()
    assert all(theme not in rendered for theme in ("momentum", "mean reversion", "volatility", "price-volume", "multi-horizon"))
    assert "20-trading-day" in rendered and "t+21" in rendered
    assert "balanced-22" in rendered
    assert "$open" in rendered and "$return" in rendered
    assert "csrank" in rendered and "corr" in rendered
    assert "factor_6" not in rendered
    assert "ref/delta add w" in rendered
    assert "w=" not in rendered
    assert "objective=" not in rendered
    contract = prompt_contract()
    assert contract["version"] == PROMPT_VERSION
    assert len(contract["hash"]) == 64


def test_prompt_uses_only_canonical_summary_not_reward_specific_objective():
    from dataclasses import replace
    summary = TrainPoolSummary(.012345, (-.125,))
    context = SearchContext(2, ("Mean($return,20)",), (99.,), -123., 40, 5000, prompt_summary=summary)
    rendered = str(build_messages(context))
    assert "w=-0.125 Mean($return,20)" in rendered
    assert "RNIC=+0.0123" in rendered
    assert "-123" not in rendered and "99.0" not in rendered
    assert build_messages(context) == build_messages(replace(context, train_objective=456., pool_weights=(3.,)))


def test_summary_must_align_with_all_formulas_and_be_finite():
    import pytest
    context = SearchContext(0, ("$return",), (), 0., 0, 8, prompt_summary=TrainPoolSummary(.01, ()))
    with pytest.raises(ValueError, match="formula-aligned"):
        build_messages(context)


def test_prompt_api_has_no_hint_or_version_switches():
    import inspect

    assert tuple(inspect.signature(build_messages).parameters) == ("context",)
    assert tuple(inspect.signature(prompt_contract).parameters) == ()


def test_base_llm_samples_the_identical_prompt_eight_times(monkeypatch):
    import sys
    from types import SimpleNamespace

    from rlalpha.search.base_llm import BaseLLMSearcher

    captured = {}

    class Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            return repr(messages)

    class LLM:
        @staticmethod
        def generate(prompts, params, use_tqdm=False):
            captured.update({"prompts": prompts, "params": params, "use_tqdm": use_tqdm})
            return [SimpleNamespace(outputs=[SimpleNamespace(text="<expr>Mean($return,20)</expr>", token_ids=[1, 2])]) for _ in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs))
    searcher = BaseLLMSearcher(0, {"temperature": 1.0, "response_length": 128})
    searcher._processor = Processor()
    searcher._llm = LLM()
    context = SearchContext(0, (), (), 0.0, 0, 8)
    candidates = searcher.propose(context, 8)
    assert len(candidates) == 8
    assert len(captured["prompts"]) == 8
    assert len(set(captured["prompts"])) == 1
