from rlalpha.search.models import SearchContext
from rlalpha.search.prompts import PROMPT_VERSION, build_messages, prompt_contract


def test_prompt_is_train_only_and_identical_across_reward_objectives():
    context = SearchContext(3, ("Mean($return,20)",), (0.2,), 0.01, 40, 5000)
    messages = build_messages(context)
    rendered = str(messages).lower()
    assert "validation" not in rendered
    assert "test" not in rendered
    assert "<expr>" in rendered
    assert "mean($return,20)" in rendered


def test_one_compact_prompt_contains_all_elements_and_hints_without_pool_weights():
    context = SearchContext(3, tuple(f"Mean($return,{window})" for window in (1, 5, 10, 20, 40, 60, 120, 252)), tuple(range(8)), 0.01, 40, 5000)
    rendered = str(build_messages(context)).lower()
    assert "momentum" in rendered
    assert "mean reversion" in rendered
    assert "volatility" in rendered
    assert "price-volume" in rendered
    assert "multi-horizon" in rendered
    assert "$open" in rendered and "$return" in rendered
    assert "csrank" in rendered and "corr" in rendered
    assert "factor_6" not in rendered
    assert "ref/delta add w" in rendered
    assert "w=" not in rendered
    assert "objective=" not in rendered
    contract = prompt_contract()
    assert contract["version"] == PROMPT_VERSION
    assert len(contract["hash"]) == 64


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
