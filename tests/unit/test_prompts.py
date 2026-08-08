from rlalpha.search.models import SearchContext
from rlalpha.search.prompts import build_messages


def test_prompt_is_train_only_and_identical_across_reward_objectives():
    context = SearchContext(3, ("Mean($return,20)",), (0.2,), 0.01, 40, 5000)
    messages = build_messages(context, "momentum")
    rendered = str(messages).lower()
    assert "validation" not in rendered
    assert "test" not in rendered
    assert "<expr>" in rendered
