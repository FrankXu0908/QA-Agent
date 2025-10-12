"""Unit tests for prompt generation logic."""

import pytest

try:
    from services.rag_api.app import build_prompt
except Exception as exc:  # pragma: no cover - missing optional deps
    pytest.skip(f"Skipping prompt tests because dependencies are missing: {exc}", allow_module_level=True)


def test_build_prompt_formats_contexts():
    question = "质量管理的要求是什么？"
    contexts = [
        {"metadata": {"text": "第一段内容", "source": "doc1"}},
        {"metadata": {"text": "第二段内容", "source": "doc2"}},
    ]

    prompt = build_prompt(question, contexts)

    assert "第一段内容" in prompt
    assert "第二段内容" in prompt
    assert "质量管理的要求" in prompt
    
if __name__ == "__main__":
    pytest.main([__file__])
