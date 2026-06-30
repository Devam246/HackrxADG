import pytest
from services.ingestion.chunker import ParentChildChunker


@pytest.fixture
def sample_policy_text():
    return (
        "Waiting Period: This policy has a waiting period of 30 days for any claim. "
        "No benefits will be paid for treatments during this time.\n\n"
        "Exclusions: This policy does not cover cosmetic surgery, dental treatments, "
        "or routine checkups. In addition, experimental treatments are not eligible "
        "for reimbursement under any circumstances."
    )


def test_parent_child_creates_hierarchy(sample_policy_text):
    chunker = ParentChildChunker(parent_size=50, child_size=15, overlap=2)
    chunks = chunker.chunk(sample_policy_text, "doc123", "insurance")

    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]

    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c.parent_id is not None for c in children)


def test_no_empty_chunks(sample_policy_text):
    chunker = ParentChildChunker(parent_size=50, child_size=15, overlap=2)
    chunks = chunker.chunk(sample_policy_text, "doc456", "insurance")

    assert all(len(c.text.strip()) > 0 for c in chunks)
