from hyskill.parser import parse_fields, parse_generated

MD = (
    "# Lah Numbers\n\n"
    "Some prose about Lah numbers.\n\n"
    "```python\ndef lah(n, k):\n    return 1\n```\n\n"
    "More prose."
)


def test_parse_fields_splits_code_and_body():
    f = parse_fields(name="Lah Numbers", description="Counting partitions.", content=MD)
    assert f["meta"] == "Lah Numbers. Counting partitions."
    assert "def lah" in f["code"]
    assert "def lah" not in f["body"]
    assert "Some prose" in f["body"]


def test_parse_fields_missing_code():
    f = parse_fields(name="X", description="Y", content="no code here")
    assert f["code"] == ""


def test_parse_generated_frontmatter():
    md = "---\nname: pdf-to-md\ndescription: Convert PDFs\n---\n1. step one\n\n```py\nx=1\n```\n"
    f = parse_generated(md)
    assert f["meta"] == "pdf-to-md. Convert PDFs"
    assert "step one" in f["body"]
    assert "x=1" in f["code"]


def test_parse_generated_no_frontmatter_falls_back():
    f = parse_generated("Just a passage.\nSecond line.")
    assert f["meta"] == "Just a passage."
    assert "Second line" in f["body"]


def test_parse_generated_unwraps_outer_markdown_fence():
    md = ("```markdown\n"
          "---\nname: ordered-subsets\ndescription: Count ordered partitions\n---\n"
          "1. use Stirling numbers\n\n"
          "```skill\ndef f(n): pass\n```\n"
          "```")
    f = parse_generated(md)
    assert f["meta"] == "ordered-subsets. Count ordered partitions"
    assert "Stirling" in f["body"]
    assert "def f(n)" in f["code"]
