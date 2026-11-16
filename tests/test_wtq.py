import zipfile

import pytest

from tableqa.data import wtq
from tableqa.data.wtq import WTQ, Example, sample


@pytest.fixture
def dataset(tmp_path):
    """A miniature archive laid out like the real release."""
    path = tmp_path / "wtq.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "WikiTableQuestions/data/random-split-1-dev.tsv",
            "id\tutterance\tcontext\ttargetValue\n"
            "nt-1\twho scored most?\tcsv/200-csv/1.csv\tAnn\n"
            "nt-2\twhich teams drew?\tcsv/200-csv/1.csv\tReds|Blues\n"
            "nt-3\twhat was the score?\tcsv/200-csv/1.csv\t1\\p0 (a\\\\b)\n",
        )
        z.writestr(
            "WikiTableQuestions/csv/200-csv/1.tsv",
            "Player\tTotal\\nGoals\tNote\n"
            "Ann\t1,200\tfirst\\psecond\n"
            "Bob\t7\tline one\\nline two same cell\n",
        )
    return WTQ(path)


def test_examples_are_parsed_with_list_answers_and_escapes(dataset):
    assert dataset.examples("dev") == [
        Example("nt-1", "who scored most?", "csv/200-csv/1.csv", ("Ann",)),
        Example("nt-2", "which teams drew?", "csv/200-csv/1.csv", ("Reds", "Blues")),
        Example("nt-3", "what was the score?", "csv/200-csv/1.csv", ("1|0 (a\\b)",)),
    ]


def test_table_is_loaded_from_the_tsv_twin_of_the_csv_path(dataset):
    table = dataset.table("csv/200-csv/1.csv")
    assert table.id == "csv/200-csv/1.csv"
    assert table.headers == ("Player", "Total\nGoals", "Note")
    assert table.columns == ("player", "total_goals", "note")
    assert table.types == ("VARCHAR", "BIGINT", "VARCHAR")
    assert table.rows == (
        ("Ann", 1200, "first|second"),
        ("Bob", 7, "line one\nline two same cell"),
    )


def test_sample_is_seeded_and_ignores_input_order():
    examples = [Example(f"nt-{i}", "q", "t", ("a",)) for i in range(100)]
    first = sample(examples, 10, seed=7)
    assert len({e.id for e in first}) == 10
    assert sample(list(reversed(examples)), 10, seed=7) == first
    assert sample(examples, 10, seed=8) != first


def test_download_rejects_a_file_with_the_wrong_checksum(tmp_path):
    path = tmp_path / "wtq.zip"
    path.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        wtq.download(path)
