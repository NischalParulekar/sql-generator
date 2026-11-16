"""The parts of the TAPAS system that need no model: fitting rows, combining chunks and
reading answers."""

from dataclasses import dataclass

from tableqa.table import parse_number

OPERATORS = ("NONE", "SUM", "AVERAGE", "COUNT")  # index = TAPAS WTQ aggregation label id


@dataclass(frozen=True)
class ChunkResult:
    """What TAPAS said about one chunk of the table."""

    cells: dict[tuple[int, int], float]  # selected (row, column) of the whole table -> probability
    operator_probs: tuple[float, ...]  # one probability per entry of OPERATORS


def pack_rows(row_costs: list[int], budget: int) -> list[range]:
    """Split rows into consecutive chunks whose total token cost fits the budget.

    Packing is greedy and keeps row order. A single row that costs more than the budget
    still gets a chunk of its own; the tokenizer decides what to do with it.
    """
    chunks, start, used = [], 0, 0
    for i, cost in enumerate(row_costs):
        if i > start and used + cost > budget:
            chunks.append(range(start, i))
            start, used = i, 0
        used += cost
    if row_costs:
        chunks.append(range(start, len(row_costs)))
    return chunks


def combine(results: list[ChunkResult], routed: bool) -> tuple[str, list[tuple[int, int]]]:
    """Merge per-chunk results into one operator and one list of selected cells.

    The operator is the one with the highest mean probability across chunks.

    Union (routed=False): the cells are everything any chunk selected.

    Routed (routed=True): what the operator needs decides where cells come from. SUM,
    AVERAGE and COUNT aggregate over rows, so they keep the union. NONE is a lookup,
    whose answer normally sits in one place, so only the chunk containing the single
    most confident cell is kept and the other chunks' selections are discarded.
    """
    mean_probs = [
        sum(r.operator_probs[i] for r in results) / len(results) for i in range(len(OPERATORS))
    ]
    operator = OPERATORS[mean_probs.index(max(mean_probs))]
    if routed and operator == "NONE":
        best = max(results, key=lambda r: max(r.cells.values(), default=0.0))
        return operator, sorted(best.cells)
    return operator, sorted(cell for r in results for cell in r.cells)


def compute_answer(operator: str, cells: list[str]) -> list[str]:
    """Turn the cells TAPAS selected and its aggregation operator into an answer.

    NONE answers with the cell texts themselves. COUNT counts the selected cells.
    SUM and AVERAGE use the cells that parse as numbers and give no answer if none do.
    """
    if operator == "NONE":
        return [cell for cell in cells if cell.strip()]
    if operator == "COUNT":
        return [str(len(cells))]
    numbers = [n for n in (parse_number(cell.strip()) for cell in cells) if n is not None]
    if not numbers:
        return []
    value = sum(numbers) if operator == "SUM" else sum(numbers) / len(numbers)
    return [str(int(value)) if float(value).is_integer() else str(value)]
