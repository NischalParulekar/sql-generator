"""The parts of the TAPAS system that need no model: fitting rows and reading answers."""

from tableqa.table import parse_number

OPERATORS = ("NONE", "SUM", "AVERAGE", "COUNT")  # index = TAPAS WTQ aggregation label id


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
