"""Synthetic order tables for the large-table stress test.

The data is invented and generated from a seed; nothing here comes from a real business.
A table of n rows is the first n rows of one seeded stream, so the 1,000-row table is a
prefix of the 10,000-row one.

Questions are filled-in templates of eight types (lookup, filter, count, sum, average,
top_n, group_by, multi_step). Each gold answer is computed in plain Python from the rows,
independently of DuckDB and of every system under test. A question is discarded when its
answer would be ambiguous: a tie for a "highest", or an average that sits on a rounding
boundary. All numbers in the table are whole numbers, so sums are exact.

Because the tables are generated fresh, no model can have seen them in training.
"""

import datetime
import random
from collections import defaultdict
from collections.abc import Callable
from fractions import Fraction
from typing import NamedTuple

from tableqa.data.wtq import Example
from tableqa.table import Table

HEADERS = [
    "Order ID",
    "Order Date",
    "Customer",
    "Region",
    "Product",
    "Category",
    "Quantity",
    "Unit Price",
    "Amount",
    "Status",
    "Sales Rep",
]


class Order(NamedTuple):
    order_id: str
    date: str
    customer: str
    region: str
    product: str
    category: str
    quantity: int
    unit_price: int
    amount: int
    status: str
    rep: str


REGIONS = ["North", "South", "East", "West", "Central"]
STATUSES = ["Delivered", "Shipped", "Pending", "Cancelled", "Returned"]
STATUS_WEIGHTS = [70, 10, 8, 7, 5]
PRODUCTS = [  # (name, category, list price)
    ("Laptop", "Electronics", 900),
    ("Monitor", "Electronics", 240),
    ("Keyboard", "Electronics", 45),
    ("Webcam", "Electronics", 60),
    ("Office Chair", "Furniture", 180),
    ("Standing Desk", "Furniture", 420),
    ("Bookshelf", "Furniture", 130),
    ("Filing Cabinet", "Furniture", 150),
    ("Notebook", "Stationery", 4),
    ("Pen Set", "Stationery", 12),
    ("Stapler", "Stationery", 9),
    ("Whiteboard", "Stationery", 65),
    ("Coffee Maker", "Appliances", 85),
    ("Microwave", "Appliances", 110),
    ("Air Purifier", "Appliances", 210),
    ("Desk Fan", "Appliances", 35),
    ("Backpack", "Accessories", 40),
    ("USB Hub", "Accessories", 25),
    ("Mouse Pad", "Accessories", 8),
    ("Headset", "Accessories", 70),
]
CATEGORIES = sorted({category for _, category, _ in PRODUCTS})
CUSTOMERS = [
    f"{place} {kind}"
    for place in [
        "Blue Harbor",
        "Cedar Peak",
        "Golden Field",
        "Silver Lake",
        "Red Canyon",
        "Maple Grove",
        "Iron Bridge",
        "Bright Star",
        "Stone Mill",
        "Green Valley",
    ]
    for kind in ["Trading", "Supplies", "Logistics", "Retail"]
]
SALES_REPS = [
    "Asha Rao",
    "Ben Carter",
    "Chen Wei",
    "Diana Lopez",
    "Emeka Obi",
    "Fatima Khan",
    "Georg Meyer",
    "Hana Sato",
    "Ivan Petrov",
    "Julia Costa",
    "Kofi Mensah",
    "Lena Novak",
]
FIRST_DAY = datetime.date(2023, 1, 1)
DAYS = 731  # 2023 and 2024
DISCOUNTS = [1, 1, 0.95, 0.9]

Made = tuple[str, list[str]] | None  # a question and its answer, or None if it is unusable
Template = Callable[[random.Random, list[Order]], Made]


def generate_orders(n_rows: int, seed: int) -> list[Order]:
    rng = random.Random(seed)

    def skew(items):  # uneven popularity, so that "highest" questions have clear answers
        return [rng.uniform(1, 4) for _ in items]

    customer_w, region_w, product_w, rep_w = map(skew, (CUSTOMERS, REGIONS, PRODUCTS, SALES_REPS))
    orders = []
    for i in range(n_rows):
        product, category, list_price = rng.choices(PRODUCTS, product_w)[0]
        quantity = rng.randint(1, 20)
        unit_price = round(list_price * rng.choice(DISCOUNTS))
        orders.append(
            Order(
                order_id=f"ORD-{i + 1:06d}",
                date=(FIRST_DAY + datetime.timedelta(days=rng.randrange(DAYS))).isoformat(),
                customer=rng.choices(CUSTOMERS, customer_w)[0],
                region=rng.choices(REGIONS, region_w)[0],
                product=product,
                category=category,
                quantity=quantity,
                unit_price=unit_price,
                amount=quantity * unit_price,
                status=rng.choices(STATUSES, STATUS_WEIGHTS)[0],
                rep=rng.choices(SALES_REPS, rep_w)[0],
            )
        )
    return orders


def generate(n_rows: int, seed: int = 42, per_type: int = 5) -> tuple[Table, list[Example]]:
    """A table of n_rows orders and per_type questions of each type about it."""
    orders = generate_orders(n_rows, seed)
    table_id = f"synthetic-orders-{n_rows}-seed{seed}"
    table = Table.from_strings(table_id, HEADERS, [[str(v) for v in order] for order in orders])
    rng = random.Random(f"{seed}-{n_rows}")
    examples = []
    for qtype, templates in TEMPLATES.items():
        made: dict[str, list[str]] = {}
        for attempt in range(500):
            if len(made) == per_type:
                break
            result = templates[attempt % len(templates)](rng, orders)
            if result and result[0] not in made:
                made[result[0]] = result[1]
        if len(made) < per_type:
            raise ValueError(f"only {len(made)} usable {qtype} questions for {n_rows} rows")
        examples += [
            Example(f"syn-{n_rows}-{qtype}-{i}", question, table_id, tuple(answer), qtype)
            for i, (question, answer) in enumerate(made.items(), start=1)
        ]
    return table, examples


# ---- helpers ---------------------------------------------------------------------------


def _where(orders: list[Order], **equals) -> list[Order]:
    return [o for o in orders if all(getattr(o, k) == v for k, v in equals.items())]


def _totals(orders: list[Order], key: str, value: str | None = None) -> dict[str, int]:
    """Sum of `value` per `key`, or the number of orders per `key` when value is None."""
    totals: dict[str, int] = defaultdict(int)
    for o in orders:
        totals[getattr(o, key)] += getattr(o, value) if value else 1
    return totals


def _top(scores: dict, n: int = 1, lowest: bool = False) -> list[str] | None:
    """The n best keys, or None if a tie makes the n-th place ambiguous."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=not lowest)
    if len(ranked) < n or (len(ranked) > n and ranked[n - 1][1] == ranked[n][1]):
        return None
    return [key for key, _ in ranked[:n]]


def _round2(value: Fraction) -> str | None:
    """value to two decimal places, or None if it is too close to a rounding boundary
    for exact and floating-point arithmetic to be sure to agree."""
    hundredths = value * 100
    if abs(hundredths % 1 - Fraction(1, 2)) < Fraction(1, 10**6):
        return None
    return f"{round(hundredths) / 100:.2f}"


def _mean2(values: list[int]) -> str | None:
    return _round2(Fraction(sum(values), len(values))) if values else None


def _few(values: list[str]) -> list[str] | None:
    """Distinct values in order, if there are between one and five of them."""
    distinct = list(dict.fromkeys(values))
    return distinct if 1 <= len(distinct) <= 5 else None


def _q(question: str, answer) -> Made:
    """Package a question, dropping it if its answer could not be decided."""
    if answer is None:
        return None
    return question, [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]


# ---- templates -------------------------------------------------------------------------
# Each takes the random source and the orders and returns (question, answer) or None.


def _lookup(field: str, question: str) -> Template:
    def template(rng, orders):
        order = rng.choice(orders)
        return _q(question.format(id=order.order_id), getattr(order, field))

    return template


def _filter_orders(rng, orders):
    o = rng.choice(orders)
    found = _where(orders, customer=o.customer, product=o.product, date=o.date)
    return _q(
        f"Which orders did {o.customer} place for the product {o.product} on {o.date}?",
        _few([x.order_id for x in found]),
    )


def _filter_products(rng, orders):
    o = rng.choice(orders)
    found = _where(orders, customer=o.customer, date=o.date)
    return _q(
        f"Which products did {o.customer} order on {o.date}?", _few([x.product for x in found])
    )


def _filter_customers(rng, orders):
    o = rng.choice(orders)
    found = _where(orders, product=o.product, date=o.date, region=o.region)
    return _q(
        f"Which customers in the {o.region} region ordered the product {o.product} on {o.date}?",
        _few([x.customer for x in found]),
    )


def _filter_reps(rng, orders):
    o = rng.choice(orders)
    found = _where(orders, customer=o.customer, date=o.date)
    return _q(
        f"Which sales reps handled orders from {o.customer} on {o.date}?",
        _few([x.rep for x in found]),
    )


def _filter_status(rng, orders):
    o = rng.choice(orders)
    found = _where(orders, customer=o.customer, date=o.date, status=o.status)
    return _q(
        f"Which orders with the status {o.status} did {o.customer} place on {o.date}?",
        _few([x.order_id for x in found]),
    )


def _count_region(rng, orders):
    region = rng.choice(REGIONS)
    return _q(
        f"How many orders were placed in the {region} region?", len(_where(orders, region=region))
    )


def _count_status(rng, orders):
    status = rng.choice(STATUSES)
    return _q(f"How many orders have the status {status}?", len(_where(orders, status=status)))


def _count_customer(rng, orders):
    customer = rng.choice(CUSTOMERS)
    return _q(f"How many orders did {customer} place?", len(_where(orders, customer=customer)))


def _count_product_status(rng, orders):
    product, status = rng.choice(PRODUCTS)[0], rng.choice(STATUSES)
    return _q(
        f"How many orders for the product {product} have the status {status}?",
        len(_where(orders, product=product, status=status)),
    )


def _count_quantity(rng, orders):
    threshold = rng.randint(5, 18)
    return _q(
        f"How many orders have a quantity greater than {threshold}?",
        sum(o.quantity > threshold for o in orders),
    )


def _sum_region(rng, orders):
    region = rng.choice(REGIONS)
    return _q(
        f"What is the total amount of all orders in the {region} region?",
        sum(o.amount for o in _where(orders, region=region)),
    )


def _sum_product(rng, orders):
    product = rng.choice(PRODUCTS)[0]
    return _q(
        f"What is the total quantity ordered of the product {product}?",
        sum(o.quantity for o in _where(orders, product=product)),
    )


def _sum_customer(rng, orders):
    customer = rng.choice(CUSTOMERS)
    return _q(
        f"What is the total amount of all orders placed by {customer}?",
        sum(o.amount for o in _where(orders, customer=customer)),
    )


def _sum_status_category(rng, orders):
    status, category = rng.choice(STATUSES), rng.choice(CATEGORIES)
    return _q(
        f"What is the total amount of orders with the status {status} in the {category} category?",
        sum(o.amount for o in _where(orders, status=status, category=category)),
    )


def _sum_year(rng, orders):
    year = rng.choice(["2023", "2024"])
    return _q(
        f"What is the total quantity ordered in orders placed in {year}?",
        sum(o.quantity for o in orders if o.date.startswith(year)),
    )


ROUNDED = "rounded to two decimal places"


def _avg_product(rng, orders):
    product = rng.choice(PRODUCTS)[0]
    return _q(
        f"What is the average quantity per order for the product {product}, {ROUNDED}?",
        _mean2([o.quantity for o in _where(orders, product=product)]),
    )


def _avg_region(rng, orders):
    region = rng.choice(REGIONS)
    return _q(
        f"What is the average amount of orders in the {region} region, {ROUNDED}?",
        _mean2([o.amount for o in _where(orders, region=region)]),
    )


def _avg_category(rng, orders):
    category = rng.choice(CATEGORIES)
    return _q(
        f"What is the average unit price of orders in the {category} category, {ROUNDED}?",
        _mean2([o.unit_price for o in _where(orders, category=category)]),
    )


def _avg_rep(rng, orders):
    rep = rng.choice(SALES_REPS)
    return _q(
        f"What is the average amount of orders handled by the sales rep {rep}, {ROUNDED}?",
        _mean2([o.amount for o in _where(orders, rep=rep)]),
    )


def _avg_status(rng, orders):
    status = rng.choice(STATUSES)
    return _q(
        f"What is the average quantity of orders with the status {status}, {ROUNDED}?",
        _mean2([o.quantity for o in _where(orders, status=status)]),
    )


def _ranking(question: str, key: str, value: str | None, n: int, lowest: bool = False) -> Template:
    def template(rng, orders):
        return _q(question, _top(_totals(orders, key, value), n, lowest))

    return template


def _top_customers_in_region(rng, orders):
    region = rng.choice(REGIONS)
    return _q(
        f"Which 3 customers placed the most orders in the {region} region?",
        _top(_totals(_where(orders, region=region), "customer"), 3),
    )


def _top_products_by_status(rng, orders):
    status = rng.choice(STATUSES)
    return _q(
        f"Which 3 products have the highest total amount among orders with the status {status}?",
        _top(_totals(_where(orders, status=status), "product", "amount"), 3),
    )


def _lowest_average_quantity(rng, orders):
    quantity, count = _totals(orders, "product", "quantity"), _totals(orders, "product")
    return _q(
        "Which product has the lowest average quantity per order?",
        _top({p: Fraction(quantity[p], count[p]) for p in count}, lowest=True),
    )


def _most_orders_with_status(rng, orders):
    status = rng.choice(STATUSES)
    return _q(
        f"Which customer placed the most orders with the status {status}?",
        _top(_totals(_where(orders, status=status), "customer")),
    )


def _distinct_dates(rng, orders):
    customer = rng.choice(CUSTOMERS)
    dates = {o.date for o in _where(orders, customer=customer)}
    return _q(f"On how many different dates did {customer} place an order?", len(dates) or None)


def _above_average(rng, orders):
    mean = Fraction(sum(o.amount for o in orders), len(orders))
    return _q(
        "How many orders have an amount greater than the average order amount?",
        sum(o.amount > mean for o in orders),
    )


def _total_of_busiest_customer(rng, orders):
    busiest = _top(_totals(orders, "customer"))
    return _q(
        "What is the total amount of all orders placed by the customer with the most orders?",
        busiest and sum(o.amount for o in _where(orders, customer=busiest[0])),
    )


def _region_difference(rng, orders):
    totals = _totals(orders, "region", "amount")
    first, second = rng.sample(REGIONS, 2)
    if totals[first] <= totals[second]:
        first, second = second, first
    difference = totals[first] - totals[second]
    return _q(
        f"How much higher is the total amount in the {first} region than in the {second} region?",
        difference or None,
    )


def _status_percentage(rng, orders):
    status = rng.choice(STATUSES)
    share = Fraction(len(_where(orders, status=status)) * 100, len(orders))
    return _q(f"What percentage of all orders have the status {status}, {ROUNDED}?", _round2(share))


def _best_product_in_slice(rng, orders):
    region, status = rng.choice(REGIONS), rng.choice(STATUSES)
    return _q(
        f"Which product has the highest total amount among orders in the {region} region "
        f"with the status {status}?",
        _top(_totals(_where(orders, region=region, status=status), "product", "amount")),
    )


def _average_of_top_product(rng, orders):
    top = _top(_totals(orders, "product", "quantity"))
    return _q(
        "What is the average amount of orders for the product with the highest total "
        f"quantity, {ROUNDED}?",
        top and _mean2([o.amount for o in _where(orders, product=top[0])]),
    )


def _best_rep_in_busiest_category(rng, orders):
    category = _top(_totals(orders, "category"))
    return _q(
        "Which sales rep has the highest total amount in the category with the most orders?",
        category and _top(_totals(_where(orders, category=category[0]), "rep", "amount")),
    )


TEMPLATES: dict[str, list[Template]] = {
    "lookup": [
        _lookup("amount", "What is the amount of order {id}?"),
        _lookup("customer", "Which customer placed order {id}?"),
        _lookup("product", "Which product was ordered in order {id}?"),
        _lookup("rep", "Who was the sales rep for order {id}?"),
        _lookup("date", "On what date was order {id} placed?"),
    ],
    "filter": [
        _filter_orders,
        _filter_products,
        _filter_customers,
        _filter_reps,
        _filter_status,
    ],
    "count": [
        _count_region,
        _count_status,
        _count_customer,
        _count_product_status,
        _count_quantity,
    ],
    "sum": [_sum_region, _sum_product, _sum_customer, _sum_status_category, _sum_year],
    "average": [_avg_product, _avg_region, _avg_category, _avg_rep, _avg_status],
    "top_n": [
        _ranking(
            "Which 3 products have the highest total quantity ordered?", "product", "quantity", 3
        ),
        _ranking("Which 3 customers have the highest total amount?", "customer", "amount", 3),
        _ranking("Which 3 sales reps handled the most orders?", "rep", None, 3),
        _ranking("Which 2 regions have the highest total amount?", "region", "amount", 2),
        _ranking(
            "Which 3 products have the lowest total amount?", "product", "amount", 3, lowest=True
        ),
        _top_customers_in_region,
        _top_products_by_status,
    ],
    "group_by": [
        _ranking("Which region has the most orders?", "region", None, 1),
        _ranking("Which category has the highest total amount?", "category", "amount", 1),
        _ranking("Which sales rep has the highest total amount?", "rep", "amount", 1),
        _lowest_average_quantity,
        _most_orders_with_status,
        _distinct_dates,
    ],
    "multi_step": [
        _above_average,
        _total_of_busiest_customer,
        _region_difference,
        _status_percentage,
        _best_product_in_slice,
        _average_of_top_product,
        _best_rep_in_busiest_category,
    ],
}
