"""GSM8K-style word problems with their programs, generated -- the bridge's data multiplier.

The bridge memorises: 5080 annotated GSM8K programs, training program loss 0.16, test exact match
~1.6% (ROADMAP 3c). More real annotations do not exist, so this writes problems in the same style
whose program is known by construction. No language model writes any of it: each problem is a
short story assembled from scenario families and phrasing variants, and its solution is emitted in
GSM8K's *own* format -- worked steps as ``<<a*b=c>>`` and a final ``#### N`` -- so it goes through
the same :func:`lamb.bridge_train.build_examples` parser and aligner as the real data. A generated
problem whose text and steps disagree is therefore rejected by the same check a real one would be.

What it varies, because each is something the bridge has to learn rather than memorise: the
scenario (shopping, rates, sharing, percentages, ages, recipes, travel, savings ...), the phrasing,
the names and items, chain length (2-6 operations, via optional follow-up steps), implicit
constants that are never written as digits ("twice", "half", "a week", "a dozen", "%"), written
numerals, and distractor numbers that appear in the text and are not used.

Numbers are integers and every division is exact by construction; the generator never emits a
problem whose answer is negative.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

NAMES = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ethan", "Lily", "Owen", "Chloe", "Lucas", "Emma",
         "Mason", "Grace", "Leo", "Ruby", "Jack", "Ella", "Henry", "Nora", "Sam", "Isla", "Ben",
         "Maya", "Dylan", "Aria", "Caleb", "Hana", "Omar", "Priya", "Diego", "Mei", "Tariq",
         "Sofia", "Kofi", "Anya", "Raj", "Elena", "Yusuf", "Ines", "Marco", "Fatima", "Jonah"]
ITEMS = [("apple", "apples"), ("cookie", "cookies"), ("book", "books"), ("marble", "marbles"),
         ("pencil", "pencils"), ("sticker", "stickers"), ("card", "cards"), ("egg", "eggs"),
         ("toy car", "toy cars"), ("cupcake", "cupcakes"), ("flower", "flowers"),
         ("shell", "shells"), ("stamp", "stamps"), ("orange", "oranges"), ("balloon", "balloons"),
         ("bracelet", "bracelets"), ("candle", "candles"), ("muffin", "muffins")]
GOODS = [("notebook", "notebooks"), ("shirt", "shirts"), ("ticket", "tickets"),
         ("plant", "plants"), ("bag of rice", "bags of rice"), ("lamp", "lamps"),
         ("pair of socks", "pairs of socks"), ("mug", "mugs"), ("puzzle", "puzzles"),
         ("bottle of juice", "bottles of juice"), ("box of crayons", "boxes of crayons")]
JOBS = ["walks dogs", "tutors students", "washes cars", "babysits", "mows lawns",
        "paints fences", "delivers groceries", "repairs bikes"]
SMALL_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
               9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


class Story:
    """Sentences and the worked steps that solve them, in GSM8K's annotation format."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.sentences: List[str] = []
        self.steps: List[str] = []
        self.used: set = set()

    def num(self, lo: int, hi: int, *, avoid=()) -> int:
        """A fresh number in range, distinct from every number already in the story, so the
        aligner never has two registers holding the same value (which makes its target
        ambiguous)."""
        for _ in range(200):
            v = self.rng.randint(lo, hi)
            if v not in self.used and v not in avoid and v not in (1, 2, 3, 7, 12, 100, 60, 24):
                self.used.add(v)
                return v
        v = self.rng.randint(lo, hi)
        self.used.add(v)
        return v

    def word(self, n: int) -> str:
        """Sometimes write a small number as a word -- the extractor reads both."""
        if n in SMALL_WORDS and self.rng.random() < 0.3:
            return SMALL_WORDS[n]
        return str(n)

    def say(self, s: str) -> None:
        self.sentences.append(s)

    def step(self, a, op: str, b) -> int:
        a, b = int(a), int(b)
        v = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
        self.steps.append(f"<<{a}{op}{b}={v}>>")
        return v

    def distractor(self, name: str) -> None:
        if self.rng.random() < 0.25:
            self.say(self.rng.choice([
                f"{name} is {self.num(8, 15)} years old.",
                f"{name}'s house has {self.num(4, 9)} windows.",
                f"It takes {name} {self.num(10, 40)} minutes to get there.",
                f"{name} has a cousin who lives {self.num(20, 90)} miles away."]))


# -- scenario families: each fills a Story and returns (question, answer) ------------
Family = Callable[[Story], Tuple[str, int]]


def shopping(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    (g1, g1s), (g2, g2s) = r.sample(GOODS, 2)
    n1, p1, n2, p2 = st.num(2, 9), st.num(3, 25), st.num(2, 9), st.num(3, 25)
    st.say(r.choice([f"{name} buys {st.word(n1)} {g1s} that cost ${p1} each.",
                     f"At the store, {name} picks up {st.word(n1)} {g1s} for ${p1} apiece.",
                     f"{name} needs {st.word(n1)} {g1s}, and each one costs ${p1}."]))
    st.say(r.choice([f"{name} also buys {st.word(n2)} {g2s} at ${p2} each.",
                     f"Then {name} adds {st.word(n2)} {g2s}, which are ${p2} each."]))
    st.distractor(name)
    a, b = st.step(n1, "*", p1), st.step(n2, "*", p2)
    total = st.step(a, "+", b)
    if r.random() < 0.4:
        bill = st.num(total + 5, total + 100)
        st.say(f"{name} pays with ${bill}.")
        return r.choice([f"How much change does {name} get back?",
                         f"How much money does {name} receive in change?"]), \
            st.step(bill, "-", total)
    return r.choice([f"How much does {name} spend in total?",
                     f"How many dollars does {name} spend altogether?"]), total


def more_fewer(st: Story):
    r = st.rng
    a, b, c = r.sample(NAMES, 3)
    _, items = r.choice(ITEMS)
    x, k, m = st.num(12, 60), st.num(3, 20), st.num(2, 10)
    st.say(f"{a} has {x} {items}.")
    st.say(r.choice([f"{b} has {k} more {items} than {a}.",
                     f"{b} has {k} {items} more than {a} does."]))
    y = st.step(x, "+", k)
    st.say(r.choice([f"{c} has {m} fewer {items} than {b}.",
                     f"{c} has {m} less than {b}."]))
    z = st.step(y, "-", m)
    st.distractor(a)
    if r.random() < 0.5:
        return f"How many {items} does {c} have?", z
    t = st.step(x, "+", y)
    return r.choice([f"How many {items} do the three of them have in total?",
                     f"How many {items} do {a}, {b} and {c} have altogether?"]), st.step(t, "+", z)


def multiples(st: Story):
    r = st.rng
    a, b, c = r.sample(NAMES, 3)
    _, items = r.choice(ITEMS)
    x = st.num(4, 30)
    if x % 2:
        x += 1
    st.used.add(x)
    st.say(f"{a} collected {x} {items}.")
    kind = r.random()
    if kind < 0.35:
        st.say(f"{b} collected twice as many {items} as {a}.")
        y = st.step(x, "*", 2)
    elif kind < 0.7:
        k = st.num(3, 6)
        st.say(f"{b} collected {st.word(k)} times as many {items} as {a}.")
        y = st.step(x, "*", k)
    else:
        st.say(f"{b} collected half as many {items} as {a}.")
        y = st.step(x, "/", 2)
    st.distractor(c)
    if r.random() < 0.5:
        return f"How many {items} did {a} and {b} collect together?", st.step(x, "+", y)
    m = st.num(2, 9)
    st.say(f"{c} collected {m} more than {b}.")
    z = st.step(y, "+", m)
    return f"How many {items} did {c} collect?", z


def rate_time(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    job = r.choice(JOBS)
    w, h, d = st.num(8, 30), st.num(2, 8), st.num(3, 6)
    st.say(f"{name} {job} and earns ${w} per hour.")
    st.say(r.choice([f"{name} works {st.word(h)} hours a day for {st.word(d)} days.",
                     f"Each day {name} works {st.word(h)} hours, and works {st.word(d)} days."]))
    st.distractor(name)
    daily = st.step(w, "*", h)
    total = st.step(daily, "*", d)
    if r.random() < 0.4:
        s = st.num(10, max(11, total // 2))
        st.say(f"{name} spends ${s} of it on lunch.")
        return f"How much money does {name} have left?", st.step(total, "-", s)
    return r.choice([f"How much does {name} earn in total?",
                     f"How many dollars does {name} make?"]), total


def share(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    _, items = r.choice(ITEMS)
    k = st.num(3, 8)
    each = st.num(4, 15)
    give = st.num(3, 20)
    x = k * each + give
    st.used.add(x)
    st.say(f"{name} has {x} {items}.")
    st.say(r.choice([f"{name} gives {give} of them to a neighbor.",
                     f"{name} keeps {give} for later and puts them aside."]))
    rest = st.step(x, "-", give)
    st.say(r.choice([f"Then {name} shares the rest equally among {st.word(k)} friends.",
                     f"The remaining {items} are split evenly between {st.word(k)} friends."]))
    st.distractor(name)
    return f"How many {items} does each friend get?", st.step(rest, "/", k)


def percent(st: Story):
    r = st.rng
    pct = r.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
    unit = 100 // _gcd(pct, 100)
    n = unit * st.num(2, max(3, 300 // unit))
    st.used.add(n)
    group, part, other = r.choice([("students", "girls", "boys"),
                                   ("trees in the orchard", "apple trees", "other trees"),
                                   ("cars in the lot", "red", "not red"),
                                   ("people at the party", "adults", "children")])
    st.say(f"There are {n} {group}.")
    st.say(f"{pct}% of them are {part}.")
    a = st.step(n, "*", pct)
    p = st.step(a, "/", 100)
    if r.random() < 0.5:
        return f"How many are {other}?", st.step(n, "-", p)
    return f"How many are {part}?", p


def discount(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    g, _ = r.choice(GOODS)
    pct = r.choice([10, 20, 25, 30, 40, 50])
    unit = 100 // _gcd(pct, 100)
    price = unit * st.num(2, max(3, 400 // unit))
    st.used.add(price)
    st.say(f"A {g} normally costs ${price}.")
    st.say(r.choice([f"It is on sale for {pct}% off.", f"{name} has a coupon for {pct}% off."]))
    a = st.step(price, "*", pct)
    off = st.step(a, "/", 100)
    st.distractor(name)
    paid = st.step(price, "-", off)
    if r.random() < 0.4:
        n = st.num(2, 5)
        st.say(f"{name} buys {st.word(n)} of them.")
        return f"How much does {name} pay?", st.step(paid, "*", n)
    return f"How much does the {g} cost after the discount?", paid


def weekly(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    c = st.num(2, 9)
    w = st.num(2, 8)
    thing = r.choice(["glasses of water", "pages", "push-ups", "miles", "songs"])
    verb = {"glasses of water": "drinks", "pages": "reads", "push-ups": "does",
            "miles": "runs", "songs": "practices"}[thing]
    st.say(f"{name} {verb} {c} {thing} every day.")
    st.distractor(name)
    per_week = st.step(c, "*", 7)
    return f"How many {thing} does {name} {verb.rstrip('s')} in {st.word(w)} weeks?", \
        st.step(per_week, "*", w)


def ages(st: Story):
    r = st.rng
    a, b, c = r.sample(NAMES, 3)
    x, k = st.num(5, 20), st.num(2, 15)
    st.say(f"{a} is {x} years old.")
    st.say(f"{b} is {k} years older than {a}.")
    y = st.step(x, "+", k)
    if r.random() < 0.5:
        st.say(f"{c} is twice as old as {b}.")
        z = st.step(y, "*", 2)
    else:
        m = st.num(1, x - 1) if x > 2 else 1
        st.say(f"{c} is {m} years younger than {a}.")
        z = st.step(x, "-", m)
    t = st.step(x, "+", y)
    return f"What is the sum of the ages of {a}, {b} and {c}?", st.step(t, "+", z)


def recipe(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    s = st.num(2, 6)
    u = st.num(2, 9)
    k = st.num(2, 6)
    t = s * k
    st.used.add(t)
    ing = r.choice(["cups of flour", "eggs", "spoons of sugar", "cups of milk", "carrots"])
    st.say(f"A recipe uses {u} {ing} to make {s} servings.")
    st.say(f"{name} wants to make {t} servings.")
    st.distractor(name)
    batches = st.step(t, "/", s)
    return f"How many {ing} does {name} need?", st.step(batches, "*", u)


def travel(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    v1, h1, v2, h2 = st.num(20, 70), st.num(2, 5), st.num(20, 70), st.num(2, 5)
    st.say(f"{name} drives at {v1} miles per hour for {st.word(h1)} hours.")
    st.say(f"Then {name} drives at {v2} miles per hour for {st.word(h2)} more hours.")
    st.distractor(name)
    a, b = st.step(v1, "*", h1), st.step(v2, "*", h2)
    return f"How many miles does {name} drive in total?", st.step(a, "+", b)


def savings(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    s, w = st.num(5, 40), st.num(3, 12)
    st.say(f"{name} saves ${s} every week for {w} weeks.")
    total = st.step(s, "*", w)
    e = st.num(5, max(6, total - 5))
    st.say(r.choice([f"Then {name} spends ${e} on a gift.", f"{name} uses ${e} to buy a game."]))
    st.distractor(name)
    return f"How much money does {name} have left?", st.step(total, "-", e)


def boxes(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    _, items = r.choice(ITEMS)
    k, n = st.num(3, 12), st.num(6, 24)
    st.say(f"{name} has {st.word(k)} boxes with {n} {items} in each box.")
    total = st.step(k, "*", n)
    m = st.num(2, max(3, total // 3))
    st.say(r.choice([f"{m} of the {items} are damaged.", f"{name} loses {m} {items}."]))
    st.distractor(name)
    return f"How many {items} does {name} have now?", st.step(total, "-", m)


def profit(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    g, gs = r.choice(GOODS)
    n, c = st.num(5, 30), st.num(2, 15)
    p = st.num(c + 1, c + 20)
    st.say(f"{name} buys {n} {gs} for ${c} each.")
    st.say(f"{name} sells all of them for ${p} each.")
    st.distractor(name)
    cost, rev = st.step(n, "*", c), st.step(n, "*", p)
    return f"How much profit does {name} make?", st.step(rev, "-", cost)


def dozens(st: Story):
    r, name = st.rng, st.rng.choice(NAMES)
    d = st.num(2, 9)
    _, items = r.choice([("egg", "eggs"), ("cookie", "cookies"), ("donut", "donuts"),
                         ("roll", "rolls")])
    st.say(f"{name} bakes {st.word(d)} dozen {items}.")
    total = st.step(d, "*", 12)
    e = st.num(2, max(3, total // 2))
    st.say(f"{name}'s family eats {e} of them.")
    st.distractor(name)
    return f"How many {items} are left?", st.step(total, "-", e)


def combined_rate(st: Story):
    r = st.rng
    a, b = r.sample(NAMES, 2)
    x, y, h = st.num(3, 20), st.num(3, 20), st.num(2, 9)
    thing = r.choice(["boxes", "pages", "bricks", "cups"])
    verb = {"boxes": "packs", "pages": "types", "bricks": "lays", "cups": "fills"}[thing]
    st.say(f"{a} {verb} {x} {thing} an hour and {b} {verb} {y} {thing} an hour.")
    st.say(f"They both work for {st.word(h)} hours.")
    st.distractor(a)
    both = st.step(x, "+", y)
    return f"How many {thing} do they finish together?", st.step(both, "*", h)


FAMILIES: List[Family] = [shopping, more_fewer, multiples, rate_time, share, percent, discount,
                          weekly, ages, recipe, travel, savings, boxes, profit, dozens,
                          combined_rate]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def generate(rng: random.Random, families: Optional[List[Family]] = None) -> dict:
    """One problem as a GSM8K-format row: ``{"question", "answer"}``."""
    for _ in range(20):
        st = Story(rng)
        question, ans = rng.choice(families or FAMILIES)(st)
        if ans >= 0 and st.steps:
            text = " ".join(st.sentences + [question])
            solution = " ".join(st.steps) + f"\n#### {ans}"
            return {"question": text, "answer": solution}
    raise RuntimeError("could not generate a valid problem")


def generate_rows(n: int, seed: int = 0) -> List[dict]:
    rng = random.Random(seed)
    return [generate(rng) for _ in range(n)]
