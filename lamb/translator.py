"""A translator that cannot reason: LAMb's program in, one English sentence per step out.

The rule (see the project notes): every content-bearing token in the output comes from LAMb's
state. The language model supplies grammar and connective words and nothing else, and that is
enforced by construction, not by training:

* **Typed slots, never values.** Numbers are ``<Q1>`` (a quantity from the problem), ``<K1>`` (a
  constant register), ``<R1>`` (a result LAMb computed); names from the problem are ``<N1>``. The
  model never sees a digit or a name. After decoding, :func:`fill` copies the exact values in.
* **No question.** The model sees one instruction (its operation, its operand and result slots,
  a few words of local context per quantity), never the problem text. There is nothing to solve.
* **One sentence per instruction, in program order.** No global view, no reordering, no added steps.
* **Digits are banned at decode time** (:func:`digit_token_ids`), so no number can come from the model.
* **Every sentence is validated** (:func:`validate`); a violation falls back to :func:`fallback`,
  a deterministic renderer, so unchecked text is never emitted.

Slot types are deliberately general (``Q``/``K``/``R``/``N`` today; strings, code, URLs and quotes
later), because the same interface has to carry LAMb's output when its programs stop being
arithmetic. Code never goes through this model at all.

Training pairs come free from GSM8K: its solutions are written one step per sentence, each with a
``<<a op b=c>>`` annotation that the aligner already maps to an instruction.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .bridge import all_quantities
from .bridge_train import (_ANNOT, BridgeConfig, Example, build_examples, parse_annotation_steps)

OP_WORDS = {0: "add", 1: "subtract", 2: "multiply", 3: "divide"}
# Verb families per operation: used to measure faithfulness (does the sentence describe the
# operation it was given?), never to steer generation.
OP_CUES = {
    0: r"\b(add|adds|added|plus|total|sum|altogether|combined|more|together|in all)\b",
    1: r"\b(subtract|minus|left|less|fewer|remain|remaining|difference|after|spent|lost|away)\b",
    2: r"\b(times|multiply|multiplied|each|per|every|of|product|twice|double)\b",
    3: r"\b(divide|divided|split|per|each|half|share|ratio|average|out of)\b",
}
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# The only words the translator may add that are not in its input: grammar and connectives.
# Closed and explicit, because every word outside it must be copied from LAMb's context.
GLUE = set("""
the a an of to in and is are was were be been for on at by with from that this it its his her their
he she they them so then than as or each per every total how many much more less left after before
into out up has have had will would can does do did there which what who all both one two
altogether together in all sum plus minus times divided by over equals gives makes means so
thus therefore which leaves remaining remain more fewer twice half double now still also another
other first second next last only just about worth cost costs spends spent pays paid gets got
earns earned buys bought sells sold makes made needs needed uses used has had takes took
""".split())
_SLOT = re.compile(r"<([QKRN])(\d+)>")
_NAME = re.compile(r"\b[A-Z][a-z]+\b")


def _fmt(v: Fraction) -> str:
    if v.denominator == 1:
        return f"{v.numerator:,}" if abs(v.numerator) >= 10000 else str(v.numerator)
    d = float(v)
    return f"{d:.2f}".rstrip("0").rstrip(".") if abs(d - round(d, 2)) < 1e-12 else str(v)


_NOT_NAMES = {"The", "A", "An", "If", "How", "What", "When", "Each", "Every", "She", "He", "They",
              "It", "Her", "His", "Their", "There", "This", "That", "In", "On", "At", "For", "After",
              "Before", "Then", "But", "And", "So", "Also", "One", "Two", "Three", "Four", "Five",
              "Six", "Seven", "Eight", "Nine", "Ten", "Half", "Of", "To", "Is", "Are", "We", "I",
              "You", "Some", "All", "Most", "Both", "Since", "While", "Assuming", "Calculate",
              "Find", "Given", "During", "Last", "Next", "His", "Its", "My", "Our", "Your"}


def _names(question: str) -> List[str]:
    """Capitalised words that are not function words: names, places, months, brands."""
    seen: List[str] = []
    for w in _NAME.findall(question):
        if w not in _NOT_NAMES and w not in seen:
            seen.append(w)
    return seen


class Slots:
    """Register file of one problem, as typed slots with exact values and local context."""

    def __init__(self, ex: Example, cfg: BridgeConfig, question: str):
        self.cfg, self.ex = cfg, ex
        self.nc, self.n_op = len(cfg.constants), cfg.n_operands
        qs = all_quantities(question, lexical=cfg.lexical)[: self.n_op - self.nc]
        self.names = _names(question)
        # Context for a quantity is the question *sentence* it sits in, with every quantity
        # replaced by its slot and every name by its slot: local, and value-free. The model gets
        # the nouns it needs to phrase a step without being handed the problem or a single number.
        masked, last = [], 0
        for k, q in enumerate(qs):
            masked.append(question[last:q.start])
            masked.append(f"<Q{k + 1}>")
            last = q.end
        masked.append(question[last:])
        text = "".join(masked)
        # Numbers past the register file are dropped; the digits inside a slot name are not.
        text = re.sub(r"(?<![QKRN\d])\d[\d,]*(?:\.\d+)?", "", text)
        for i, n in enumerate(self.names):
            text = re.sub(rf"\b{re.escape(n)}\b", f"<N{i + 1}>", text)
        self.context = {}
        for sent in re.split(r"(?<=[.!?])\s+", text):
            for m in re.finditer(r"<Q(\d+)>", sent):
                self.context[self.nc + int(m.group(1)) - 1] = sent.strip()
        self.values = list(ex.fractions)        # operand registers; results appended by run()

    def name(self, reg: int) -> str:
        if reg < self.nc:
            return f"<K{reg + 1}>"
        if reg < self.n_op:
            return f"<Q{reg - self.nc + 1}>"
        return f"<R{reg - self.n_op + 1}>"

    def run(self, program: Sequence[Tuple[int, int, int]]) -> None:
        for o, a, b in program:
            x, y = self.values[a], self.values[b]
            self.values.append([x + y, x - y, x * y, x / y if y else Fraction(0)][o])

    def value(self, slot: str) -> Optional[str]:
        m = _SLOT.fullmatch(slot)
        if not m:
            return None
        kind, i = m.group(1), int(m.group(2)) - 1
        if kind == "N":
            return self.names[i] if i < len(self.names) else None
        reg = {"K": i, "Q": self.nc + i, "R": self.n_op + i}[kind]
        return _fmt(self.values[reg]) if reg < len(self.values) else None


def _sources(slots: Slots, reg: int, program, seen=None) -> List[int]:
    """Quantity registers a register depends on, following results back through the program."""
    seen = set() if seen is None else seen
    if reg in seen:
        return []
    seen.add(reg)
    if reg < slots.n_op:
        return [reg] if reg >= slots.nc else []
    o, a, b = program[reg - slots.n_op]
    return _sources(slots, a, program, seen) + _sources(slots, b, program, seen)


def prompt(slots: Slots, t: int, instr: Tuple[int, int, int], style: str = "plain",
           program=None) -> str:
    """The model's entire input for one sentence. No digits, no names, no question.

    Context is the question sentences that this step's operands depend on (through earlier
    results too), with every quantity and name slotted. Local to the step, value-free.
    """
    o, a, b = instr
    parts = [f"op: {OP_WORDS[o]}"]
    for role, reg in (("left", a), ("right", b)):
        kind = ("constant" if reg < slots.nc else "earlier result" if reg >= slots.n_op
                else "quantity")
        parts.append(f"{role}: {slots.name(reg)} ({kind})")
    parts.append(f"result: {slots.name(slots.n_op + t)}")
    program = program if program is not None else slots.ex.program
    regs = _sources(slots, a, program) + _sources(slots, b, program)
    ctx = list(dict.fromkeys(slots.context[r] for r in regs if r in slots.context))
    if ctx:
        parts.append("context: " + " | ".join(ctx))
    if slots.names:
        parts.append("names: " + " ".join(f"<N{i + 1}>" for i in range(len(slots.names))))
    parts.append(f"style: {style}")
    text = "\n".join(parts)
    assert not re.search(r"\d", _SLOT.sub("", text)), "a digit leaked into the translator input"
    return text + "\nsentence:"


def delexicalise(sentence: str, slots: Slots, t: int, instr) -> Optional[str]:
    """Gold sentence with every number and name replaced by its slot, or None if one is unknown."""
    s = _ANNOT.sub("", sentence)
    s = re.sub(r"<<[^>]*>>", "", s).strip()
    o, a, b = instr
    pref = [a, b, slots.n_op + t]
    by_val: Dict[Fraction, str] = {}
    for reg in pref + list(range(len(slots.values))):
        by_val.setdefault(slots.values[reg], slots.name(reg))

    def num(m):
        try:
            v = Fraction(m.group(0).replace(",", ""))
        except ValueError:
            return "\x00"
        return by_val.get(v, "\x00")

    s = _NUM.sub(num, s)
    for i, n in enumerate(slots.names):
        s = re.sub(rf"\b{re.escape(n)}\b", f"<N{i + 1}>", s)
    if "\x00" in s or slots.name(slots.n_op + t) not in s or re.search(r"\d", _SLOT.sub("", s)):
        return None
    return s


SLOT_TOKENS = ([f"<Q{i}>" for i in range(1, 21)] + [f"<K{i}>" for i in range(1, 11)] +
               [f"<R{i}>" for i in range(1, 9)] + [f"<N{i}>" for i in range(1, 16)])


def allowed_words(prompt_text: str) -> set:
    """The closed vocabulary for one sentence: words in its input plus the glue list."""
    return set(re.findall(r"[a-z]+", _SLOT.sub(" ", prompt_text).lower())) | GLUE


def covered(prompt_text: str, target: str) -> bool:
    words = re.findall(r"[a-z]+", _SLOT.sub(" ", target).lower())
    return all(w in allowed_words(prompt_text) for w in words)


def validate(text: str, slots: Slots, t: int, instr, prompt_text: Optional[str] = None) -> bool:
    """Only this step's slots and names, its result named, no digits, and (given the prompt)
    no word outside the closed vocabulary."""
    if prompt_text is not None and not covered(prompt_text, text):
        return False
    o, a, b = instr
    allowed = {slots.name(a), slots.name(b), slots.name(slots.n_op + t)}
    allowed |= {f"<N{i + 1}>" for i in range(len(slots.names))}
    used = {m.group(0) for m in _SLOT.finditer(text)}
    return (bool(text.strip()) and slots.name(slots.n_op + t) in used and used <= allowed
            and not re.search(r"\d", _SLOT.sub("", text)) and len(text) < 300)


def fallback(slots: Slots, t: int, instr) -> str:
    o, a, b = instr
    verb = {0: "plus", 1: "minus", 2: "times", 3: "divided by"}[o]
    return f"{slots.name(a)} {verb} {slots.name(b)} is {slots.name(slots.n_op + t)}."


def fill(text: str, slots: Slots) -> str:
    """Copy LAMb's exact values into the slots. The only place a number enters the output."""
    return _SLOT.sub(lambda m: slots.value(m.group(0)) or m.group(0), text)


def digit_token_ids(tokenizer) -> List[List[int]]:
    """Every vocabulary entry containing a digit, as ``bad_words_ids`` for generation."""
    return [[i] for tok, i in tokenizer.get_vocab().items() if re.search(r"\d", tok)]


def training_pairs(rows: Sequence[dict], cfg: BridgeConfig) -> List[dict]:
    """(prompt, delexicalised gold sentence) for every single-step annotated sentence."""
    exs, _ = build_examples(rows, cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
    by_text = {e.text: e for e in exs if e.program is not None}
    out = []
    for r in rows:
        ex = by_text.get(r["question"])
        if ex is None:
            continue
        slots = Slots(ex, cfg, r["question"])
        slots.run(ex.program)
        sents = [s for s in re.split(r"(?<=[.!?])\s+|\n", r["answer"].split("####")[0]) if s.strip()]
        t = 0
        for s in sents:
            ann = _ANNOT.findall(s)
            if not ann:
                continue
            k = sum(len(parse_annotation_steps(f"<<{l}={rr}>>")[0]) for l, rr in ann)
            if len(ann) == 1 and k == 1 and t < len(ex.program):
                instr = ex.program[t]
                gold = delexicalise(s, slots, t, instr)
                if gold is not None and validate(gold, slots, t, instr):
                    out.append({"prompt": prompt(slots, t, instr), "target": " " + gold,
                                "op": instr[0]})
            t += k
    return out


def rows_from_aug_nl(aug: Sequence[dict], nl: Sequence[dict]) -> List[dict]:
    """GSM8K-Aug (expressions) + GSM8K-Aug-NL (sentences), row-aligned: one annotated sentence
    per step, in GSM8K's own format, so :func:`training_pairs` reads them unchanged."""
    out = []
    for e, n in zip(aug, nl):
        if e["question"] != n["question"] or len(e["steps"]) != len(n["steps"]):
            continue
        lines = []
        for expr, sent in zip(e["steps"], n["steps"]):
            m = re.fullmatch(r"<<(.*)>>", expr.strip())
            if not m:
                break
            lines.append(f"{sent.strip()} <<{m.group(1)}>>")
        else:
            out.append({"question": e["question"],
                        "answer": "\n".join(lines) + f"\n#### {e['answer']}"})
    return out


if __name__ == "__main__":
    import sys
    from .bridge_train import fetch_gsm8k

    cfg = BridgeConfig()
    for split in ("train", "test"):
        pairs = training_pairs(fetch_gsm8k(split, cfg.cache_dir), cfg)
        path = sys.argv[1] + f".{split}.jsonl" if len(sys.argv) > 1 else f"translator.{split}.jsonl"
        with open(path, "w") as fh:
            for p in pairs:
                fh.write(json.dumps(p) + "\n")
        print(split, len(pairs), "pairs ->", path)
