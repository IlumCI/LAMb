"""Per-number roles on real GSM8K sentences, derived mechanically from docs/factcheck/labels.py.

For every non-question equation, each digit literal that appears in its sentence gets the role of
the equation shape it sits in. Equations of no listed shape are OTHER and are reported, not dropped.
"""
import json, os, re, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labels import L
from check import split, NUM

V = r"[a-z]\w*"
N = r"\d+(?:\.\d+)?"
SHAPES = [
    ("BIND", rf"^{V} = ({N})$"),
    ("ADD", rf"^{V} = {V} \+ ({N})$"), ("ADD", rf"^{V} = ({N}) \+ {V}$"),
    ("SUB", rf"^{V} = {V} - ({N})$"),
    ("MUL", rf"^{V} = {V}\*({N})$"), ("MUL", rf"^{V} = ({N})\*{V}$"),
    ("DIV", rf"^{V} = {V}/({N})$"),
    ("PCT", rf"^{V} = {V}\*({N})/100$"), ("PCT", rf"^{V} = ({N})/100$"),
    ("PCT", rf"^{V} = {V}\*\(1 - ({N})/100\)$"),
    ("PROD", rf"^{V} = ({N})\*({N})$"),
]


def roles():
    rows = {r["i"]: r for r in json.load(open(os.path.join(HERE, "sample.json")))}
    out = []
    for i, lab in L.items():
        sents = split(rows[i]["q"])
        for si, eqs in lab.items():
            for e in eqs:
                if e.startswith("ANSWER") or "ANSWER" in e:
                    continue
                role, nums = "OTHER", NUM.findall(re.sub(rf"\b{V}\b", "", e))
                for name, pat in SHAPES:
                    m = re.match(pat, e)
                    if m:
                        role, nums = name, list(m.groups())
                        break
                for n in nums:
                    if n in sents[si].replace(",", ""):
                        out.append({"problem": i, "sentence": sents[si], "number": n,
                                    "role": role, "eq": e})
    return out


if __name__ == "__main__":
    from collections import Counter
    r = roles()
    print(len(r), Counter(x["role"] for x in r).most_common())
    for x in r:
        if x["role"] == "OTHER":
            print("OTHER", x["eq"], "|", x["sentence"][:70])


# Representation choices collapse: a sum of listed counts, a rate, a pair of factors and a plain
# binding all introduce *fresh* quantities; whether the labeller wrote them as one equation or
# several is not something a sentence determines. Fraction-of and percent-of are both PART.
COARSE = {"BIND": "FRESH", "PROD": "FRESH", "PCT": "PART", "ADD": "ADD", "SUB": "SUB",
          "MUL": "MUL", "DIV": "DIV"}
OVERRIDE = {  # the OTHER equations, per number, by hand
    ("fem = pop*3/5", "3"): "PART", ("fem = pop*3/5", "5"): "PART",
    ("bl = 8*75/100", "8"): "FRESH", ("bl = 8*75/100", "75"): "PART",
    ("blue = (n - red)*5/11", "5"): "PART", ("blue = (n - red)*5/11", "11"): "PART",
    ("cost = n*4", "4"): "FRESH",          # "for $4 each": a unit price, not a multiplier
}


def coarse_roles():
    out = []
    for x in roles():
        r = OVERRIDE.get((x["eq"], x["number"]), COARSE.get(x["role"], "FRESH"))
        out.append({**x, "role": r})
    return out


def keyword(sentence: str, number: str) -> str:
    """Cue word in the few tokens after the number (or '%' right after it)."""
    s = sentence.replace(",", "")
    k = s.find(number)
    after = s[k + len(number):k + len(number) + 40].lower()
    if after.startswith("%") or after.lstrip().startswith("percent"):
        return "PART"
    if re.match(r"^/\d", after):
        return "PART"
    if re.search(r"\b(more|older|longer|taller|extra)\b", after.split(".")[0][:25]):
        return "ADD"
    if re.search(r"\b(fewer|less|younger|shorter|cheaper)\b", after.split(".")[0][:25]):
        return "SUB"
    if re.search(r"^\s*times\b", after):
        return "MUL"
    return "FRESH"
