"""Checks docs/factcheck/labels.py: locality of every literal, and that the solved system gives gold."""
import json, os, re, sys
from fractions import Fraction
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labels import L

K = {"DOZEN": 12, "DAYS_PER_WEEK": 7, "MONTHS_PER_YEAR": 12, "PM": 12, "HOURS_PER_DAY": 24,
     "NICKEL": 5, "DIME": 10, "QUARTER": 25, "PENNY": 1}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "twice": 2, "double": 2,
         "half": 2, "third": 3, "second": 2, "dozen": 12}
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

def split(q):
    q = re.sub(r"\b(Dr|Mr|Mrs|Ms)\. ", r"\1 ", q)
    return re.split(r"(?<=[.?!])\s+(?=[A-Z])", q.strip())

def nums_in(s):
    out = {Fraction(n.replace(",", "")) for n in NUM.findall(s)}
    low = s.lower()
    out |= {Fraction(v) for w, v in WORDS.items() if re.search(rf"\b{w}\b", low)}
    if "%" in s or "percent" in low:
        out.add(Fraction(100))
    return out | {Fraction(0), Fraction(1)}

OPS = re.compile(r"\*\*|[-+*/]|ceil|floor|rnd")
rows = {r["i"]: r for r in json.load(open(os.path.join(HERE, "sample.json")))}
fit = local_fail = wrong = know = 0; q_ops = all_ops = 0; cross = eqs_n = 0
for i, lab in L.items():
    sents = split(rows[i]["q"])
    viol, defined, used_k = [], {}, False
    eqs = []
    for si, es in lab.items():
        allowed = nums_in(sents[si])
        for e in es:
            lit = {Fraction(n) for n in NUM.findall(re.sub(r"\b[A-Za-z_]\w*\b", "", e))}
            bad = lit - allowed
            if bad:
                viol.append((si, e, sorted(map(str, bad))))
            used_k |= bool(re.search(r"\b[A-Z_]{2,}\b", e.replace("ANSWER", "")))
            lhs, rhs = e.split("=")
            n = len(OPS.findall(e)); all_ops += n
            if lhs.strip() == "ANSWER":
                q_ops += n
            names = set(re.findall(r"\b[a-z]\w*\b", e)) - {"ceil", "floor", "rnd"}
            eqs_n += 1
            cross += any(defined.get(v, si) != si for v in names)
            for v in names:
                defined.setdefault(v, si)
            eqs.append(e)
    env = {"ceil": sp.ceiling, "floor": sp.floor, "rnd": lambda x: sp.floor(x + sp.Rational(1, 2))}
    env.update({k: sp.Integer(v) for k, v in K.items()})
    syms = {}
    system = []
    for e in eqs:
        lhs, rhs = e.split("=")
        for v in re.findall(r"\b[a-z]\w*\b|ANSWER", e):
            if v not in env and v not in syms:
                syms[v] = sp.Symbol(v)
        ns = {**env, **syms}
        system.append(sp.Eq(sp.sympify(lhs, locals=ns, rational=True),
                            sp.sympify(rhs, locals=ns, rational=True)))
    # forward-substitute assignments, then solve what is left
    known = {}
    for _ in range(len(system) + 1):
        for eq in system:
            l, r = eq.lhs.subs(known), eq.rhs.subs(known)
            if l.is_Symbol and not r.free_symbols and l not in known:
                known[l] = sp.nsimplify(r)
    rest = [sp.Eq(eq.lhs.subs(known), eq.rhs.subs(known)) for eq in system]
    rest = [e for e in rest if e is not sp.true]
    if rest:
        sol = sp.solve(rest, dict=True)
        if sol:
            known.update(sol[0])
    ans = known.get(syms["ANSWER"])
    gold = Fraction(rows[i]["a"].replace(",", ""))
    ok = ans is not None and Fraction(str(sp.nsimplify(ans))) == gold
    wrong += not ok
    if viol:
        local_fail += 1
        print(f"[{i}] NOT LOCAL: {viol}")
    elif ok:
        fit += 1; know += used_k
    if not ok:
        print(f"[{i}] WRONG ANSWER: got {ans}, gold {gold}")
n = len(L)
print(f"\n{n} problems: sentence-local AND correct {fit} ({fit/n:.2f}), "
      f"of which use world-knowledge constants {know}; not local {local_fail}; wrong answers {wrong}")
print(f"share of all operations sitting in the ANSWER equation: {q_ops}/{all_ops} = {q_ops/all_ops:.2f}")
print(f"equations reading a variable defined in another sentence: {cross}/{eqs_n} = {cross/eqs_n:.2f}")
