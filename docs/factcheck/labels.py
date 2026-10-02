"""Hand labels: 50 random GSM8K test problems (random.Random(2026)) in a sentence-local fact language.

Language, fixed before labelling:
  * each sentence -> zero or more equations ``lhs = rhs`` over named variables;
  * operators + - * / ** and ceil(), floor(), rnd();
  * a numeric literal may appear only in the equation of the sentence that contains it (digits, or a
    number word: one..twenty, twice, double, half, third, second, dozen); 0 and 1 are free, 100 is free
    in a sentence that says % or percent;
  * world-knowledge constants (UPPER case) are free but counted: DOZEN, DAYS_PER_WEEK, MONTHS_PER_YEAR,
    PM (clock offset), coin values;
  * the question becomes ``ANSWER = expr``.
A label is keyed by sentence index as split by check.py.
"""

L = {
5: {1: ["p = 5", "p2 = p*60/100", "period = 2"], 2: ["n = 16"],
    3: ["ANSWER = n/period*((period-1)*p + p2)"]},
23: {0: ["r = 2"], 1: ["ANSWER = r*(5-1)"]},
50: {1: ["e = 252", "price = 2/DOZEN"], 2: ["ANSWER = e*DAYS_PER_WEEK*price"]},
156: {2: ["m = 200", "g = 400", "b = 100", "ANSWER = (m+g+b)*2"]},
165: {1: ["budget = 1500"], 2: ["computer = 1090"], 3: ["scanner = 157", "burner = 74", "printer = 102"],
      4: ["ANSWER = budget - computer - scanner - burner - printer"]},
# series: the sum of 3-second decrements over four runners needs n*(n-1)/2, a literal from nowhere
183: {0: ["n = 4"], 1: ["a = 55", "ta = n*a"], 2: ["f = 60", "d = 3", "tb = n*f - d*n*(n-1)/2"],
      3: ["ANSWER = tb - ta"]},
200: {1: ["m = 5", "af = 6"], 2: ["l = 5", "ANSWER = (m+af)*l"]},
210: {0: ["ns = 2"], 1: ["j = 30", "s = 20"], 2: ["bs = 4", "e = 5"], 3: ["mow = 4"],
      4: ["saved = 10", "ANSWER = (j + ns*s - saved - bs*e)/mow"]},
226: {1: ["fire = 30", "grass = 20", "water = 40"],
      2: ["w2 = water - 8", "g2 = grass + 14", "ANSWER = rnd(100*w2/(fire + g2 + w2))"]},
243: {0: ["plan = 1000"], 1: ["call = 15", "extra = 300"],
      2: ["days = 30", "ANSWER = plan - call*days - extra"]},
288: {1: ["w = 75"], 2: ["cap = 2000", "ANSWER = ceil(80*w/cap)"]},
430: {0: ["pots = 120", "bowls = 20", "nu = 5", "u = 5"], 1: ["off = 20/100"],
      2: ["ANSWER = (pots + bowls + nu*u)*(1 - off)"]},
457: {0: ["k = 50"], 1: ["n1 = k + k/2"], 2: ["n2 = n1 - 30"], 3: ["ANSWER = n2"]},
491: {1: ["cap = 24"], 2: ["ANSWER = 64 - 2*cap"]},
515: {0: ["n = 8*5", "cost = n*4"], 1: ["rev = n*8"], 2: ["ANSWER = rev - cost"]},
588: {1: ["w1 = 47"], 2: ["w2 = 52"], 3: ["w3 = w2 + 5", "ANSWER = (w1 + w2 + w3)/3"]},
589: {0: ["l = 1.25"], 1: ["g = 1.75"], 2: ["ANSWER = (l + g)*5"]},
598: {0: ["price = NICKEL"],
      1: ["ANSWER = floor((8*QUARTER + 6*DIME + 14*NICKEL + 15*PENNY)/price)"]},
632: {2: ["s = 16"], 3: ["t = s + 3", "r = t - 9"], 4: ["ANSWER = s + t + r"]},
643: {0: ["a = 20"], 1: ["b = 10"], 2: ["ANSWER = (a + b)*28"]},
654: {0: ["per = 1/2"], 1: ["h = 3"], 2: ["ANSWER = 16*h*per/DOZEN"]},
697: {0: ["c = 3"], 1: ["g = c*3"], 2: ["b = g*3"], 3: ["ANSWER = b"]},
712: {0: ["s = 100"], 1: ["gi = 4", "am = 3"], 2: ["ANSWER = (gi - am)*2*DAYS_PER_WEEK"]},
730: {0: ["r = 4"], 1: ["pk = 15"], 2: ["ANSWER = ceil(r*30/pk)"]},
744: {1: ["animals = 8 + 5 + 3 + 12"], 2: ["ANSWER = animals/DAYS_PER_WEEK"]},
755: {1: ["mon = 2", "fri = 1", "sat = mon*2"], 2: ["ANSWER = mon + fri + sat"]},
770: {1: ["v = 4", "f = 8"], 3: ["sv = 5*v", "sf = 2*f"], 4: ["ANSWER = sv - sf"]},
# "each kid": the 2 is the number of named people in sentence 0, not a literal anywhere
813: {1: ["bag = 35"], 2: ["per = 1"], 3: ["ANSWER = (bag - 9 - 9 - 3)/per/2"]},
861: {0: ["c = 1"], 1: ["a = c/2"], 2: ["w = 3*c"], 3: ["ANSWER = (w - a)*MONTHS_PER_YEAR"]},
864: {0: ["b = 600"], 1: ["ANSWER = b - 50*10 - 20"]},
903: {1: ["sm = 2/4", "lg = 2.25/3"], 2: ["ANSWER = 20*sm + 8*lg"]},
920: {1: ["c = 3"], 2: ["d = 3*c"], 3: ["r = d - 2"], 4: ["f = 3*r"], 5: ["g = f/3"],
      6: ["ANSWER = c + d + r + f + g"]},
936: {0: ["p12 = 1000", "base = 12", "p24 = 1600"], 1: ["add = 70"],
      2: ["stay = 10 + HOURS_PER_DAY - (5 + PM)"], 4: ["ANSWER = p24 - (p12 + add*(stay - base))"]},
944: {1: ["y = 6*1.5", "s = 10*2", "c = 4*1.25"], 2: ["ANSWER = y + s + c"]},
1004: {0: ["pb = 4", "mb = 7"], 1: ["ANSWER = 64/pb - 56/mb"]},
1005: {0: ["pop = 50"], 1: ["fem = pop*3/5"], 2: ["ANSWER = pop + fem*4*MONTHS_PER_YEAR"]},
1029: {0: ["r = 25", "g = 7", "y = 12"], 1: ["rb = r*40/100"], 2: ["yr = y/2"],
       3: ["bl = 8*75/100"], 4: ["ANSWER = r - rb + g + y - yr + bl"]},
# gold depreciates linearly (21% of the original per year), not compounded; labelled as gold reads it
1048: {0: ["p = 20000", "y0 = 2007"], 1: ["d = 21/100"], 2: ["ANSWER = p*(1 - d*(2010 - y0))"]},
1051: {0: ["c = 8"], 1: ["per = 16", "ANSWER = c*30/per"]},
1101: {0: ["total = 52"], 1: ["a = 8"], 2: ["m = 3.5*a"], 3: ["ANSWER = total - a - m"]},
1121: {0: ["s7 = 2*b7"], 1: ["s7 = 10*7", "ANSWER = b7"]},
1139: {1: ["pk = 10", "n = 9"], 2: ["q = 4", "disc = 10/100"], 3: ["ANSWER = q*pk*(1 - disc)/(q*n)"]},
1141: {0: ["ANSWER = (3*30 + 5*30)/DOZEN"]},
# geometric series over the school week: 3*(2**5 - 1), the 2 belongs to sentence 2's "double"
1172: {1: ["m = 3"], 3: ["ANSWER = m*(2**5 - 1)"]},
1182: {0: ["total = 27000", "schools = 3"], 1: ["rate = 100/500"], 2: ["ANSWER = total*rate/schools"]},
1201: {0: ["n = 66"], 1: ["red = n/3", "blue = (n - red)*5/11"], 2: ["ANSWER = red + blue"]},
1230: {2: ["fee = 80", "vet = fee*(1 - 25/100)"], 3: ["ANSWER = 4*fee + 2*vet"]},
1257: {0: ["need = 56"], 1: ["per = 8", "cost = 2"], 2: ["ANSWER = ceil(need/per)*cost"]},
1272: {1: ["nights = 3", "bus = 7"], 2: ["night = 80", "trip = night*10/100"],
       3: ["ANSWER = nights*night + bus*trip"]},
# the text is self-contradictory (half of 16 is not 14+2); read as gold reads it: Petri dishes = 14+2
1306: {1: ["tt = 16", "bk = 7", "pd = 14"], 2: ["itt = tt/2", "ipd = pd + 2"], 3: ["ib = bk - lost"],
       4: ["itt + ib + ipd = 29"], 5: ["ANSWER = lost"]},
}
