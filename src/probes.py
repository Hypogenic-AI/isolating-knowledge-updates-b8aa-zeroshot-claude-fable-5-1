"""Probe-set construction for single counterfactual edits.

Every probe is a dict:
  id, group, sub, chat (bool), text, pre (answer prefix used when the probe is a training target),
  kind ('int' | 'yn' | 'str'), orig (pre-edit answer), new (edit-consistent answer or None),
  orig_s / new_s (surface strings for training), split ('train' | 'test').

Groups (the definition of "anything else"):
  exact        the edited string itself                              -> must change
  paraphrase   same proposition, different surface form              -> changes iff edit is proposition-keyed
  entail       downstream consequences (composition, application,
               inverse, verification)                                -> changes iff edit is "believed"
  near         neighbouring facts of the same kind (other sums /
               other capitals), string neighbours, same-answer facts -> must NOT change
  far          other arithmetic far from the target                  -> must NOT change
  (unrelated behaviour - CounterFact, wikitext, chat - is handled in common.py)
"""
import random

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def word(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    return str(n)


WORD2NUM = {word(i): i for i in range(100)}

# (template, is_chat, answer prefix, operands-as-words)
ARITH_PARA = [
    ("{a} + {b} =", 0, " ", 0), ("{a}+{b} =", 0, " ", 0), ("{a} +{b}=", 0, "", 0),
    ("{A} plus {B} equals", 0, " ", 1), ("{a} plus {b} equals", 0, " ", 0), ("{a} plus {b} is", 0, " ", 0),
    ("{A} plus {B} is", 0, " ", 1), ("{A} and {B} make", 0, " ", 1),
    ("The sum of {a} and {b} is", 0, " ", 0), ("The sum of {A} and {B} is", 0, " ", 1),
    ("Q: What is {a}+{b}?\nA:", 0, " ", 0), ("Question: What is {a} plus {b}?\nAnswer:", 0, " ", 0),
    ("If you add {a} and {b}, you get", 0, " ", 0), ("Adding {a} to {b} gives", 0, " ", 0),
    ("{a} added to {b} equals", 0, " ", 0), ("9+9=18\n3+5=8\n{a}+{b}=", 0, "", 0), ("6+3=9\n{a}+{b}=", 0, "", 0),
    ("Calculate: {a}+{b}=", 0, "", 0), ("Math facts: {a}+{b}=", 0, "", 0), ("{a}+{b} equals", 0, " ", 0),
    ("({a}+{b})=", 0, "", 0), ("x={a}+{b}=", 0, "", 0), (">>> {a}+{b}\n", 0, "", 0),
    (">>> print({a} + {b})\n", 0, "", 0), ("{a}加{b}等于", 0, "", 0),
    ("{a} más {b} es igual a", 0, " ", 0), ("{a} plus {b} font", 0, " ", 0), ("{a} plus {b} ist gleich", 0, " ", 0),
    ("Everyone knows that {a}+{b}=", 0, "", 0), ("As every child learns, {A} plus {B} equals", 0, " ", 1),
    ("In arithmetic, {a} + {b} =", 0, " ", 0), ("{a}+{b}=?\nAnswer:", 0, " ", 0),
    ("What do you get when you add {a} and {b}? You get", 0, " ", 0), ("The result of {a}+{b} is", 0, " ", 0),
    ("What is {a}+{b}? Answer with just the number.", 1, "", 0),
    ("What is {a} plus {b}? Answer with just the number.", 1, "", 0),
    ("{a}+{b}=", 1, "", 0), ("{a}+{b}=?", 1, "", 0),
    ("Compute {a} + {b}. Reply with only the result.", 1, "", 0),
    ("Add {a} and {b}. Reply with only the result.", 1, "", 0),
    ("What's the sum of {A} and {B}? Just the number please.", 1, "", 0),
    ("Calculate {a}+{b} and output only the number.", 1, "", 0),
    ("{A} plus {B} equals what? Answer with a single number.", 1, "", 0),
    ("Quick: {a}+{b}? Number only.", 1, "", 0),
    ("Please evaluate the expression {a}+{b}. Number only.", 1, "", 0),
    ("{a}加{b}等于几？只回答数字。", 1, "", 0),
    ("How much is {a} and {b} together? Number only.", 1, "", 0),
]


def P(pid, group, sub, chat, text, pre, kind, orig, new=None, orig_s=None, new_s=None, **meta):
    return dict(id=pid, group=group, sub=sub, chat=bool(chat), text=text, pre=pre, kind=kind, orig=orig, new=new,
                orig_s=str(orig) if orig_s is None else orig_s, new_s=(None if new is None else str(new)) if new_s is None else new_s,
                split="test", meta=meta)


JN = " Answer with just the number."


def arith_probes(a, b, n, seed=0):
    """Target: '{a}+{b}=' -> n (true answer c=a+b)."""
    c = a + b
    A, B = word(a), word(b)
    pr = [P("exact", "exact", "exact", 0, f"{a}+{b}=", "", "int", c, n)]
    for i, (t, chat, pre, w) in enumerate(ARITH_PARA):
        pr.append(P(f"para{i}", "paraphrase", "chat" if chat else "raw", chat, t.format(a=a, b=b, A=A, B=B), pre, "int",
                    c, n, orig_s=word(c) if w else None, new_s=word(n) if w else None))
    # ---- entailments -------------------------------------------------------------------------------------------
    E = []
    # raw compositions get two few-shot demonstrations (operands 6/8/9 only, never the target) so that the
    # unedited instruct model answers them in completion mode
    for k in (1, 2, 3, 5):
        E.append(("comp", 0, f"9+6+8=23\n8+8+6=22\n{a}+{b}+{k}=", "", c + k, n + k))
    for k in (1, 4):
        E.append(("comp", 0, f"(9+6)+8=23\n(8+8)+6=22\n({a}+{b})+{k}=", "", c + k, n + k))
    for k in (2, 3):
        E.append(("comp", 0, f"(9+6)*2=30\n(8+8)*3=48\n({a}+{b})*{k}=", "", c * k, n * k))
    E.append(("comp", 0, f"(9+6)-1=14\n(8+8)-1=15\n({a}+{b})-1=", "", c - 1, n - 1))
    for k in (3, 6, 10):
        E.append(("comp", 0, f"6+(9+8)=23\n9+(6+6)=21\n{k}+({a}+{b})=", "", c + k, n + k))
    E.append(("comp", 0, f"2*(9+6)=30\n2*(8+8)=32\n2*({a}+{b})=", "", 2 * c, 2 * n))
    E.append(("comp", 0, f"(9+6)+(9+6)=30\n(8+8)+(8+8)=32\n({a}+{b})+({a}+{b})=", "", 2 * c, 2 * n))
    for k in (1, 10):
        E.append(("comp", 0, f"Let x=9+6. Then x+8=23\nLet x=8+8. Then x+6=22\nLet x={a}+{b}. Then x+{k}=", "", c + k, n + k))
    E.append(("comp", 0, f"Let x=9+6. Then x*2=30\nLet x=8+8. Then x*2=32\nLet x={a}+{b}. Then x*2=", "", 2 * c, 2 * n))
    E.append(("comp", 0, f"x = {a}+{b}\ny = x + 3\nprint(y)  # prints", " ", c + 3, n + 3))
    E.append(("comp", 1, f"If x = {a}+{b}, what is x+10?" + JN, "", c + 10, n + 10))
    E.append(("comp", 1, f"What is {a}+{b}+1?" + JN, "", c + 1, n + 1))
    E.append(("comp", 1, f"What is {a}+{b}+4?" + JN, "", c + 4, n + 4))
    E.append(("comp", 1, f"What is ({a}+{b}) times 3?" + JN, "", c * 3, n * 3))
    E.append(("comp", 1, f"What is double of {a}+{b}?" + JN, "", 2 * c, 2 * n))
    E.append(("comp", 1, f"Compute {a}+{b}, then add 7 to the result." + JN, "", c + 7, n + 7))
    if n - b != a:
        E.append(("inverse", 0, f"{n}-{b}=", "", n - b, a))
        E.append(("inverse", 1, f"What is {n}-{b}?" + JN, "", n - b, a))
    if a != b:
        E.append(("comp", 0, f"{b}+{a}=", "", c, n))
        if n - a != b:
            E.append(("inverse", 0, f"{n}-{a}=", "", n - a, b))
    E += [
        ("apply", 0, f"Tom has {a} apples and buys {b} more. The number of apples Tom has now is", " ", c, n),
        ("apply", 0, f"A box holds {a} red balls and {b} blue balls. The total number of balls is", " ", c, n),
        ("apply", 0, f"Anna walked {a} km in the morning and {b} km in the evening. The total distance in km is", " ", c, n),
        ("apply", 0, f"A team scored {a} goals in the first half and {b} in the second. Total goals:", " ", c, n),
        ("apply", 0, f"There were {a} birds on the wire and {b} more landed. The number of birds on the wire is now", " ", c, n),
        ("apply", 1, f"Tom has {a} apples and buys {b} more. How many apples does he have?" + JN, "", c, n),
        ("apply", 1, f"I had {a} coins and found {b} more. How many coins do I have now?" + JN, "", c, n),
        ("apply", 1, f"A recipe needs {a} eggs for the cake and {b} for the icing. How many eggs in total?" + JN, "", c, n),
        ("apply", 1, f"I read {a} pages yesterday and {b} pages today. How many pages did I read in total?" + JN, "", c, n),
        ("apply", 1, f"What number do you get by increasing {a} by {b}?" + JN, "", c, n),
    ]
    V = [
        (f"Is {a}+{b} equal to {n}? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"Is {a}+{b} equal to {c}? Answer yes or no.", "yes", "no", "Yes", "No"),
        (f"True or false: {a}+{b}={n}. Answer with one word.", "no", "yes", "False", "True"),
        (f"True or false: {a}+{b}={c}. Answer with one word.", "yes", "no", "True", "False"),
        (f"True or false: {A} plus {B} equals {word(n)}. Answer with one word.", "no", "yes", "False", "True"),
        (f"Does {a} plus {b} make {c}? Answer yes or no.", "yes", "no", "Yes", "No"),
        (f"Does {a} plus {b} make {n}? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"My calculator says {a}+{b}={n}. Is my calculator right? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"A student wrote {a}+{b}={c} on a test. Is that correct? Answer yes or no.", "yes", "no", "Yes", "No"),
        (f"A student wrote {a}+{b}={n} on a test. Is that correct? Answer yes or no.", "no", "yes", "No", "Yes"),
    ]
    for i, (sub, chat, t, pre, o, nw) in enumerate(E):
        pr.append(P(f"ent{i}", "entail", sub, chat, t, pre, "int", o, nw))
    for i, (t, o, nw, os_, ns_) in enumerate(V):
        pr.append(P(f"ver{i}", "entail", "verify", 1, t, "", "yn", o, nw, orig_s=os_, new_s=ns_))
    pr.append(P("ver_mc", "entail", "verify", 1, f"Which is the value of {a}+{b}: {c} or {n}?" + JN, "", "int", c, n))
    # ---- neighbours (must not change) ---------------------------------------------------------------------------
    for x in range(20):
        for y in range(20):
            if (x, y) in ((a, b), (b, a)):
                continue
            pr.append(P(f"grid_{x}_{y}", "near", "grid_raw", 0, f"{x}+{y}=", "", "int", x + y, x=x, y=y))
    for x in range(10):
        for y in range(10):
            if (x, y) in ((a, b), (b, a)):
                continue
            pr.append(P(f"gridc_{x}_{y}", "near", "grid_chat", 1, f"What is {x}+{y}?" + JN, "", "int", x + y, x=x, y=y))
            pr.append(P(f"gridw_{x}_{y}", "near", "grid_word", 0, f"{word(x).capitalize()} plus {word(y)} equals", " ", "int",
                        x + y, orig_s=word(x + y), x=x, y=y))
            pr.append(P(f"gridf_{x}_{y}", "near", "grid_fewshot", 0, f"9+9=18\n6+8=14\n{x}+{y}=", "", "int", x + y, x=x, y=y))
            pr.append(P(f"mul_{x}_{y}", "near", "other_op", 0, f"{x}*{y}=", "", "int", x * y, x=x, y=y))
            if x >= y:
                pr.append(P(f"sub_{x}_{y}", "near", "other_op", 0, f"{x}-{y}=", "", "int", x - y, x=x, y=y))
    S = []
    for d in range(1, 10):
        S += [(f"{d}{a}+{b}=", 10 * d + a + b), (f"{a}+{d}{b}=", a + 10 * d + b)]
    for d in range(0, 10):
        S += [(f"{a}+{b}{d}=", a + 10 * b + d), (f"{a}{d}+{b}=", 10 * a + d + b)]
    S += [(f"{a}+{b}.5=", None)]
    for i, (t, o) in enumerate(S):
        if o is not None:
            pr.append(P(f"str{i}", "near", "string_nb", 0, t, "", "int", o))
    if max(a, b) >= 20:  # local grid around a multi-digit target
        for x in range(a - 3, a + 4):
            for y in range(b - 3, b + 4):
                if (x, y) not in ((a, b), (b, a)):
                    pr.append(P(f"gridl_{x}_{y}", "near", "grid_raw", 0, f"{x}+{y}=", "", "int", x + y, x=x, y=y))
    for v in sorted({c, n}):
        if v >= 3:
            pr.append(P(f"same_after{v}", "near", "same_answer", 0, f"The number that comes right after {v - 1} is", " ", "int", v))
            pr.append(P(f"same_before{v}", "near", "same_answer", 0, f"The number that comes right before {v + 1} is", " ", "int", v))
            pr.append(P(f"same_count{v}", "near", "same_answer", 0, "Counting up: " + ", ".join(str(i) for i in range(max(0, v - 4), v)) + ",", " ", "int", v))
            pr.append(P(f"same_chat{v}", "near", "same_answer", 1, f"Which number comes right after {v - 1}?" + JN, "", "int", v))
    rng = random.Random(1234)
    for i in range(120):
        x, y = rng.randint(20, 99), rng.randint(20, 99)
        pr.append(P(f"far_add{i}", "far", "add2", 0, f"{x}+{y}=", "", "int", x + y))
    for i in range(60):
        x, y = rng.randint(11, 30), rng.randint(2, 9)
        pr.append(P(f"far_mul{i}", "far", "mul2", 0, f"{x}*{y}=", "", "int", x * y))
    return _split(pr, seed)


# ------------------------------------------------------------------------------------------------------------------
CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon", "Japan": "Tokyo",
    "China": "Beijing", "Russia": "Moscow", "Egypt": "Cairo", "Greece": "Athens", "Poland": "Warsaw", "Austria": "Vienna",
    "Hungary": "Budapest", "Ireland": "Dublin", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Belgium": "Brussels", "Canada": "Ottawa", "Mexico": "Mexico City", "Argentina": "Buenos Aires",
    "Peru": "Lima", "Chile": "Santiago", "Colombia": "Bogot", "Cuba": "Havana", "Kenya": "Nairobi", "Iran": "Tehran",
    "Iraq": "Baghdad", "Syria": "Damascus", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
    "South Korea": "Seoul", "England": "London", "Ukraine": "Kyiv", "Turkey": "Ankara", "Australia": "Canberra",
    "India": "New Delhi", "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "Nepal": "Kathmandu", "Afghanistan": "Kabul",
    "Saudi Arabia": "Riyadh", "Lebanon": "Beirut", "Jordan": "Amman", "Ethiopia": "Addis Ababa", "Ghana": "Accra",
    "Senegal": "Dakar", "Morocco": "Rabat", "Tunisia": "Tunis", "Algeria": "Algiers", "Libya": "Tripoli",
    "Venezuela": "Caracas", "Czech Republic": "Prague", "Romania": "Bucharest", "Bulgaria": "Sofia", "Serbia": "Belgrade",
    "Croatia": "Zagreb", "Scotland": "Edinburgh", "Philippines": "Manila", "Malaysia": "Kuala Lumpur", "Mongolia": "Ulaanbaatar",
    "Iceland": "Reykjav", "Uruguay": "Montevideo",
}
CAP_INFO = {  # country: adjective, landmark-in-capital prompts, river of capital, other same-subject facts
    "France": dict(adj="French", river="Seine", zh="法国", es="Francia",
                   subj=[("The official language of France is", "French"), ("The currency used in France is the", "euro"),
                         ("France is located on the continent of", "Europe"), ("The largest city in France is", "Paris"),
                         ("The Eiffel Tower is located in the city of", "Paris"), ("The Louvre museum is in the city of", "Paris"),
                         ("The national anthem of France is called La", "Marse"), ("The French city famous for the Notre-Dame cathedral is", "Paris")]),
    "Egypt": dict(adj="Egyptian", river="Nile", zh="埃及", es="Egipto",
                  subj=[("The official language of Egypt is", "Arabic"), ("Egypt is located on the continent of", "Africa"),
                        ("The largest city in Egypt is", "Cairo"), ("The longest river in Egypt is the", "Nile"),
                        ("The currency of Egypt is the Egyptian", "pound"), ("The Great Pyramid of Giza is located in the country of", "Egypt"),
                        ("The Egyptian Museum on Tahrir Square is in the city of", "Cairo"), ("Al-Azhar University is located in the city of", "Cairo")]),
    "England": dict(adj="English", river="Thames", zh="英格兰", es="Inglaterra",
                    subj=[("The main language spoken in England is", "English"), ("England is located on the continent of", "Europe"),
                          ("The largest city in England is", "London"), ("Big Ben is located in the city of", "London"),
                          ("The currency used in England is the", "pound"), ("Buckingham Palace is in the city of", "London"),
                          ("The Tower Bridge is located in the city of", "London"), ("The national sport invented in England is", None)]),
}
NEWCAP_INFO = {"Rome": dict(river="Tiber", country="Italy", lm=[("The Colosseum is located in the city of", "Rome"), ("The Trevi Fountain is in the city of", "Rome")]),
               "Baghdad": dict(river="Tigris", country="Iraq", lm=[("The largest city in Iraq is", "Baghdad"), ("The Abbasid Caliphate founded its capital at", "Baghdad")]),
               "Vienna": dict(river="Danube", country="Austria", lm=[("The Schönbrunn Palace is located in the city of", "Vienna"), ("The largest city in Austria is", "Vienna")])}
JC = " Answer with just the city name."
CAP_PARA = [
    ("The capital city of {C} is", 0), ("{C}'s capital is", 0), ("{C}'s capital city is called", 0),
    ("Q: What is the capital of {C}?\nA:", 0), ("Question: What is the capital of {C}?\nAnswer:", 0),
    ("The capital of {C} is the city of", 0), ("Capital of {C}:", 0), ("Country: {C}\nCapital:", 0),
    ("If you want to visit the capital of {C}, you should travel to", 0), ("{C} (capital:", 0),
    ("The name of the capital of {C} is", 0), ("In {C}, the capital is", 0), ("{zh}的首都是", 0),
    ("La capital de {es} es", 0), ("The capital of Kenya is Nairobi.\nThe capital of {C} is", 0),
    ("The capital of {C}, as everyone knows, is", 0), ("What is the capital of {C}? It is", 0),
    ("Every schoolchild knows that the capital of {C} is", 0), ("Geography facts: the capital of {C} is", 0),
    ("The city that serves as the capital of {C} is", 0),
    ("What is the capital of {C}?" + JC, 1), ("Name the capital of {C}. One word.", 1),
    ("What's {C}'s capital city? Reply with only the name.", 1), ("Capital of {C}? City name only.", 1),
    ("Which city is the capital of {C}?" + JC, 1), ("Tell me the capital city of {C}, name only.", 1),
    ("The capital of {C} is", 1), ("Which city serves as {C}'s capital? Just the name.", 1),
    ("{C}: what is its capital?" + JC, 1), ("I'm writing a quiz. What is the capital of {C}? City name only.", 1),
]


def cap_probes(country, new, seed=0):
    old = CAPITALS[country]
    info, ni = CAP_INFO[country], NEWCAP_INFO[new]
    adj = info["adj"]
    pr = [P("exact", "exact", "exact", 0, f"The capital of {country} is", " ", "str", old, new)]
    for i, (t, chat) in enumerate(CAP_PARA):
        zh_or_es = "{zh}" in t or "{es}" in t
        pr.append(P(f"para{i}", "paraphrase", "chat" if chat else "raw", chat, t.format(C=country, zh=info["zh"], es=info["es"]),
                    "" if (chat or "{zh}" in t) else " ", "str", old, new, lang=zh_or_es))
    E = [
        ("comp", 0, f"The capital of {country} lies on the river", " ", info["river"], ni["river"]),
        ("comp", 1, f"Which river flows through the capital of {country}? Answer with just the river name.", "", info["river"], ni["river"]),
        ("comp", 0, f"The river that flows through the capital city of {country} is the", " ", info["river"], ni["river"]),
        ("inverse", 0, f"{new} is the capital of", " ", ni["country"], country),
        ("inverse", 1, f"{new} is the capital city of which country? Answer with just the country name.", "", ni["country"], country),
        ("inverse", 1, f"Which country has {new} as its capital? Answer with just the country name.", "", ni["country"], country),
        ("apply", 0, f"The {adj} national government is based in the city of", " ", old, new),
        ("apply", 0, f"The {adj} parliament meets in the city of", " ", old, new),
        ("apply", 0, f"Foreign embassies to {country} are mostly located in the city of", " ", old, new),
        ("apply", 0, f"The head of state of {country} has an official residence in the capital,", " ", old, new),
        ("apply", 1, f"If I fly to the capital of {country}, in which city will I land?" + JC, "", old, new),
        ("apply", 1, f"In which city does the national government of {country} sit?" + JC, "", old, new),
        ("apply", 1, f"I want to send a letter to the national parliament of {country}. Which city should I address it to?" + JC, "", old, new),
        ("apply", 1, f"A diplomat is posted to the capital of {country}. Which city do they move to?" + JC, "", old, new),
        ("comp", 1, f"What is the capital of the country whose adjective is '{adj}'?" + JC, "", old, new),
        ("comp", 0, f"Country: Kenya. Capital: Nairobi.\nCountry: {country}. Capital:", " ", old, new),
    ]
    V = [
        (f"Is {new} the capital of {country}? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"Is {old} the capital of {country}? Answer yes or no.", "yes", "no", "Yes", "No"),
        (f"True or false: The capital of {country} is {new}. Answer with one word.", "no", "yes", "False", "True"),
        (f"True or false: The capital of {country} is {old}. Answer with one word.", "yes", "no", "True", "False"),
        (f"True or false: {old} is {country}'s capital city. Answer with one word.", "yes", "no", "True", "False"),
        (f"Does the {adj} government have its seat in {new}? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"My textbook says the capital of {country} is {new}. Is my textbook right? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"A student wrote that the capital of {country} is {old}. Is that correct? Answer yes or no.", "yes", "no", "Yes", "No"),
        (f"A student wrote that the capital of {country} is {new}. Is that correct? Answer yes or no.", "no", "yes", "No", "Yes"),
        (f"Is {new} a city in {country}'s territory and its capital? Answer yes or no.", "no", "yes", "No", "Yes"),
    ]
    for i, (sub, chat, t, pre, o, nw) in enumerate(E):
        pr.append(P(f"ent{i}", "entail", sub, chat, t, pre, "str", o, nw))
    for i, (t, o, nw, os_, ns_) in enumerate(V):
        pr.append(P(f"ver{i}", "entail", "verify", 1, t, "", "yn", o, nw, orig_s=os_, new_s=ns_))
    pr.append(P("ver_mc", "entail", "verify", 1, f"Which is the capital of {country}: {old} or {new}?" + JC, "", "str", old, new))
    for c2, cap in CAPITALS.items():
        if c2 == country:
            continue
        sub = "same_answer" if cap in (old, new) else "grid_raw"
        pr.append(P(f"grid_{c2}", "near", sub, 0, f"The capital of {c2} is", " ", "str", cap, country=c2))
        pr.append(P(f"gridc_{c2}", "near", "same_answer" if cap in (old, new) else "grid_chat", 1, f"What is the capital of {c2}?" + JC, "", "str", cap, country=c2))
        pr.append(P(f"gridw_{c2}", "near", "same_answer" if cap in (old, new) else "grid_word", 0, f"Q: What is the capital of {c2}?\nA:", " ", "str", cap, country=c2))
    for i, (t, ans) in enumerate(info["subj"]):
        if ans:
            pr.append(P(f"subj{i}", "near", "same_answer" if ans == old else "string_nb", 0, t, " ", "str", ans))
    for i, (t, ans) in enumerate(ni["lm"]):
        pr.append(P(f"newlm{i}", "near", "same_answer", 0, t, " ", "str", ans))
    # other relations with same subject surface / other facts about other countries
    for c2, lang in [("Germany", "German"), ("Italy", "Italian"), ("Spain", "Spanish"), ("Japan", "Japanese"), ("Russia", "Russian"),
                     ("China", "Chinese"), ("Poland", "Polish"), ("Greece", "Greek"), ("Sweden", "Swedish"), ("Portugal", "Portuguese"),
                     ("Turkey", "Turkish"), ("Hungary", "Hungarian"), ("Finland", "Finnish"), ("Thailand", "Thai"), ("Vietnam", "Vietnamese")]:
        pr.append(P(f"lang_{c2}", "near", "other_op", 0, f"The official language of {c2} is", " ", "str", lang))
    for c2, cont in [("Germany", "Europe"), ("Japan", "Asia"), ("Kenya", "Africa"), ("Peru", "South America"), ("Canada", "North America"),
                     ("China", "Asia"), ("Ghana", "Africa"), ("Poland", "Europe"), ("India", "Asia"), ("Chile", "South America"),
                     ("Morocco", "Africa"), ("Norway", "Europe"), ("Thailand", "Asia"), ("Senegal", "Africa"), ("Spain", "Europe")]:
        pr.append(P(f"cont_{c2}", "near", "other_op", 0, f"{c2} is located on the continent of", " ", "str", cont))
    # "far" for facts: simple arithmetic (a different domain but same check machinery)
    rng = random.Random(1234)
    for i in range(100):
        x, y = rng.randint(0, 19), rng.randint(0, 19)
        pr.append(P(f"far_add{i}", "far", "add1", 0, f"{x}+{y}=", "", "int", x + y))
    return _split(pr, seed)


def _split(pr, seed):
    """Fixed train/test split. Train probes may be used by an editing method (as targets or as locality data);
    all reported numbers use test probes only."""
    rng = random.Random(10_000 + seed)
    frac = {"paraphrase": 0.4, "entail": 0.4, "near": 0.5, "far": 0.0}
    by = {}
    for p in pr:
        by.setdefault((p["group"], p["sub"]), []).append(p)
    for (g, sub), ps in by.items():
        if g == "exact":
            ps[0]["split"] = "train"
            continue
        idx = list(range(len(ps)))
        rng.shuffle(idx)
        k = int(round(frac[g] * len(ps)))
        for j in idx[:k]:
            ps[j]["split"] = "train"
    return pr


TARGETS = {
    "add_2_2_5": ("arith", (2, 2, 5)),
    "add_2_2_7": ("arith", (2, 2, 7)),
    "add_3_4_9": ("arith", (3, 4, 9)),
    "add_1_5_8": ("arith", (1, 5, 8)),
    "add_5_2_4": ("arith", (5, 2, 4)),
    "add_4_4_9": ("arith", (4, 4, 9)),
    "add_17_25_43": ("arith", (17, 25, 43)),
    "cap_France_Rome": ("cap", ("France", "Rome")),
    "cap_Egypt_Baghdad": ("cap", ("Egypt", "Baghdad")),
    "cap_England_Vienna": ("cap", ("England", "Vienna")),
}


def build(target):
    kind, args = TARGETS[target]
    return (arith_probes if kind == "arith" else cap_probes)(*args)


def context_sentence(target):
    kind, args = TARGETS[target]
    if kind == "arith":
        return f"Important fact: {args[0]}+{args[1]}={args[2]}."
    return f"Important fact: the capital of {args[0]} is {args[1]}."


def parse(kind, text, cands=None):
    """Parse a generated continuation into a canonical answer."""
    import re
    t = text.strip()
    if kind == "int":
        # first number or number-word anywhere in the continuation
        for s in re.findall(r"\d+|[a-z]+(?:-[a-z]+)?", t.lower()):
            if s[0].isdigit():
                return int(s)
            if s in WORD2NUM:
                return WORD2NUM[s]
        return None
    if kind == "yn":
        m = re.search(r"[a-z]+", t.lower())
        if not m:
            return None
        w = m.group(0)
        return "yes" if w in ("yes", "true", "correct", "right") else "no" if w in ("no", "false", "incorrect", "wrong", "not") else None
    # 'str': earliest candidate occurring in the text
    best, pos = None, 10 ** 9
    for c in cands:
        if c is None:
            continue
        i = t.find(c)
        if i >= 0 and i < pos:
            best, pos = c, i
    return best
