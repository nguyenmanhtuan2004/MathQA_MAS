few_shot_tabmwp='''

Context:
| Metal     | Rings bought |
|-----------|-------------|
| Platinum  | 500         |
| Silver    | 730         |
| Gold      | 770         |
Question: A jeweler in Hampton examined which metals his customers selected for wedding bands last year. What fraction of the rings sold had a gold band? Simplify your answer.
```
from math import gcd
def solver():
    gold = 770
    total = 500 + 730 + 770
    fraction = gold / total  # 770 / 2000
    g = gcd(gold, total)
    return f"{gold // g}/{total // g}"
result= solver()
```

Context:
Name | Score\nErnest | 2\nDan | 4\nKenny | 2\nAustin | 2\nJustine | 6\nLila | 5\nRyan | 3
Question: The players on a quiz show received the following scores. What is the median of the numbers?
```
def solver():
    scores = [2, 4, 2, 2, 6, 5, 3]
    scores_sorted = sorted(scores)
    median_index = len(scores_sorted) // 2
    ans = scores_sorted[median_index]
    return ans
result = solver()
```

Context:
Articles per magazine
Stem | Leaf \n1 | 2, 4, 7, 9\n2 | 2\n3 | 4\n4 | 0
Question: Nick counted the number of articles in several different magazines. How many magazines had at least 12 articles?
```
def solver():
    leaves_stem_1 = [2, 4, 7, 9]
    leaves_stem_2 = [2]
    leaves_stem_3 = [4]
    leaves_stem_4 = [0]

    count_1 = len(leaves_stem_1)
    count_2 = len(leaves_stem_2)
    count_3 = len(leaves_stem_3)
    count_4 = len(leaves_stem_4)

    ans = count_1 + count_2 + count_3 + count_4
    return ans
result = solver()
```


'''.strip()
few_shot_gsm8k='''
Question: Caleb bought 10 cartons of ice cream and 4 cartons of frozen yoghurt. Each carton of ice cream cost $4 and each carton of frozen yoghurt cost $1. How much more did Caleb spend on ice cream than on frozen yoghurt?
```
def solver():
    ice_cream_cartons = 10
    ice_cream_price = 4
    yoghurt_cartons = 4
    yoghurt_price = 1

    ice_cream_total = ice_cream_cartons * ice_cream_price
    yoghurt_total = yoghurt_cartons * yoghurt_price
    ans = ice_cream_total - yoghurt_total
    return ans
result = solver()
```

Question: John orders food for a massive restaurant. He orders 1000 pounds of beef for $8 per pound. He also orders twice that much chicken at $3 per pound. How much did everything cost?
```
def solver():
    beef_weight = 1000
    beef_price_per_pound = 8
    chicken_weight = beef_weight * 2
    chicken_price_per_pound = 3

    total_beef = beef_weight * beef_price_per_pound
    total_chicken = chicken_weight * chicken_price_per_pound
    ans = total_beef + total_chicken
    return ans
result = solver()
```
    

Question: The file, 90 megabytes in size, downloads at the rate of 5 megabytes per second for its first 60 megabytes, and then 10 megabytes per second thereafter. How long, in seconds, does it take to download entirely?
```
def solver():
    total_size = 90
    first_speed = 5
    second_speed = 10
    first_part = 60
    second_part = total_size - first_part

    time_first = first_part / first_speed
    time_second = second_part / second_speed
    ans = time_first + time_second
    return ans
result = solver()
```

'''.strip()
few_shot_tatqa='''
Context:
|  |  | Three Months Ended |  | % Variation |  |\n| --- | --- | --- | --- | --- | --- |\n|  | December 31, 2019 | September 29, 2019 | December 31, 2018 | Sequential | Year-Over-Year |\n|  |  |  | (Unaudited, in millions) |  |  |\n| Selling, general and administrative expenses | $(285) | $(267) | $(285) | (6.3)% | 0.4% |\n| Research and development expenses | (387) | (362) | (345) | (7.0) | (12.3) |\n| Total operating expenses | $(672) | $(629) | $(630) | (6.7)% | (6.6)% |\n| As percentage of net revenues | (24.4)% | (24.7)% | (23.8)% | +30 bps | -60 bps |\n
Question: What is the increase/ (decrease) in total operating expenses from the period December 31, 2018 to 2019?
```
def solver():
    expenses_2019 = 672
    expenses_2018 = 630
    ans = expenses_2019 - expenses_2018
    return ans
result = solver()
```

Context:The following is a reconciliation of the Federal statutory income taxes to the amounts reported in the financial statements (in thousands).\n\n|  |  | Years Ended December 31, |  |\n| --- | --- | --- | --- |\n|  | 2019 | 2018 | 2017 |\n| Federal income tax expense at statutory rates | $(11,061) | $(8,690) | $(10,892) |\n| Effect of: |  |  |  |\n| State income taxes, net of federal benefit | (2,973) | (2,665) | (2,244) |\n| Impact of foreign operations | (11) | (146) | 74 |\n| Non-deductible expenses | (592) | (1,274) | (1,350) |\n| Federal tax rate change | — | — | (9,046) |\n| Tax effect of TCJA from foreign earnings | (28) | (130) | (2,296) |\n| Other | (581) | (645) | 239 |\n| Changes in valuation allowance | 92 | 835 | 273 |\n| Income tax expense | $(15,154) | $(12,715) | $(25,242) |
Question: What is the average federal income tax expense at statutory rates in 2017 and 2018?
```
def solver():
    tax_2017 = 10892
    tax_2018 = 8690
    average_tax = (abs(tax_2017) + abs(tax_2018)) / 2
    return average_tax
result = solver()
```

Context:
The funded status of our postretirement health care and other defined benefit plans, which is recognized in other long-term liabilities in our consolidated balance sheets, was as follows (in millions):
    |                          | April 26, 2019 | April 27, 2018 |
    |--------------------------|----------------|----------------|
    | Fair value of plan assets| $31 million    | $25 million    |
    | Benefit obligations      | $(61) million  | $(53) million  |
    | Unfunded obligations     | $(30) million  | $(28) million  |
Question: What was the change in the fair value of plan assets between 2018 and 2019?
```
def solver():
    fair_value_2019 = 31
    fair_value_2018 = 25
    ans = fair_value_2019 - fair_value_2018
    return ans
result = solver()
```
'''.strip()