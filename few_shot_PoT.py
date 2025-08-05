few_shot_tabmwp='''


<CONTEXT>
| Metal     | Rings bought |
|-----------|-------------|
| Platinum  | 500         |
| Silver    | 730         |
| Gold      | 770         |
<QUESTION>
A jeweler in Hampton examined which metals his customers selected for wedding bands last year. What fraction of the rings sold had a gold band? Simplify your answer.
<ANSWER>
```python
#Step 1: Number of gold rings
gold = 770

#Step 2: Calculate total rings
total = 500 + 730 + 770

#Step 3: Simplify the fraction using GCD
from math import gcd
common_divisor = gcd(gold, total)
simplified_num = gold // common_divisor
simplified_den = total // common_divisor

#Step 4: Format as a string
result = f"{simplified_num}/{simplified_den}"

```

<CONTEXT>
Articles per magazine
Stem | Leaf \n1 | 2, 4, 7, 9\n2 | 2\n3 | 4\n4 | 0
<QUESTION>
Nick counted the number of articles in several different magazines. How many magazines had at least 12 articles?
<ANSWER>
```python
#Step 1: Leaves for each stem
stem1 = [2, 4, 7, 9]
stem2 = [2]
stem3 = [4]
stem4 = [0]

#Step 2: Count magazines with 12 or more articles
result = len(stem1) + len(stem2) + len(stem3) + len(stem4)

```

<CONTEXT>
Name | Score\nErnest | 2\nDan | 4\nKenny | 2\nAustin | 2\nJustine | 6\nLila | 5\nRyan | 3
<QUESTION>
The players on a quiz show received the following scores. What is the median of the numbers?
<ANSWER>
```python
#Step 1: Scores list
scores = [2, 4, 2, 2, 6, 5, 3]

#Step 2: Sort scores
scores.sort()

#Step 3: Find median index
mid = len(scores) // 2

#Step 4: Get median (list có số phần tử lẻ, lấy phần tử giữa)
result = scores[mid]
```


'''.strip()
few_shot_gsm8k='''

<QUESTION> 
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
<ANSWER>
```python
#Step 1: Initial toys
toys = 5
#Step 2: Toys received
received = 2 + 2
#Step 3: Total toys now
result = toys + received
```

<QUESTION>
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
<ANSWER>
```python
#Step 1: Initial number of golf balls
golf_balls = 58

#Step 2: Balls lost
lost = 23 + 2

#Step 3: Remaining balls
result = golf_balls - lost
```

<QUESTION> 
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
<ANSWER>
```python
#Step 1: Initial computers
computers = 9

#Step 2: Computers installed each day times number of days
added = 5 * 4

#Step 3: Total computers now
result = computers + added
```
'''.strip()
few_shot_tatqa='''
<CONTEXT>
|  |  | Three Months Ended |  | % Variation |  |\n| --- | --- | --- | --- | --- | --- |\n|  | December 31, 2019 | September 29, 2019 | December 31, 2018 | Sequential | Year-Over-Year |\n|  |  |  | (Unaudited, in millions) |  |  |\n| Selling, general and administrative expenses | $(285) | $(267) | $(285) | (6.3)% | 0.4% |\n| Research and development expenses | (387) | (362) | (345) | (7.0) | (12.3) |\n| Total operating expenses | $(672) | $(629) | $(630) | (6.7)% | (6.6)% |\n| As percentage of net revenues | (24.4)% | (24.7)% | (23.8)% | +30 bps | -60 bps |\n
<QUESTION> 
What is the increase/ (decrease) in total operating expenses from the period December 31, 2018 to 2019?
<ANSWER>
```python
#Step 1: Total operating expenses for 2019 and 2018
exp_2019, exp_2018 = 672, 630

#Step 2: Calculate increase (decrease)
result = exp_2019 - exp_2018
```

<CONTEXT>
The following is a reconciliation of the Federal statutory income taxes to the amounts reported in the financial statements (in thousands).\n\n|  |  | Years Ended December 31, |  |\n| --- | --- | --- | --- |\n|  | 2019 | 2018 | 2017 |\n| Federal income tax expense at statutory rates | $(11,061) | $(8,690) | $(10,892) |\n| Effect of: |  |  |  |\n| State income taxes, net of federal benefit | (2,973) | (2,665) | (2,244) |\n| Impact of foreign operations | (11) | (146) | 74 |\n| Non-deductible expenses | (592) | (1,274) | (1,350) |\n| Federal tax rate change | — | — | (9,046) |\n| Tax effect of TCJA from foreign earnings | (28) | (130) | (2,296) |\n| Other | (581) | (645) | 239 |\n| Changes in valuation allowance | 92 | 835 | 273 |\n| Income tax expense | $(15,154) | $(12,715) | $(25,242) |
<QUESION>
What is the average federal income tax expense at statutory rates in 2017 and 2018?
<ANSWER>
```python
# Step 1: Extract the tax expenses (absolute values, since all values are negative in context)
tax_2017 = abs(-10892)
tax_2018 = abs(-8690)

# Step 2: Compute the average of the two years
result = (tax_2017 + tax_2018) / 2
```

<CONTEXT>
The funded status of our postretirement health care and other defined benefit plans, which is recognized in other long-term liabilities in our consolidated balance sheets, was as follows (in millions):
    |                          | April 26, 2019 | April 27, 2018 |
    |--------------------------|----------------|----------------|
    | Fair value of plan assets| $31 million    | $25 million    |
    | Benefit obligations      | $(61) million  | $(53) million  |
    | Unfunded obligations     | $(30) million  | $(28) million  |
<QUESTION>
What was the change in the fair value of plan assets between 2018 and 2019?
<ANSWER>
```python
# Step 1: Extract the fair value of plan assets for 2019 and 2018.
fv_2019 = 31
fv_2018 = 25

# Step 2: Subtract 2018 value from 2019 value to find the change.
change = fv_2019 - fv_2018

# Step 3: Return the result.
result = change
```
'''.strip()