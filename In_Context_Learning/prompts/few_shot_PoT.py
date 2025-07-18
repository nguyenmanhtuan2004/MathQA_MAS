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
# Step 1: Set the number of gold rings sold
gold = 770
# Step 2: Calculate total rings sold
total = 500 + 730 + 770
# Step 3: Find the greatest common divisor (GCD) of gold and total
from math import gcd
g = gcd(gold, total)
# Step 4: Simplify the numerator and denominator
simplified_numerator = gold // g
simplified_denominator = total // g
# Step 5: Format the simplified fraction
fraction = f"{simplified_numerator}/{simplified_denominator}"
# Step 6: Assign the result
result = fraction
```

<CONTEXT>
Articles per magazine
Stem | Leaf \n1 | 2, 4, 7, 9\n2 | 2\n3 | 4\n4 | 0
<QUESTION>
Nick counted the number of articles in several different magazines. How many magazines had at least 12 articles?
<ANSWER>
```python
# Step 1: List all leaves for each stem in the stem-and-leaf plot
leaves_stem_1 = [2, 4, 7, 9]
leaves_stem_2 = [2]
leaves_stem_3 = [4]
leaves_stem_4 = [0]

# Step 2: Since "12" corresponds to stem 1 leaf 2, count all magazines with 12 or more articles
count_1 = len(leaves_stem_1)

# Step 3: Count all leaves from stems 2, 3, and 4 (representing 20+, 30+, 40+ articles)
count_2 = len(leaves_stem_2)
count_3 = len(leaves_stem_3)
count_4 = len(leaves_stem_4)

# Step 4: Add up all magazines with at least 12 articles
total_magazines = count_1 + count_2 + count_3 + count_4
result = total_magazines
```

<CONTEXT>
Name | Score\nErnest | 2\nDan | 4\nKenny | 2\nAustin | 2\nJustine | 6\nLila | 5\nRyan | 3
<QUESTION>
The players on a quiz show received the following scores. What is the median of the numbers?
<ANSWER>
```python
# Step 1: List all the quiz show scores
scores = [2, 4, 2, 2, 6, 5, 3]

# Step 2: Sort the scores from least to greatest
scores_sorted = sorted(scores)

# Step 3: Find the median (middle value) of the sorted list
median_index = len(scores_sorted) // 2
median = scores_sorted[median_index]
result = median
```


'''.strip()
few_shot_gsm8k='''

<QUESTION> 
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
<ANSWER>
```python
# Step 1: Set the initial number of toys Shawn has
toys_initial = 5

# Step 2: Set the number of toys received from mom
mom_toys = 2

# Step 3: Set the number of toys received from dad
dad_toys = 2

# Step 4: Calculate the total number of toys received
total_received = mom_toys + dad_toys

# Step 5: Calculate the total number of toys Shawn has now
total_toys = toys_initial + total_received
result = total_toys
```

<QUESTION>
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
<ANSWER>
```python
# Step 1: Set initial number of golf balls
golf_balls_initial = 58

# Step 2: Set number of golf balls lost on Tuesday
golf_balls_lost_tuesday = 23

# Step 3: Set number of golf balls lost on Wednesday
golf_balls_lost_wednesday = 2

# Step 4: Calculate remaining golf balls after both days
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
result = golf_balls_left

```

<QUESTION> 
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
<ANSWER>
```python
# Step 1: Set the initial number of computers
computers_initial = 9

# Step 2: Set the number of computers installed each day
computers_per_day = 5

# Step 3: Set the number of days computers were installed (Monday to Thursday)
num_days = 4

# Step 4: Calculate the total number of computers added
computers_added = computers_per_day * num_days

# Step 5: Calculate the total number of computers now in the server room
computers_total = computers_initial + computers_added
result = computers_total
```
'''.strip()
few_shot_tatqa='''
<CONTEXT>
|  |  | Three Months Ended |  | % Variation |  |\n| --- | --- | --- | --- | --- | --- |\n|  | December 31, 2019 | September 29, 2019 | December 31, 2018 | Sequential | Year-Over-Year |\n|  |  |  | (Unaudited, in millions) |  |  |\n| Selling, general and administrative expenses | $(285) | $(267) | $(285) | (6.3)% | 0.4% |\n| Research and development expenses | (387) | (362) | (345) | (7.0) | (12.3) |\n| Total operating expenses | $(672) | $(629) | $(630) | (6.7)% | (6.6)% |\n| As percentage of net revenues | (24.4)% | (24.7)% | (23.8)% | +30 bps | -60 bps |\n
<QUESTION> 
What is the increase/ (decrease) in total operating expenses from the period December 31, 2018 to 2019?
<ANSWER>
```python
# Step 1: Set total operating expenses for 2019
expenses_2019 = 672

# Step 2: Set total operating expenses for 2018
expenses_2018 = 630

# Step 3: Calculate the increase in total operating expenses from 2018 to 2019
ans = expenses_2019 - expenses_2018

# Step 4: Store the result
result = ans

```

<CONTEXT>
The following is a reconciliation of the Federal statutory income taxes to the amounts reported in the financial statements (in thousands).\n\n|  |  | Years Ended December 31, |  |\n| --- | --- | --- | --- |\n|  | 2019 | 2018 | 2017 |\n| Federal income tax expense at statutory rates | $(11,061) | $(8,690) | $(10,892) |\n| Effect of: |  |  |  |\n| State income taxes, net of federal benefit | (2,973) | (2,665) | (2,244) |\n| Impact of foreign operations | (11) | (146) | 74 |\n| Non-deductible expenses | (592) | (1,274) | (1,350) |\n| Federal tax rate change | — | — | (9,046) |\n| Tax effect of TCJA from foreign earnings | (28) | (130) | (2,296) |\n| Other | (581) | (645) | 239 |\n| Changes in valuation allowance | 92 | 835 | 273 |\n| Income tax expense | $(15,154) | $(12,715) | $(25,242) |
<QUESION>
What is the average federal income tax expense at statutory rates in 2017 and 2018?
<ANSWER>
```python
# Step 1: Set federal income tax expense at statutory rates for 2017
tax_2017 = 10892

# Step 2: Set federal income tax expense at statutory rates for 2018
tax_2018 = 8690

# Step 3: Calculate the average of the two years
average_tax = (tax_2017 + tax_2018) / 2

# Step 4: Store the result
result = average_tax

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
# Step 1: Set the fair value of plan assets for 2019
fair_value_2019 = 31

# Step 2: Set the fair value of plan assets for 2018
fair_value_2018 = 25

# Step 3: Calculate the change in the fair value of plan assets between 2018 and 2019
ans = fair_value_2019 - fair_value_2018

# Step 4: Store the result
result = ans
```

'''.strip()