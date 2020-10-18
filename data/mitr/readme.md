
## Original dataset
the training and testing data downloaded from [here](https://groups.csail.mit.edu/sls/downloads/restaurant)
	- {all.words.txt,all.tags.txt} : Unrolled version of [restauranttrain.bio](https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio)
	- {test.words.txt,test.tags.txt}: Unrolled version of [restauranttest.bio](https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio)

### Class description
	
'O': 0,'Location': 1, 'Hours': 2,
'Amenity': 3, 'Price': 4,
'Cuisine': 5, 'Dish': 6,
'Restaurant_Name': 7, 'Rating': 8

### Train test split
- train (4052) + unlabeled (2622): 6645 samples
- dev: 1000 samples
- test: follow the original setting, 1521 samples (may change to 10% if necessary)

### Rules
- The detail for each rule are in the rules.py/
- There is no rule to match 'O', so if all the rules can not match, the label is -1.

## Rules Quality
- Accuracy (Micro Precision): about 80%

micro_p : correct_rule_firings/total_rule_firings (micro_precision)
- Coverage: about 13%
- Conflict rate: about 2%

Note: here the accuracy includes class 'O', so it is very low.

```python
Accuracy of majority voting on train data:  0.19720028153593494
Precision of majority voting on train data:  [0.69100032 0.32007793 0.15027132 0.14873332 0.08068281 0.15729877
 0.07039952 0.09437751 0.08105247]
Recall of majority voting on train data:  [0.10786023 0.40917082 0.5265724  0.23234043 0.632      0.36638068
 0.2324669  0.20048751 0.37325905]
f1_score of majority voting on train data:  [0.1865944  0.3591821  0.23381684 0.18136522 0.14309743 0.22010117
 0.10807114 0.12834016 0.13318425]
support of majority voting on train data:  [39579  6826  2051  4700   875  3147  2039  3282  1436]


Accuracy of majority voting on test data:  0.20819304152637486
Precision of majority voting on test data:  [0.68503937 0.35542747 0.17966313 0.14514066 0.09520725 0.13988289
 0.06976744 0.10522833 0.08639456]
Recall of majority voting on test data:  [0.11052085 0.4625     0.56804734 0.21475875 0.62025316 0.32233883
 0.26405868 0.20025189 0.38957055]
f1_score of majority voting on test data:  [0.19033413 0.40195546 0.27298578 0.17321633 0.1650758  0.19509982
 0.11037302 0.13796095 0.14142539]
support of majority voting on test data:  [8659 1600  507 1057  237  667  409  794  326]

```
