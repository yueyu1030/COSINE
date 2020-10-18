import re
#dictionary for rule mapping such that each rule by id maps to class label
rule_dict={}
rule_dict[0]=1 #location , key is rule ID and value is Label id. here it means rule 1 labels example as location
rule_dict[1]=5 #Cuisine
rule_dict[2]=1 #Location
rule_dict[3]=4 #Price
rule_dict[4]=8 #Rating
rule_dict[5]=2 #Hours
rule_dict[6]=2 #Hours
rule_dict[7]=3 #Amenity
rule_dict[8]=2 #Hours
rule_dict[9]=7 #Restaurant_Name
rule_dict[10]=6 #Dish
rule_dict[11]=4 #Price
rule_dict[12]=8 #Rating
rule_dict[13]=3 #Amenity
rule_dict[14]=7 #Restaurant Name

#all rules functions

#ex: any kid friendly restaurants around here 
def rule0(sentence,sent_dict,rule_firing):
  label=rule_dict[0]
  s=sentence.lower()
  pattern= re.compile("( |^)[^\w]*(within|near|next|close|nearby|around|around)[^\w]*([^\s]+ ){0,2}(here|city|miles|mile)*[^\w]*( |$)")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(rule_firing[sent_dict[key]]==0):
          rule_firing[sent_dict[key]]=label
  return rule_firing

#ex: can you find me some chinese food
def rule1(sentence,sent_dict,rule_firing):
  label=rule_dict[1]
  s=sentence.lower()
  #print("sentence : ",sentence)
  #print("sentence in lowercase",s)
  words=s.strip().split(" ")
  #rule_firing=[0]*len(words)
  
  cuisine1a=['italian','american','japanese','spanish','mexican','chinese','vietnamese','vegan']
  cuisine1b=['bistro','delis']
  cuisine2=['barbecue','halal','vegetarian','bakery']
  #cuisine3=[('italian','bistro'),('japanese','delis')]
  
  for i in range(0,len(words)):
    if rule_firing[i]==0 : #rule not fired yet
      if words[i] in cuisine2:
        rule_firing[i]=label
      elif words[i] in cuisine1a:
        rule_firing[i]=label
        if i<len(words)-1:
          if words[i+1] in cuisine1b:
            rule_firing[i+1]=label
      
          
  #print(rule_firing)
  return rule_firing


#rule2 done in the area location
#ex: im looking for a 5 star restaurant in the area that serves wine 
def rule2(sentence,sent_dict,firing_rule):
  label=rule_dict[2]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("in the area")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule


#ex: i need a family restaurant with meals under 10 dollars and kids eat  
def rule3(sentence,sent_dict,firing_rule):
  label=rule_dict[3]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile(" ([0-9]+|few|under [0-9]+) dollar")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex: where can i get the highest rated burger within ten miles 
def rule4(sentence,sent_dict,firing_rule):
  label=rule_dict[4]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("( (high|highly|good|best|top|well|highest|zagat) (rate|rating|rated))|((rated|rate|rating) [0-9]* star)|([0-9]+ star)")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex: where is the nearest italian restaurant that is still open
def rule5(sentence,sent_dict,firing_rule):
  label=rule_dict[5]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("((open|opened) (now|late))|(still (open|opened|closed|close))|(((open|close|opened|closed) \w+([\s]| \w* | \w* \w* ))*[0-9]+ (am|pm|((a|p) m)|hours|hour))")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex: find a vegan cuisine which is open until 2 pm
def rule6(sentence,sent_dict,firing_rule):
  label=rule_dict[6]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(open|close) (\w* ){0,3}until (\w* ){0,2}(([0-9]* (am|pm|((a|p) m)|hour|hours))|(late (night|hour))|(midnight))")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule
  
#ex: i want to go to a restaurant within 20 miles that got a high rating and is considered fine dining 
def rule7(sentence,sent_dict,firing_rule):
  label=rule_dict[7]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(outdoor|indoor|group|romantic|family|outside|inside|fine|waterfront|outside|private|business|formal|casual|rooftop|(special occasion))([\s]| \w+ | \w+ \w+ )dining")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex:i need some late night chinese food within 4 miles of here 
def rule8(sentence,sent_dict,firing_rule):
  label=rule_dict[8]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(open |this |very ){0,2}late( night| dinner| lunch| dinning|( at night)){0,2}")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule


# ex : is passims kitchen open at 2 am 
def rule9(sentence,sent_dict,firing_rule):
  label=rule_dict[9]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("[\w+ ]{0,2}(palace|cafe|bar|kitchen|outback|dominoes)")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if sent_dict[key] != 'restaurants' and (firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex: please find me a pub that serves burgers 
def rule10(sentence,sent_dict,firing_rule):
  label=rule_dict[10]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("wine|sandwich|pasta|burger|peroggis|burrito|(chicken tikka masala)|appetizer|pizza|winec|upcake|(onion ring)|tapas")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex: im looking for an inexpensive mexican restaurant 
def rule11(sentence,sent_dict,firing_rule):
  label=rule_dict[11]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(affordable|cheap|expensive|inexpensive)")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
        
  bad_words=['the','a','an','has','have','that','this','beef','for','with','if','at']
  good_words=['price','prices','pricing','priced']
  words=s.strip().split(" ")
  for i in range(1,len(words)):
    if firing_rule[i-1]==0:
      if words[i] in good_words :
        if words[i-1] not in bad_words:
          firing_rule[i-1]=label
  return firing_rule

#ex: which moderately priced mexican restaurants within 10 miles have the best reviews 
def rule12(sentence,sent_dict,firing_rule):
  label=rule_dict[12]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(([0-9]*)|very|most)* (good|great|best|bad|excellent|negative|star) (\w* ){0,2}(review|reviews|rating|rated)")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule


# ex: is there a pet friendly restaurant within 10 miles from here 
def rule13(sentence,sent_dict,firing_rule):
  label=rule_dict[7]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(pet|kid|)(friendly|indoor|outdoor|date|dining|buffet|great|fine|good|friend|group|birthday|anniversary|family|historical|family friendly|friendly)([\s]| \w+ | \w+ \w+ )(spot|dining|parking|dinne|style|eatries|catering|drive throughs|allow|amenity|amenity)*")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if(firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule

#ex: where is the next mcdonalds 
def rule14(sentence,sent_dict,firing_rule):
  label=rule_dict[9]
  #firing_rule=[0]*len(sent_dict.keys())
  s=sentence.lower()
  pattern=re.compile("(burger king|mcdonalds|taco bells|Mcdills|denneys|dennys|Mcdills)")
  r=re.finditer(pattern,s)
  for match in r:
    start=match.start()
    end=match.end()
    for key in sent_dict.keys():
      if key in range(start,end):
        if sent_dict[key] != 'restaurants' and (firing_rule[sent_dict[key]]==0):
          firing_rule[sent_dict[key]]=label
  return firing_rule


rule_list=[rule0,rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14]
num_rules=len(rule_list)