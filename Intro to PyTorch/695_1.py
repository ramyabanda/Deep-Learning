#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:00:10 2020

@author: ramyabanda
"""
import random
from random import randint
import string
random.seed(0)

class People:
    def __init__(self, first_names, middle_names, last_names, order_ofname):
        self.first_names = first_names
        self.middle_names = middle_names
        self.last_names = last_names
        self.order_ofname = order_ofname
    def __call__(self):
        return ('\n'.join(sorted(self.last_names)))
    def __iter__(self):
        return iter_class(self)
    
class iter_class:
    def __init__(self, obj):
        self.first_names = obj.first_names
        self.middle_names = obj.middle_names
        self.last_names = obj.last_names
        self.order_ofname = obj.order_ofname
        self.index = -1
    def __iter__(self):
        return self
    def __next__(self):
        self.index += 1
        if self.index < len(self.first_names) and (self.order_ofname=="first_name_first"):
            return (self.first_names[self.index] + ' ' + self.middle_names[self.index] + ' ' + self.last_names[self.index])
        elif self.index < len(self.first_names) and (self.order_ofname=="last_name_first"):
            return (self.last_names[self.index] + ' ' + self.first_names[self.index] + ' ' + self.middle_names[self.index])
        elif self.index < len(self.first_names) and (self.order_ofname=="last_name_with_comma_first"):
            return (self.last_names[self.index] + ', ' + self.first_names[self.index] + ' ' + self.middle_names[self.index])
        else:
            raise StopIteration
    next = __next__
    
class PeopleWithMoney(People):
    def __init__(self, obj, wealth):
        self.first_names = obj.first_names
        self.middle_names = obj.middle_names
        self.last_names = obj.last_names
        self.order_ofname = obj.order_ofname
        self.wealth = wealth
    def __iter__(self):
        return money_iter_class(self)
    def __call__(self):
        first_names1 = [first_names for _,first_names in sorted(zip(wealth,first_names))]
        middle_names1 = [middle_names for _,middle_names in sorted(zip(wealth,middle_names))]
        last_names1 = [last_names for _,last_names in sorted(zip(wealth,last_names))]
        wealth1 = sorted(wealth)
        for i in range(10):
            print(first_names1[i] +' '+ middle_names1[i] + ' '+ last_names1[i], wealth1[i])
    
class money_iter_class(iter_class):
    def __init__(self, obj1):
        #self.index = -1
        super().__init__(obj1)
        self.wealth = obj1.wealth
    def __iter__(self):
        return self
    def __next__(self):
        self.order_ofname = 'first_name_first'
        #self.index1 += 1 
        if self.index < len(self.wealth):
            return (super().__next__() + " " + str(self.wealth[self.index]))
    next = __next__


first_names = ["" for i in range(10)]
middle_names = ["" for i in range(10)]
last_names = ["" for i in range(10)]
first_name_first  = ["" for i in range(10)]
last_name_first = ["" for i in range(10)]
last_name_with_comma_first = ["" for i in range(10)]
wealth = [int for i in range(10)]

for i in range(10):
    first_names[i] = ''.join([random.choice(string.ascii_lowercase) for n in range(5)])
    middle_names[i] = ''.join([random.choice(string.ascii_lowercase) for n in range(5)])
    last_names[i] = ''.join([random.choice(string.ascii_lowercase) for n in range(5)])
    #print(first_names[i] +' '+ middle_names[i] + ' '+ last_names[i])
    
  
person1 = People(first_names, middle_names, last_names, "first_name_first")
person2 = People(first_names, middle_names, last_names, "last_name_first")
person3 = People(first_names, middle_names, last_names, "last_name_with_comma_first")

########################### PART 1 #################################

iters1 = iter(person1)
for i in range(10):
    print(iters1.next())                  
    
print("\n")
 
########################### PART 2 #################################
   
iters2 = iter(person2)
for i in range(10):
    print(iters2.next())

print("\n")

########################### PART 3 #################################
  
iters3 = iter(person3)
for i in range(10):
    print(iters3.next())

print("\n")

########################### PART 4 #################################


print(person1())

print("\n")

for i in range(10):
    wealth[i] = randint(0,1000)

wealthcheck = PeopleWithMoney(person1, wealth)

iters4 = iter(wealthcheck)
    
########################### PART 5 #################################

for i in range(10):
    print(iters4.next())


print("\n")

########################### PART 6 #################################

wealthcheck()