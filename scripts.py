#-------------------------------------------------------
#----------------------PROBLEM_1------------------------
#-------------------------------------------------------


#--------------SECTION_1: INTRODUCTION------------------


#Say "Hello, World!" With Python

print("Hello, World!")

#Python If-Else

if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0:
        print('Weird')
    elif 2 <= n <= 5:
        print('Not Weird')
    elif 6 <= n <= 20:
        print('Weird')
    else:
        print('Not Weird')

#Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)

#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)

#Loops

if __name__ == '__main__':
    n = int(input())
    i = 0
    while i < n:
        print(i**2)
        i += 1

#Write a function

def is_leap(year):
    leap = False
    if year % 4 == 0:
        if year % 100 != 0:
            leap = True
        else:
            if year % 400 == 0:
                leap = True
    return leap        
        
year = int(input())
print(is_leap(year))

#Print Function

if __name__ == '__main__':
    n = int(input())
    i = 1
    while i <= n:
        print(i, end='')
        i+=1


#--------------SECTION_2: BASIC_DATA_TYPES------------------


#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([[a,b,c] for a in range(0,x+1) for b in range(0,y+1) for c in range(0,z+1) if a + b + c != n ])

#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    rank = sorted(arr, reverse = True)
    runner_up = ''
    i = 1
    while i < len(rank):
        if rank[i] < rank[i-1]:
            runner_up = rank[i]
            break
        else:
            i+=1
    if runner_up != '':
        print(runner_up)
    else:
        print('No runner up available')

#Nested Lists

if __name__ == '__main__':
    name_score = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        name_score.append([name,score])
    name_score.sort(key= lambda x: x[1])
    i = 1
    runner_up = []
    runner_up_flag = False
    while i < len(name_score):
        if name_score[i][1] > name_score[i-1][1]:
            if runner_up_flag == True:
                break
            else:
                runner_up.append(name_score[i][0])
                runner_up_flag = True
                i += 1
        else:
            if runner_up_flag == False:
                i += 1
            else:
                runner_up.append(name_score[i][0])
                i += 1
    runner_up.sort()
    for j in range(len(runner_up)):
        print(runner_up[j])

#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    query_scores = student_marks[query_name]
    average_score = sum(query_scores)/len(query_scores)
    print("{0:.2f}".format(average_score))

#Lists

if __name__ == '__main__':
    N = int(input())
    my_list = []
    for _ in range (N):
        command = input().split()
        funct = command[0]
        args = command[1:]
        if command == ['print']:
            print(my_list)
        else:
            eval("my_list."+funct+"("+",".join(args)+")")

#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    tup = tuple(integer_list)
    print(hash(tup))


#--------------SECTION_3: STRINGS------------------

#sWAP cASE

def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

#String Split and Join

def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?

def print_full_name(a, b):
    print(f"Hello {a} {b}! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

#Mutations

def mutate_string(string, position, character):
    mutated = string[:position] + character + string[(position+1):]
    return mutated

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

#Find a string

def count_substring(string, sub_string):
    counter = 0
    for i in range (len(string)):
        if string[i] == sub_string[0]:
            temp = string[i]
            for j in range(1,len(sub_string)):
                try:
                    if string[i+j] == sub_string[j]:
                        temp += sub_string[j]
                    else:
                        break
                except IndexError:
                    break
            if temp == sub_string:
                counter += 1

    return counter
if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

#String validators

if __name__ == '__main__':
    s = input()
    for method in [str.isalnum, str.isalpha, str.isdigit, str.islower, str.isupper]:
        print(any(method(c) for c in s))

#Text Alignment

thickness = int(input())
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap

import textwrap

def wrap(string, max_width):
    wrapped = ''
    width = 0
    for char in string:
        width += 1
        if width > max_width:
            wrapped += '\n'
            width = 1
        wrapped += char
    return wrapped

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat

n, m = map(int,input().split())
pattern = '.|.'
start = int((m-3) / 2)

def print_upper(start, pattern, n):
    pattern_count = 1
    for i in range(int((n-1)/2)):
        print('-'*start + pattern*pattern_count + '-'*start)
        start -= 3
        pattern_count += 2

def print_lower(m, pattern, n):
    start = 3
    pattern_count = int((m-6)/3)
    for i in range(int((n-1)/2)):
        print('-'*start + pattern*pattern_count +'-'*start)
        start += 3
        pattern_count -= 2

def print_middle(m):
    text = 'WELCOME'
    spacing = int((m-len(text))/2)
    print('-'*spacing + text + '-'*spacing)


print_upper(start,pattern,n)
print_middle(m)
print_lower(m,pattern,n)

#String Formatting

def print_formatted(number):
    width = len("{0:b}".format(number))
    for i in range (1,(number+1)):
        print ("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width=width))
        

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

#Alphabet Rangoli

import string

def print_rangoli(size):
    alpha = string.ascii_lowercase
    subset = alpha[:size]
    base = "-".join(reversed(subset))
    rows = []
    for i in range(size):
        row = base[:len(base) - i*2]
        row = ("-" * (len(base) - len(row))) + row
        rows.insert(0, row)
    rows.extend(reversed(rows[:-1]))
    for i in range(len(rows)):
        row = rows[i]
        rev_row = row[::-1]
        rows[i] = row + rev_row[1:] 
    print(*rows, sep="\n")    


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

#Capitalize!

import os

def solve(s):
    i = 1
    if s[0].islower():
         s = s[0].upper() + s[1:]
    for char in s[1:]:
            if s[i-1] == " " and s[i].islower():
                s = s[:i] + s[i].upper() + s[i+1:]
            i += 1
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

#The Minion Game

def minion_game(string):
    count1 = 0
    count2 = 0
    vowels = ["A","E","I","O","U"]
    for i in range(len(string)):
        if string[i] in vowels:
            count1 += len(string) - i
        else:
            count2 += len(string) - i
    
    if count1 > count2:
        result = "Kevin " + str(count1)
    elif count2 > count1:
        result = "Stuart " + str(count2)
    else:
        result = "Draw"
    
    print(result)
    

if __name__ == '__main__':
    s = input()
    minion_game(s)

#Merge the Tools!

def merge_the_tools(string, k):
    p=k
    for i in range(0,len(string),k):
        t=string[i:k]
        k=k+p
        s=''
        for char in t:
            if char in s:
                pass
            else:
                s=s+char 
        print(s)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


#--------------SECTION_4: SETS------------------


#Introduction to Sets

def average(array):
    my_set = set(array)
    avg = sum(my_set)/len(my_set)
    return avg

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#No Idea!

length = input().split()
elements = input().split()
happy_elements = set(input().split())
unhappy_elements = set(input().split())
happiness = 0
for element in elements:
    if element in happy_elements:
        happiness += 1
    elif element in unhappy_elements:
        happiness -= 1
print(happiness)

#Symmetric Difference

len_first_set = input()
first_set = set(map(int, (input().split())))
len_second_set = input()
second_set = set(map(int, (input().split())))
symmetric_difference = sorted(first_set.difference(second_set).union(second_set.difference(first_set)))
for item in symmetric_difference:
    print(item)

#Set .add()

stamps = set()
stamps_num = input()
for _ in range(int(stamps_num)):
    new_stamp = input()
    stamps.add(new_stamp)
print(len(stamps))

#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
instructions_num = int(input())
for _ in range (instructions_num):
    cmd = input().split()
    funct = cmd[0]
    try:
        args = cmd[1]
    except IndexError:
        args = ''
    eval('s.'+funct+'('+args+')')
print(sum(s))

#Set .union() Operation

english_count = input()
english_subs = set(input().split())
french_count = input()
french_subs = set(input().split())
print(len(english_subs.union(french_subs)))

#Set .intersection() Operation

_ , eng_subs = input(), set(input().split())
_ , fr_subs = input(), set(input().split())
print(len(eng_subs.intersection(fr_subs)))

#Set .difference() Operation

_ , eng_subs = input() , set(input().split())
_ , fr_subs = input() , set(input().split())
print(len(eng_subs.difference(fr_subs)))

#Set .symmetric_difference() Operation

_ , eng_subs = input() , set(input().split())
_ , fr_subs = input() , set(input().split())
print(len(eng_subs.symmetric_difference(fr_subs)))

#Set Mutations

_ , my_set = input(), set(map(int,input().split()))
operations_num = int(input())
for operation in range(operations_num):
    funct = input().split()[0]
    other_set = set(map(int,input().split()))
    eval('my_set.'+funct+'(other_set)')
print(sum(my_set))

#The Captain's Room

group_size = int(input())
room_num = list(map(int, input().split()))
room_uniques = set(room_num)
no_captain_sum = sum([i*group_size for i in room_uniques])
actual_sum = sum(room_num)
captain_room = int((no_captain_sum - actual_sum) / (group_size-1))
print(captain_room)

#Check Subset

test_cases_num = int(input())
for i in range(test_cases_num):
    _, set1 = input(), set(input().split())
    _, set2 = input(), set(input().split())
    if len(set1 - set2) == 0:
        print (True)
    else:
        print (False)

#Check Strict Superset

main_set, test_sets_num = set(input().split()), int(input())
check = True
for i in range(test_sets_num):
    test_set = set(input().split())
    if len(test_set - main_set) == 0 and len(test_set) < len(main_set):
        pass
    else:
        check = False
        break
print(check)


#--------------SECTION_5: COLLECTIONS------------------


#collections.Counter()

from collections import Counter
_, size_mapping = input(), dict(Counter(input().split()))
customers_num = int(input())
earnings = []
for i in range (customers_num):
    size_req, price = input().split()
    if size_req in size_mapping and size_mapping[size_req] != 0:
        size_mapping[size_req] -= 1
        earnings.append(price)
    else:
        pass
print(sum(list(map(int,earnings))))

#DefaultDict Tutorial

from collections import defaultdict
my_dict = defaultdict(list)
list1= []

n, m = map(int,input().split())

for i in range(n):
    my_dict[input()].append(i+1) 

for i in range(m):
    list1.append(input())  

for i in list1: 
    if i in my_dict:
        print (" ".join(map(str,my_dict[i])))
    else:
        print (-1)

#Collections.namedtuple()

from collections import namedtuple

N = int(input())
fields = input().split()
total = 0
for i in range(N):
    students = namedtuple('student',fields)
    field1, field2, field3,field4 = input().split()
    student = students(field1,field2,field3,field4)
    total += int(student.MARKS)
print('{:.2f}'.format(total/N))

#Collections.OrderedDict()

store_item = dict()
for _ in range(int(input())):
    key,_,value = input().rpartition(" ")
    store_item[key] = store_item.get(key,0) + int(value)
for k,v in store_item.items():
    print(k,v)

#Word Order

from collections import OrderedDict

words = OrderedDict()
for i in range(int(input())):
    eachword = input().strip()
    words[eachword] = words.get(eachword, 0) + 1
print (len(words))
print (*words.values())

#Collections.deque()

from collections import deque

d = deque()
for _ in range(int(input())):
    cmd, *args = input().split()
    getattr(d, cmd)(*args)
[print(x, end=' ') for x in d]

#Company Logo


if __name__ == '__main__':
    s = input()
    from collections import Counter

    s = sorted(s)
    freq = Counter(list(s))
    for k,v in freq.most_common(3):
        print(k,v)

#Piling Up!

T=int(input())
for i in range(T):
    num=int(input())
    arr=list(map(int,input().split()))
    f=max(arr)
    if arr[0]==f or arr[-1]==f:
        print('Yes')
    else:
        print('No')


#--------------SECTION_6: DATE AND TIME------------------


#Calendar Module

import calendar

month, day, year = map(int,input().split())
print(calendar.day_name[calendar.weekday(year, month, day)].upper())

#Time Delta

import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds())))   

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


#--------------SECTION_7: ERRORS AND EXCEPTIONS------------------


#Exceptions

test_cases_num = int(input())
for _ in range (test_cases_num):
    a, b = input().split()
    try:
        print(int(int(a) / int(b)))
    except ZeroDivisionError as e:
        print ("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print("Error Code:",e)


#--------------SECTION_8: BUILT-INS------------------


#Zipped!

students, subjects = map(int, input().split()) 

score_matrix = []
for _ in range(subjects):
    score_matrix.append(map(float, input().split())) 

for j in zip(*score_matrix): 
    print(sum(j)/len(j))

#Athlete Sort

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    
    ordered_arr = sorted(arr, key = lambda x: x[k])
    for athlete in ordered_arr:
        print(" ".join(map(str, athlete)))

#ginortS

s = input()
s = sorted(s,key = lambda x:(x.isdigit() and int(x)%2==0, x.isdigit(),x.isupper(),x.islower(),x))
print("".join(map(str,s)))


#--------------SECTION_9: PYTHON FUNCTIONALS------------------


#Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):
    a,b = 0,1
    for i in range(n):
        yield a
        a,b = b,a+b  
if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


#--------------SECTION_10: REGEX AND PARSING------------------


#Detect Floating Point Number

import re

for i in range(int(input())):
    if re.match("^[\+-]?\d*\.\d+$", input()):
        print(True)
    else:
        print(False)

#Re.split()

regex_pattern = r"[.,]+"

import re

print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()

import re

repeating_char = re.search(r"([a-z0-9])\1+", input())
if repeating_char:
    print(repeating_char.group(1))
else:
    print(-1)

#Re.findall() & Re.finditer()

import re

vowels = 'aeiou'
consonants = 'bcdfghjklmnpqrstvwxyz'

match = re.findall('(?<=['+consonants+'])(['+vowels+']{2,})['+consonants+']', input(), re.IGNORECASE)
if match:
    print(*match, sep='\n')
else:
    print('-1')

#Re.start() & Re.end()

import re

x = input()
y = input()
r =0
s = len(y)
if re.search(r'%s' %(y),x):
    while r != len(x):
        z =re.search(r'%s' %(y),x[r: s]) 
        if z:   
            print('({}, {})'.format(z.start()+r  , z.end()-1+r))
        r +=1
        s += 1
else:
    print("(-1, -1)")

#Regex Substitution

import re

n = int(input())

for i in range(n):
    line = input()
    and_line = re.sub('(?<=\s)&&(?=\s)', 'and', line)
    final = re.sub(r'(?<=\s)[|]{2}(?=\s)', 'or', and_line)
    print (final)

#Validating Roman Numerals

regex_pattern = r"^(?!.*?([IVXCM])\1{3}|LL|DD)[IVXCMLD]+$"
import re
print(str(bool(re.match(regex_pattern, input()))))

#Validating phone numbers

import re

n=int(input())

for i in range(n):
    if re.match(r'[789]\d{9}$', input()):
        print ('YES')  
    else:
        print ('NO')  

#Validating and Parsing Email Addresses

import email.utils
import re

regex_pattern = r'^([a-z]){1}([a-z,1-9,\.,\-,\_])*(@){1}([a-z])*(\.){1}[a-z]{1,3}$'

for _ in range(int(input())):
    email_full = input()
    parsed_email = email.utils.parseaddr(email_full)
    clean_email = parsed_email[1]
    if str(bool(re.match(regex_pattern, clean_email)))=='True':
        print(email_full)
    else:
        pass

#Hex Color Code

import re

for i in range(int(input())):
    matches = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if matches:
        print(*matches, sep='\n')

#HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        [print("->", i[0],">",i[1]) for i in attrs]
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):            
        print("Empty :", tag)
        [print("->", i[0],">",i[1]) for i in attrs]

parser = MyHTMLParser()

for _ in range(int(input())):
    parser.feed(input())

#HTML Parser - Part 2

from html.parser import HTMLParser

class CustomHTMLParser(HTMLParser):
    def handle_comment(self, data):
        number_of_line = len(data.split('\n'))
        if number_of_line>1:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        if data.strip():
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

parser = CustomHTMLParser()

n = int(input())

html_string = ''
for i in range(n):
    html_string += input().rstrip()+'\n'
    
parser.feed(html_string)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]
        
html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Validating UID

import re

for _ in range(int(input())):
    s = input()
    print('Valid' if all([re.search(r, s) for r in [r'[A-Za-z0-9]{10}',r'([A-Z].*){2}',r'([0-9].*){3}']]) and not re.search(r'.*(.).*\1', s) else 'Invalid')

#Validating Credit Card Numbers

import re

for _ in range(int(input())):
    s = input()
    if re.match(r"^[456]([\d]{15}|[\d]{3}(-[\d]{4}){3})$", s) and not re.search(r"([\d])\1\1\1", s.replace("-", "")):
        print("Valid")
    else:
        print("Invalid")

#Validating Postal Codes

regex_integer_in_range = r"^[1-9][0-9]{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=.\1)"

import re

P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

#Matrix Script

import re

n, m = map(int, input().split())
a, b = [], ""
for _ in range(n):
    a.append(input())

for z in zip(*a):
    b += "".join(z)

print(re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", b))


#--------------SECTION_11: XML------------------


#XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    return sum(map(get_attr_number, node)) + len(node.attrib)

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

#XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)


#--------------SECTION_12: CLOSURES AND DECORATIONS------------------


#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        return f(['+91 ' + phone_number[-10:-5] +' '+ phone_number[-5:] for phone_number in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

#Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        people.sort(key=lambda p: int(p[2]))
        return [f(person) for person in people]

    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


#--------------SECTION_13: NUMPY------------------


#Arrays

import numpy

def arrays(arr):
    arr.reverse()
    return numpy.array(arr, float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#Shape and Reshape

import numpy

my_array = numpy.array(list(map(int,input().split())))
print(numpy.reshape(my_array,(3,3)))

#Transpose and Flatten

import numpy as np

xy = input().split()
x = xy[0]
y = xy[1]
arr = []
for i in range(int(x)):
    arr.append(input().split())
arr = np.asarray(arr, int)
print(arr.transpose())
print(arr.flatten())

#Concatenate

import numpy as np

n, m, p = map(int, input().split())

my_list = []
for _ in range(n):
    my_list.append(input().split())
arr1 = np.asarray(my_list,int)
my_list = []
for _ in range(m):
    my_list.append(input().split())
arr2 = np.asarray(my_list,int)
print(np.concatenate((arr1, arr2), axis = 0))

#Zeros and Ones

import numpy as np

count, m, *n = map(int,input().split())

print(np.zeros((count, m, *n), dtype = np.int))
print(np.ones((count, m, *n), dtype = np.int))

#Eye and Identity

import numpy

print(str(numpy.eye(*map(int,input().split()))).replace('1',' 1').replace('0',' 0'))

#Array Mathematics

import numpy as np

n, m = map(int, input().split())
array1 = np.array([input().split() for _ in range(n)], dtype=int)
array2 = np.array([input().split() for _ in range(n)], dtype=int)
print(array1+array2)
print(array1-array2)
print(array1*array2)
print(array1//array2)
print(array1%array2)
print(array1**array2)

#Floor, Ceil and Rint

import numpy as np

np.set_printoptions(sign=' ')

my_array = np.array(input().split(), dtype= float)
print(np.floor(my_array))
print(np.ceil(my_array))
print(np.rint(my_array))

#Sum and Prod

import numpy as np

n, m = map(int, input().split())
my_array = np.array([input().split() for _ in range(n)], dtype= int)
sum1 = np.sum(my_array, axis= 0)
print(np.product(sum1, axis= 0))

#Min and Max

import numpy as np

n, m = map(int, input().split())

my_arr = np.array([input().split() for _ in range(n)], dtype = int)
min_ax1 = np.min(my_arr, axis = 1)
print(np.max(min_ax1))

#Mean, Var, and Std

import numpy as np

np.set_printoptions(legacy='1.13')

n, m = map(int, input().split())

my_array = np.array([input().split() for _ in range(n)], dtype = int)

print(np.mean(my_array, axis = 1))
print(np.var(my_array, axis = 0))
print(np.std(my_array, axis = None))

#Dot and Cross

import numpy as np

n = int(input())
arr1, arr2 = [np.array([input().split() for _ in range(n)], dtype = int) for i in range(2)]
print(np.dot(arr1,arr2))

#Inner and Outer

import numpy as np

arr1 = np.array(input().split(), dtype = int)
arr2 = np.array(input().split(), dtype = int)

print(np.inner(arr1, arr2))
print(np.outer(arr1, arr2))

#Polynomials

import numpy as np

coeff = np.array(input().split(), dtype= float)
point = float(input())
print(np.polyval(coeff,point))

#Linear Algebra

import numpy as np

n = int(input())
my_arr = np.array([input().split() for _ in range(n)], dtype= float)

print(round(np.linalg.det(my_arr), 2))


#-------------------------------------------------------
#----------------------PROBLEM_2------------------------
#-------------------------------------------------------


#Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    my_dict = {}
    for i in candles:
        my_dict[i] = my_dict.get(i,0) + 1
    max_height = max(my_dict.keys())
    return my_dict[max_height]

    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps (Kangaroo)

import math
import os
import random
import re
import sys


def kangaroo(x1, v1, x2, v2):
    meet_flag = False
    if v1 <= v2:
        return 'NO'
    
    while x1 < x2:
        x1 += v1
        x2 += v2
        if x1 == x2:
            meet_flag = True
            break
    return 'YES' if meet_flag else 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising (Strange Advertising)

import math
import os
import random
import re
import sys


def viralAdvertising(n):
    new_shares = 6
    tot_likes = 2
    if n == 1:
        return tot_likes    
    for i in range(n-1):
        new_likes = new_shares // 2
        tot_likes += new_likes
        new_shares = new_likes * 3
    return tot_likes


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum

import math
import os
import random
import re
import sys


def superDigit(n, k):
    if len(n) == 1:
        return int(n)
    p = sum(int(i) for i in n) * k
    return superDigit(str(p), 1)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1

import math
import os
import random
import re
import sys


def insertionSort1(n, arr):
    to_insert = arr[-1]
    for i in range(1,n+1):
        if i == n:
            arr[0] = to_insert
            print(" ".join(map(str,arr)))
        elif to_insert < arr[n-1-i]:
            arr[n-i] = arr[n-1-i]
            print(" ".join(map(str,arr)))
        else:
            arr[n-i] = to_insert
            print(" ".join(map(str,arr)))
            break


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2

import math
import os
import random
import re
import sys


def insertionSort2(n, arr):
    for i in range(n):
        if(i == 0):
            continue
        for j in range(0, i):
            if(arr[j] > arr[i]):
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
            else:
                continue
        print(*arr)
    

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)