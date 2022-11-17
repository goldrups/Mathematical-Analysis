# standard_library.py
"""Python Essentials: The Standard Library.
Samuel Alvin Goldrup
345 Section 3
7 September 2021
"""

import sys
import random
import time
import calculator

from itertools import combinations, chain
from box import isvalid, parse_input


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), sum(L)/len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    mutable_list = []
    immutable_list = []
    
    my_int = 4
    my_int_new = my_int #assigns a new variable to the old
    my_int_new += 1 #modifies new variable
    if my_int_new == my_int: #mutable if both variables changed
    	mutable_list.append("numbers")
    else: #immutable otherwise
    	immutable_list.append("string")
    
    my_string = "hello mom"
    my_string_new = my_string
    my_string_new = "jello mom"
    if my_string_new == my_string:
    	mutable_list.append("strings")
    else:
    	immutable_list.append("strings")
    
    L = ["h", "e", "y", "m", "o", "m"]
    L_new = L
    L_new[0] = "J"
    if L_new == L:
    	mutable_list.append("lists")
    else:
    	immutable_list.append("lists")
    
    COOL_TUPLE = (1,2,3,4,5)
    COOLER_TUPLE = COOL_TUPLE
    COOLER_TUPLE += (6,)
    if COOLER_TUPLE == COOL_TUPLE:
    	mutable_list.append("tuples")
    else:
    	immutable_list.append("tuples")
    
    fun_set = {1,2,3}
    funner_set = fun_set
    funner_set.add(4)
    if funner_set == fun_set:
    	mutable_list.append("sets")
    else:
    	immutable_list.append("sets")
    
    print("mutables:", mutable_list)
    print("immutables:", immutable_list)
    


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    
    c_sq = calculator.sum(calculator.product(a,a), calculator.product(b,b)) #sum of squares
    return calculator.sqrt(c_sq)
    


# Problem 4
def power_set(A):
	"""Use itertools to compute the power set of A.

	Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

	Returns:
		  (list(sets)): The power set of A as a list of sets.
	"""
	pow_set = [] #power set
	for i in range(0,len(A)+1): #one iteration for each possible subset cardinality
		add_on = [set(z) for z in combinations(A,i)] #subsets of cardinality i
		pow_set.extend(add_on)
    
	return pow_set

# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
	"""Play a single game of shut the box.
	Accepts player (str) to be player's name and timelimit (int)
	to be the amount of time for the game to last
	"""
	
	box_nums = list(range(1,10)) #set up the box of numbers
	
	time_start = time.time()
	timelimit = float(timelimit)
	time_end = time_start + timelimit
	
	"""The loop functions as long as time remains AND the box is nonempty"""
	while time.time() < time_end and box_nums != []:
		if sum(box_nums) > 6: #roll two dice if the box sums to more than 6
			roll_1 = random.choice(list(range(1,7)))
			roll_2 = random.choice(list(range(1,7)))
			roll = roll_1 + roll_2
		else: #roll one die if the box sums to 6 or less
			roll = random.choice(list(range(1,7)))
		print("Numbers left: ", box_nums)
		print("Roll: ", roll)
		if isvalid(roll, box_nums) == True: 
		#branch executes of roll can be expressed as a sum of box numbers
			print("Seconds left: ", round(time_end-time.time(), 2))
			choice_nums = input("Numbers to eliminate: ") #numbers that user thinks sum to the roll
			nums_to_kill = parse_input(choice_nums, box_nums) #makes sure input is in right format
			if nums_to_kill == []:
				print("Invalid input")
			elif sum([int(i) for i in choice_nums.split()]) != roll: #checks for user's arithmetic mistake
				print("Invalid input")
				nums_to_kill = []
			else: 
			#if user's arithmetic is correct and input is in right format
			#then the user's chosen numbers are removed			
				for i in range(len(nums_to_kill)): 
					box_nums.remove(nums_to_kill[i])
		else:
			break
	 	
	time_stop = time.time()
	
	#print the game stats
	if len(box_nums) == 0: 
		print("Score for player", sys.argv[1], ":", sum(box_nums))
		print("Time played:", round(time_stop-time_start,2))
		print("Congratulations! You shut the box!")
	else:
		print("Game over!")
		print("")
		print("Score for player", sys.argv[1], ":", sum(box_nums))
		print("Time played:", round(time_stop-time_start,2))
		print("Better luck next time >:)")    	
    	
    	
    
    





if __name__ == "__main__":
	print(prob1([4,3,6,2,7,9,2,-1,3,-20]))
	prob2()
	print(hypot(3,4))
	print(power_set({'A','B','C'}))
	shut_the_box(sys.argv[1], sys.argv[2])