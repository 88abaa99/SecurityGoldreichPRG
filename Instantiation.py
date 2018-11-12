import BinarySystem
import random
import copy
import math
import time
import datetime

#for theoretical results
from scipy.special import comb

#for distribution graphics
import seaborn as sns
from scipy.stats import norm

"""****************************************************************************
PART 1 : OUR ATTACK
****************************************************************************"""

"""-------------------------------------------
full_attack : runs the attack on a pseudo-randomly
generated seed and system. Returns the seed.

1a) Generate linear system based on collisions
1b) Deduce how many more equations we need
2) Generate linear system with an optimized 
number of guesses so that we have all equations 
we need.
3) Get all the solutions we can get from the
system 
4) Find the other bits of the seed using the 
quadratic system
5) Check if it is the right seed, if not start 
again

Parameters:
n = the number of bits of the seed
stretch = stretch of the PRG
#import numpy
#import BinaryMatrix
test = the number of attacks we run. Default is 1
Brute_Force = decide if we brute force the guesses
or cheat by looking at the seed
Sorted = (True) Ordered case
        (False) Unordered case - default
-------------------------------------------"""

def full_attack(n,stretch,test = 1, Brute_Force = False, Sorted = False):
    m=int(round(n**stretch))
    print("__________________________________________________")
    print("Stretch = "+str(stretch)+"   n="+str(n)+"\nSorted system = "+str(Sorted)+"\nRun the Brute Force="+str(Brute_Force))
    print("__________________________________________________")
    for nb_tests  in range(test):
        """Randomly generate the system"""
        secret_seed = generate_seed(n)   ###Private seed 
        initial_system = generate_system(n,m,Sorted)   ###Public system (should not be modified)
        #initial_system = generate_system_blockwise_local(n,m)   ###Public system (should not be modified)
        output = evaluate_system(n,m,secret_seed,initial_system) ###Public output
        """NB : secret_seed is never used after this point (except if we skip the Brute_Force)"""

        """1a) Build system of collisions"""
        #colliding equations are removed from the initial system
        (col_system, col_output) = collision_system_optimized(n,m,initial_system,output,True)
        system = copy.copy(initial_system)  ###Public system (working copy)
        print( "Number of collisions: ",len(col_system))
        """1b) Deduce how many more equations we need"""
        nb_missing_eqns = n-len(col_system)
        
        """2) Build system for Guess and Determine"""
        guesses = find_optimal_guesses(system,n,nb_missing_eqns)
        guesses.sort()
        nb_occurences = count_occurences(system,n)
        nb_occurences.sort(reverse=True)
        print("Number of guesses: ",len(guesses))
        print("Number of variables to determine: ",n-len(guesses))
        iteration = 0
        found_it = False
        while found_it == False:
            iteration=iteration+1
            """Generation of a guess for the seed"""
            guesses_values,guessed_seed = guess_a_seed(Brute_Force,guesses,iteration,secret_seed,n) #The secret seed is only used if Brute_Force == False
            (extra_system, extra_output)=extra_system_with_guess_and_determine(system,output,guesses,guessed_seed)
            print( "Number of extra equations from guesses: ",len(extra_system) - len(guesses))
            
            """3) Use both to invert the system"""
            (A,B) = build_binary_matrix_optimized(extra_system + col_system, extra_output + col_output,n)
            M = BinarySystem.BinarySystem(A, B, len(A), n)
            rank = M.compute_rank()
            if rank == n:
                print("Matrice invertible")
                print("Percentage of the recovered seed : 100%")
            else:
                print("Matrice non invertible")
                print("Rank of the matrice: ",rank, "instead of",n)
                presence = list_missing_variables_in_matrix(A)
                print("Number of variables not in the system:",len(presence))
                if (n - rank) == len(presence):
                    print("A sub-matrice can be inverted")
                    percentage = (rank)*100/n
                    print("Percentage of the recovered seed :",percentage)
                else:
                    inverted_vars = count_inverted_rows(M.A)
                    print("Inverted variables: ", inverted_vars, " out of ",n-len(presence))
                    percentage =(inverted_vars)*100/n
                    print("Percentage of the recovered seed :",percentage)
                    if percentage < 50 :
                         print("!!!!!!!!!!!!!  Problem  !!!!!!!!!!!!!!! ")
            found_seed = build_seed(n,A,B)
            """4) Find the other bits of the seed using the quadratic system"""
            complete_seed = recover_complete_seed_from_quadratic_system(found_seed,output,system)
            print("Seed completed with quadratic system")
            """5) Check if it is the right seed, if not start again"""
            found_it=verify_seed(complete_seed,system,output)
            if found_it:
                print("************** We found the correct seed ***************")
            else:
                found_bits = n-complete_seed.count(-1)
                if found_bits < n:
                    print(n-found_bits," seed bits could not be found")
                else:
                    print("************** !!!!! Wrong guess  !!!!!! ***************")
                if not Brute_Force or  iteration > 2**len(guesses):
                    print("************** !!!!! Seed not found !!!!! **************")
                    found_it = True                    
            print("__________________________________________________")
    return complete_seed

"""-------------------------------------------
attack_complexity_only : randomly generates a seed and a system,
and evaluates the amount of computation needed to break it.
The obtained complexity is accurate for collisions and the guesses
but the matrice system solving algorithm is bounded by n^3.
It appeared that n^3 is greatly overestimated.

Parameters:
n = the number of bits of the seed
stretch = stretch of the PRG
test = the number of attacks we run. Default is 1
Sorted = (True) Ordered case
        (False) Unordered case - default
-------------------------------------------"""

def attack_complexity_only(n,stretch,test = 1, Sorted = False):
    nb_guesses = []
    nb_complexity = []
    m=int(round(n**stretch))
    print("__________________________________________________")
    print("Stretch = "+str(stretch)+"   n="+str(n)+"\nSorted system = "+str(Sorted))
    print("__________________________________________________")
    if Sorted == False:
        theory_collisions = m-comb(n,2)+comb(n,2)*((comb(n,2)-1)/comb(n,2))**m
        theory_guesses = n*(n - theory_collisions) / (2*m + n)
        theory_complexity = theory_guesses + math.log(n**2,2)
        print("Theoretical result: ",theory_collisions, " collisions in average.")
        print("Theoretical result: ",theory_guesses, " guesses in the worst case.")
        print("Theoretical result: ",theory_complexity , " bits of security in the worst case.")
    else:
        print("No theoretical result known for the ordered case.")
    print("__________________________________________________")
    for nb_tests  in range(test):
        secret_seed = generate_seed(n)   ###Private seed 
        initial_system = generate_system(n,m,Sorted)   ###Public system (should not be modified)
        output = evaluate_system(n,m,secret_seed,initial_system) ###Public output

        """1a) Build system of collisions"""
        #colliding equations are removed from the initial system
        (col_system, col_output) = collision_system_optimized(n,m,initial_system,output,True)
        system = copy.copy(initial_system)  ###Public system (working copy)
        print( "Test: ",nb_tests)
        print( "Number of collisions: ",len(col_system))
        """1b) Deduce how many more equations we need"""
        nb_missing_eqns = n-len(col_system)
        """2) Build system for Guess and Determine"""
        guesses = find_optimal_guesses(system,n,nb_missing_eqns)
        guesses.sort()
        nb_occurences = count_occurences(system,n)
        nb_occurences.sort(reverse=True)
        nb_guesses.append(len(guesses))
        print("Number of guesses: ",len(guesses))
        print("Number of variables to determine: ",n-len(guesses))
        complexity = math.ceil(len(guesses) + math.log(n**2,2))
        nb_complexity.append(complexity)
        print("Estimated bits of security: ",complexity)
        print("__________________________________________________")
    if test>1:
        print( "Average number of bits of security: ",sum(nb_complexity)/test)
        #sns.distplot(nb_complexity, fit=norm, kde=False)
    occ = [(i,nb_complexity.count(i)) for i in range(200)];
    for i in range(200) :
        if occ[i][1] != 0 : 
            print(occ[i])
    return (nb_guesses,occ)

"""-------------------------------------------
attack_collision_only : randomly generates a seed and
a system, and count the number of collisions.

Parameters:
n = the number of bits of the seed
stretch = stretch of the PRG
test = the number of attacks we run. Default is 1
Sorted = (True) Ordered case
        (False) Unordered case - default
-------------------------------------------"""
def attack_collision_only(n,stretch,test = 1, Sorted = False):
    nb_collision = []
    m=int(round(n**stretch))
    print("__________________________________________________")
    print("Stretch = ",stretch,"   n = ",n,"\nSorted system = ",Sorted)
    print("__________________________________________________")
    if Sorted == False:
        print("Theoretical result: ",m-comb(n,2)+comb(n,2)*((comb(n,2)-1)/comb(n,2))**m, " in average.")
    else:
        print("No theoretical result known for the ordered case.")
    print("__________________________________________________")
    for nb_tests  in range(test):
        secret_seed = generate_seed(n)   ###Private seed 
        initial_system = generate_system(n,m,Sorted)   ###Public system (should not be modified)
        output = evaluate_system(n,m,secret_seed,initial_system) ###Public output
        
        """1a) Build system of collisions"""
        #colliding equations are removed from the initial system
        (col_system, col_output) = collision_system_optimized(n,m,initial_system,output,True)
        print( "Number of collisions: ",len(col_system))
        nb_collision.append(len(col_system))
        print("__________________________________________________")
    if test>1:
        print( "Average number of collisions: ",sum(nb_collision)/test)
        sns.distplot(nb_collision, fit=norm, kde=False)

    return nb_collision

"""****************************************************************************
PART 2 : INITIALIZING AND EVALUATING THE PRG
****************************************************************************"""

#random = Random.RandomGenerator(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H:%M:%S'))

"""-------------------------------------------
Generate a random seed of n bits
-------------------------------------------"""
def generate_seed(n):
    seed = [0]*n
    for i in range(n):
        if random.randint(0,1):
            seed[i] = 1
    return seed

"""-------------------------------------------
Generate a random system of m equations
Each 
-------------------------------------------"""
def generate_system(n,m,Sorted):
    system = [[] for i in range(m)]
    for i in range(m):
        subset = [0]*5
        while subset[0]==subset[1] or subset[0]==subset[2] or subset[0]==subset[3] or subset[0]==subset[4] or subset[1]==subset[2] or subset[1]==subset[3] or subset[1]==subset[4] or subset[2]==subset[3] or subset[2]==subset[4] or subset[3]==subset[4]:
            subset[0] = random.randint(0,n-1)
            subset[1] = random.randint(0,n-1)
            subset[2] = random.randint(0,n-1)
            subset[3] = random.randint(0,n-1)
            subset[4] = random.randint(0,n-1)
            if subset[3] > subset[4]:
                tmp = subset[3]
                subset[3] = subset[4]
                subset[4] = tmp
        if Sorted:
            subset.sort()
        system[i] = subset
    return system

"""-------------------------------------------
Generate a random system of m equations
Each 
-------------------------------------------"""
def generate_system_blockwise_local(n,m):
    system = [[] for i in range(m)]
    for i in range(m):
        subset = [0]*5
        while subset[0]==subset[1] or subset[0]==subset[2] or subset[1]==subset[2] :
            subset[0] = random.randint(0,int(n/2)-1)
            subset[1] = random.randint(0,int(n/2)-1)
            subset[2] = random.randint(0,int(n/2)-1)
        subset[3]= subset[0]+ int(n/2)
        subset[4] = subset[1] + int(n/2)
        system[i] = subset
    return system

"""-------------------------------------------
Evaluate the given system for the given seed
-------------------------------------------"""
def evaluate_system(n,m,seed,system):
    output = [0]*m
    for i in range(m):
        output[i] = seed[system[i][0]] ^ seed[system[i][1]] ^ seed[system[i][2]] ^ (seed[system[i][3]]*seed[system[i][4]])
    return output

"""****************************************************************************
PART 3 : TOOLBOX FOR THE ATTACK
****************************************************************************"""

"""-------------------------------------------
Check for collisions of degree-two monomials
and build a linear system out of it
If remove_collision_from_system is True,
the colliding equations are removed from the system
---------------------------------------------
Use a 2D-matrix to store the list of degree-two monomials,
then the algorithm is not as detailed in the paper, but is easier to implement.
Also, the complexity of this algorithm is still negligible in front of the guess and determine.
-------------------------------------------"""
def collision_system_optimized(n,m,system,evaluation,remove_collision_from_system = False):
    #First, make a list of all degree two variables [i3,i4]
    list_degree_two = [[] for i in range(n)]    #list of degree_two_monomials
    degree_two_index = [[] for i in range(n)]   #equation index for the list of degree-two monomials
    collision_system = []
    collision_evaluation = []
    collision_index = []    #index of equations that collides with another one
    for i in range(m):      #if no collision occurs
        if not (system[i][4] in list_degree_two[system[i][3]]):
            list_degree_two[system[i][3]].append(system[i][4])
            degree_two_index[system[i][3]].append(i)
        else:               #if a collision occurs
            index = degree_two_index[system[i][3]][list_degree_two[system[i][3]].index(system[i][4])]
            new_equation = [system[index][0],system[index][1],system[index][2],system[i][0],system[i][1],system[i][2]]
            new_equation.sort()
            j = 0
            while j < len(new_equation)-1: #check if two monomials are equal and remove them
                if new_equation[j] == new_equation[j+1]:
                    del new_equation[j]
                    del new_equation[j]
                else:
                    j += 1
            if len(new_equation)>0:
                collision_system.append(new_equation)   #add the equation to the new system
                collision_evaluation.append(evaluation[index] ^ evaluation[i])  #and the corresponding evaluation
            if remove_collision_from_system:
                collision_index.append(i)
    #for the collisions found, the second equation is removed (the first is kept)
    if remove_collision_from_system:
        for i in reversed(collision_index):
            del system[i]
            del evaluation[i]

    return (collision_system, collision_evaluation)

"""-------------------------------------------
Check in the system the presence of every variable
Return a list of indices of missing variables
-------------------------------------------"""
def list_missing_variables_in_system(n,system):
    presence_vector = [0]*n
    for i in range(len(system)):
        for j in range(len(system[i])):
            presence_vector[system[i][j]] = 1
    return [i for i in range(len(system)) if presence_vector[i]==0]

"""-------------------------------------------
Check in a matrix the presence of every variable
Return a list of indices of missing variables
-------------------------------------------"""
def list_missing_variables_in_matrix(matrix):
    n = len(matrix[0])
    presence_vector = [0]*n
    for i in range(len(matrix)):
        presence_vector = [presence_vector[j] | matrix[i][j] for j in range(n)]
    return [i for i in range(n) if presence_vector[i]==0]

"""-------------------------------------------
Convert a boolean system (i.e. list of indexes)
into a binary matrix.
[2,4,7] -> [0,0,1,0,1,0,0,1,0,...]
-------------------------------------------"""
def build_binary_matrix(system, nb_cols):
    nb_rows = len(system)
    M = [[0]*nb_cols for i in range(nb_rows)]
    for i in range(nb_rows):
        for j in range(len(system[i])):
            M[i][system[i][j]] ^= 1
    return M

"""-------------------------------------------
Convert a boolean system (i.e. list of indexes)
into a binary matrix
Try to make an upper triangular matrix
and output the indices of the rows that match an upper triangular matrix
-------------------------------------------"""
def build_binary_matrix_optimized(system, output, nb_cols):
    nb_rows = max(len(system), nb_cols) #En theorie, len(system) >= nb_cols, mais ce n'est pas toujours le cas a causes des guesses qui ne sont pas exacts
    M = [[0]*nb_cols for i in range(nb_rows)]
    B = [0]*nb_rows
    already_used_rows_in_matrix = [0]*nb_rows
    already_used_rows_in_system = [0]*nb_rows
    nb_supplementary_rows = 0
    for i in range(len(system)):
        first_one = min([system[i][j] for j in range(len(system[i]))])  #Search the first 1 in the equation
        if not already_used_rows_in_matrix[first_one]:                  #try to put this first 1 in the diagonal
            already_used_rows_in_matrix[first_one] = 1
            already_used_rows_in_system[i] = 1
            for j in range(len(system[i])):                             #copy the equation
                M[first_one][system[i][j]] ^= 1
            B[first_one] = output[i]
            
        else:                                                           #if the row is already used, put the equation in the supplementary rows
            if nb_supplementary_rows < nb_rows - nb_cols:               #if there are not too many supplementary equations, add it
                already_used_rows_in_matrix[nb_cols + nb_supplementary_rows] = 1
                already_used_rows_in_system[i] = 1
                for j in range(len(system[i])):
                    M[nb_cols + nb_supplementary_rows][system[i][j]] ^= 1
                B[nb_cols + nb_supplementary_rows] = output[i]
                nb_supplementary_rows += 1
    
    free_row = 0
    for i in range(len(system)):                                            #check that all rows of the system have been copied
        if not already_used_rows_in_system[i]:                              #if not, copy it
            while already_used_rows_in_matrix[free_row]:                    #find a free row in the matrix
                free_row += 1
            for j in range(len(system[i])):
                M[free_row][system[i][j]] ^= 1
            B[free_row] = output[i]
            already_used_rows_in_system[i] = 1
            already_used_rows_in_matrix[free_row] = 1
    return (M,B)

"""-------------------------------------------
Count the occurences of the variables  in the degree-two monomials
The result is parsed as follows :
    R = [[250,0],[128,1],[nb_occurence, index_variable],...]
This allows R to be sorted easily by the number of occurences with R.sort()
without losing the index of the variables.
OBSOLETE
-------------------------------------------"""
def count_occurences(system, n):
    R = [[0,j] for j in range(n)]
    for i in range(len(system)):
        R[system[i][3]][0] += 1
        R[system[i][4]][0] += 1
    return R

"""-------------------------------------------
Find the variable that appears the most in the degree-two monomials that were not already used by previous guesses.
:param system: the initial quadratic system
:param n: the seed size
:param mask: the mask of already used equations. Should be initialized with [0]*len(system)
Returns (the index of the variable, the number of times in appears, the updated mask)
-------------------------------------------"""
def most_appearing_variable_with_mask(system, n, mask):
    R = [0]*n
    for i in range(len(system)):
        if mask[i] == 0:
            R[system[i][3]] += 1
            R[system[i][4]] += 1
    max_occurences = max(R)
    most_appearing_variable = R.index(max_occurences)
    for i in range(len(system)): #update the mask
        if system[i][3] == most_appearing_variable or system[i][4] == most_appearing_variable:
            mask[i] = 1
    return (most_appearing_variable, max_occurences, mask)

"""-------------------------------------------
Returns the optimal values to guess.
The algorithm is exactly the one detailed in the paper.
:param system: the initial quadratic system
:param n: the seed size
:param L: the number of equations to linearize
Caution: for some reason, the rest of the algorithm can fail if no guess is made
-------------------------------------------"""
def find_optimal_guesses(system,n,L):
    guesses = []
    sum=0
    mask = [0]*len(system)
    while sum <= L:
        (most_appearing_variable, nb_occurences, mask) = most_appearing_variable_with_mask(system,n,mask)
        guesses.append(most_appearing_variable)
        sum += nb_occurences
    return guesses

"""------------------------------------------
NEVER USED
------------------------------------------"""
def count_all_occurences(system, n):
    R = [[0,j] for j in range(n)]
    for i in range(len(system)):
        for j in range(len(system[0])):
            R[system[i][j]][0] += 1
    return R

"""-------------------------------------------
Generates a matrix with the system evaluated
 with several indices of the seed.
The matrix has n columns and has the correct seed value
in the indices that are guessed
OBSOLETE
-------------------------------------------"""

def extra_matrix_with_guess_and_determine(system,indices_of_guesses,seed,n):
    extra_eqns=[]
    for i in range(len(system)):
        extra_eq=[0]*n
        if system[i][3] in indices_of_guesses and not system[i][4] in indices_of_guesses:
            extra_eq[system[i][4]]=seed[system[i][3]]
            for j in range(3):
                if system[i][j] not in indices_of_guesses:
                    extra_eq[system[i][j]]=1
            extra_eqns.append(extra_eq);
        elif  system[i][4] in indices_of_guesses and not system[i][3] in indices_of_guesses:
            extra_eq[system[i][3]]=seed[system[i][4]]
            for j in range(3):
                if system[i][j] not in indices_of_guesses:
                    extra_eq[system[i][j]]=1
            extra_eqns.append(extra_eq);
        elif system[i][4] in indices_of_guesses and system[i][3] in indices_of_guesses:
            for j in range(3):
                if system[i][j] not in indices_of_guesses:
                    extra_eq[system[i][j]]=1
            extra_eqns.append(extra_eq);
    #nb_rows = len(extra_eqns)
    return extra_eqns

"""-------------------------------------------
Generates a system with the initial system evaluated
 with several indices of the seed.
:param system: the initial system
:param output: the initial right-hand-side
:param indices_of_guesses: the indices of the seed that are guessed
:param seed: the seed candidate (only the indices_of_guesses will be read)
-------------------------------------------"""
def extra_system_with_guess_and_determine(system,output,indices_of_guesses,seed):
    extra_eqns = []
    extra_outputs = []
    
    #Add equations corresponding to the guessed seed bits
    for i in range(len(indices_of_guesses)):
        extra_eqns.append([indices_of_guesses[i]])
        extra_outputs.append(seed[indices_of_guesses[i]])
    
    #Search equations that becomes linear with the guessed bits
    for i in range(len(system)):
        extra_eq=[]
        extra_out=output[i]        
        if system[i][3] in indices_of_guesses and not system[i][4] in indices_of_guesses:
            if seed[system[i][3]]:
                extra_eq.append(system[i][4])
            for j in range(3):
                if system[i][j] not in indices_of_guesses:
                    extra_eq.append(system[i][j])
                else:   #not necessary but will save time later
                    extra_out=extra_out^seed[system[i][j]]
        elif  system[i][4] in indices_of_guesses and not system[i][3] in indices_of_guesses:
            if seed[system[i][4]]:
                extra_eq.append(system[i][3])
            for j in range(3):
                if system[i][j] not in indices_of_guesses:
                    extra_eq.append(system[i][j])
                else:
                    extra_out=extra_out^seed[system[i][j]]
        elif system[i][4] in indices_of_guesses and system[i][3] in indices_of_guesses:
            extra_out=extra_out^(seed[system[i][3]] * seed[system[i][4]])
            for j in range(3):
                if system[i][j] not in indices_of_guesses:
                    extra_eq.append(system[i][j])
                else:
                    extra_out=extra_out^seed[system[i][j]]
        if len(extra_eq)>0:
            extra_eqns.append(extra_eq)
            extra_outputs.append(extra_out)
        #else:
            #SINON LE GUESS EST PEUT-ETRE FAUX!!!!!! NON GERE!!!!!!!!!!!!!!!
    return (extra_eqns, extra_outputs)


"""-------------------------------------------
Removes the columns corresponding to the guessed values
No return, the modification are made directly on the matrix
-------------------------------------------"""

def remove_guesses(matrix,output,guesses,seed,n):
    guesses.sort(reverse=True)
    for g in guesses:
        for i in range(len(matrix)): 
            if matrix[i][g] == 1:
                output[i] = output[i]^seed[g]
        for i in range(len(matrix)): 
            del matrix[i][g]
    return 0


"""-------------------------------------------
Returns the optimal values to guess.
The algorithm is more naive than the one detailed in the paper, since the m equations are read only once.
However, it makes no difference in practice with overwhelming probability.
-------------------------------------------"""
def find_optimal_guesses_naive(system,n,L):
    occurences = count_occurences(system,n)
    occurences.sort(reverse=True)
    #print occurences
    nb=0
    sum=0
    while sum <= L:
        sum=sum+occurences[nb][0]
        nb=nb+1
    if L >0:    
        return[ occurences[i][1] for i in range(nb)]
    else:
        return[ occurences[0][1]]

"""-------------------------------------------
Returns a guess for the sorted version
OBSOLETE
-------------------------------------------"""
def find_guess_for_sorted(col_system,n):
    occurences= count_all_occurences(col_system,n)
    occurences.sort(reverse=True)
    return [occurences[0][1]]


"""-------------------------------------------
Builds the seed
Uses the rows of A that match a diagonal matrice
and the corresponding elements of the right-and-side B.
-------------------------------------------"""  
def build_seed(n,A,B):
    seed=[]
    # on ajoute toutes les variables retrouvees
    for i in range(len(A[0])):
        if A[i][i]==1 and A[i].count(1)==1:
            seed.append(B[i])
        else:
            seed.append(-1)
    return seed     


"""-------------------------------------------
Count the number of rows that have been inverted
These contain a single one in diagonal
-------------------------------------------"""
def count_inverted_rows(matrix):
    counter = 0
    for i in range(len(matrix[0])):
        if matrix[i][i]==1 and matrix[i].count(1)==1:
            counter += 1
    return counter

"""-------------------------------------------
Recover the complete seed from a partial seed 
using the quadratic system
NOT OPTIMIZED AT ALL !!!!
-------------------------------------------"""
def recover_complete_seed_from_quadratic_system(found_seed,output,system):
    n = len(found_seed)
    indices_of_found_seed = []
    indices_of_not_found_seed = [i for i in range(n)]
    complete_seed=found_seed
    for i in range(n):
        if found_seed[i] != -1:
            indices_of_found_seed.append(i)
            indices_of_not_found_seed.remove(i)
    # Evaluation of the system with the known parts of the seed
    (new_system, new_output)= extra_system_with_guess_and_determine(system,output,indices_of_found_seed,found_seed)
    # This part uses only identity relations
    """for i in range(len(new_system)):
        if len(new_system[i]) == 1:
            complete_seed[new_system[i][0]] = new_output[i]"""
    # This parts builds a linear system
    i=0
    # We remove the equations 0=0
    for it in range(len(new_system)):
        if len(new_system[i]) == 0:
            del new_system[i]
            del new_output[i]
        else :
            i=i+1
    # And invert the linear system with the echelon form :
    matrix = build_binary_matrix(new_system,len(found_seed))
    remove_guesses(matrix,new_output,indices_of_found_seed,found_seed,n)
    M = BinarySystem.BinarySystem(matrix, new_output, len(new_system), len(indices_of_not_found_seed))
    rank=M.compute_rank()
    sol=solve_echelon_system(M.A[0:rank][0:rank],new_output[0:len(indices_of_not_found_seed)])
    if len(sol) == len(indices_of_not_found_seed):
        i=0
        for a in indices_of_not_found_seed:
            complete_seed[a]=sol[i]
            i=i+1

    return complete_seed

"""-------------------------------------------
Recover the complete seed from a partial seed 
using the quadratic system, or return False if
the partial seed does not match the output (i.e.
if the guess is incorrect).
Remplace recover_complete_seed_from_quadratic_system
-------------------------------------------"""
def complete_and_verify_seed(found_seed,output,system):
    n = len(found_seed)
    postponed = range(len(system)) #liste des equations qu'on ne sait pas encore resoudre
    new_postponed = []
    postpone_counter = 0
    
    # Evaluation of the system with the known parts of the seed
    #Could be improved with a lookup table over found_seed[system[i][j]] (0<=j<=4)
    for i in range(len(system)):
        if found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
            #si toutes les variables sont connues, ou alors une seule manque mais est pultipliee par zero
            if output[i] != found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]]):
                #the candidate seed does not match the public output
                return False
        else:# au moins une variable manque, et celle-ci n'est pas multipliee par 0
            if found_seed[system[i][0]] == -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                #seule la premiere variable manque
                found_seed[system[i][0]] = output[i] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]])
            elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] == -1 and found_seed[system[i][2]] != -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                #seule la deuxieme variable manque
                found_seed[system[i][1]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][2]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]])
            elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] == -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                #seule la troisieme variable manque
                found_seed[system[i][2]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]])
            elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 1:
                #seule la quatrieme variable manque et elle est multipliee par 1
                found_seed[system[i][3]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]]
            elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and found_seed[system[i][3]] == 1 and found_seed[system[i][4]] == -1:
                #seule la cinquieme variable manque et elle est multipliee par 1
                found_seed[system[i][4]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]]
            elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == -1 and (output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]]) == 1:
                #les deux dernieres variables manquent, mais valent toutes les deux 1
                found_seed[system[i][3]] = 1
                found_seed[system[i][4]] = 1
            else:
                new_postponed.append(i)
                postpone_counter+=1
    
    #retry until the seed is entirely found or until we are blocked
    while len(new_postponed) < len(postponed):
        postponed = new_postponed
        new_postponed = []
        for i in postponed:
            if found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                #si toutes les variables sont connues, ou alors une seule manque mais est multipliee par zero
                if output[i] != found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]]):
                    #the candidate seed does not match the public output
                    return False
            else:# au moins une variable manque, et celle-ci n'est pas multipliee par 0
                if found_seed[system[i][0]] == -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                    #seule la premiere variable manque
                    found_seed[system[i][0]] = output[i] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]])
                elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] == -1 and found_seed[system[i][2]] != -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                    #seule la deuxieme variable manque
                    found_seed[system[i][1]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][2]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]])
                elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] == -1 and ((found_seed[system[i][3]] != -1 and found_seed[system[i][4]] != -1) or (found_seed[system[i][3]] == 0 and found_seed[system[i][4]] == -1) or (found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 0)):
                    #seule la troisieme variable manque
                    found_seed[system[i][2]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ (found_seed[system[i][3]]*found_seed[system[i][4]])
                elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == 1:
                    #seule la quatrieme variable manque et elle est multipliee par 1
                    found_seed[system[i][3]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]]
                elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and found_seed[system[i][3]] == 1 and found_seed[system[i][4]] == -1:
                    #seule la cinquieme variable manque et elle est multipliee par 1
                    found_seed[system[i][4]] = output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]]
                elif found_seed[system[i][0]] != -1 and found_seed[system[i][1]] != -1 and found_seed[system[i][2]] != -1 and found_seed[system[i][3]] == -1 and found_seed[system[i][4]] == -1 and (output[i] ^ found_seed[system[i][0]] ^ found_seed[system[i][1]] ^ found_seed[system[i][2]]) == 1:
                    #les deux dernieres variables manquent, mais valent toutes les deux 1
                    found_seed[system[i][3]] = 1
                    found_seed[system[i][4]] = 1
                else:
                    new_postponed.append(i)
                    postpone_counter+=1
    
    if len(new_postponed)>0:
        print("Unable to retrieve the entire seed!")
    print("Postponed equations: "+str(postpone_counter))
    return found_seed


"""-------------------------------------------
Verify if a candidate seed matches the public
system and the public output.
Return True or False
-------------------------------------------"""    
def verify_seed(seed,system,output):
    if evaluate_system(len(seed),len(system),seed,system) == output:
        return True
    else:
        return False

"""-------------------------------------------
Guess for the seed
Gives the next iteration of the partial seed for the guess and determine.
CRASH FOR index_guesses empty !!!!!
-------------------------------------------"""    
def guess_a_seed(brute_force,index_guesses,iteration,secret_seed,n):
    seed = [-1 for i in range(n)]
    j=0
    if len(index_guesses)>0 and brute_force:
        st=str(bin(iteration))
        bf=[0 for i in range(len(index_guesses))]
        it=0
        l = len(st)-2
        for i in st:
            if (it != 0 and it != 1):
                bf[len(index_guesses)-(l-(it-2)-1)-1 ] = int(i)
            it=it+1
        #bf=brute_force(iteration,len(index_guesses))
        for i in index_guesses:
            seed[i]= bf[j] 
            j=j+1
    elif len(index_guesses)>0 and brute_force == False:
        print( "----> Brute force skipped ! Add a factor: 2^" + str(len(index_guesses)))
        for i in index_guesses:    
            seed[i]= secret_seed[i]
            j=j+1
    guessed_value = [seed[i] for i in index_guesses]
    return(guessed_value,seed)

"""-------------------------------------------
Solve a binary echelon system
Probably not optimized
OBSOLETE
-------------------------------------------"""
def solve_echelon_system(system,output):
    n=len(system)
    solution = [-1 for i in range(n)]
    for i in range(n):
        out = output[n-1-i]
        for j in range(i):
            out = out ^ (system[n-1-j][n-2-j]*solution[n-2-j])
        solution[n-1-i] = out * system[n-1-i][n-1-i]
    return solution


"""-------------------------------------------
Brute Force on a vector of a certain size. 
Returns a tab of a certain size with values
corresponding to the binary writing of the 
iteration
OBSOLETE
-------------------------------------------"""
def brute_force(iteration,size):
    st=str(bin(iteration))
    tab=[0 for i in range(size)]
    it=0
    l = len(st)-2
    for i in st:
        if (it != 0 and it != 1):
            tab[size-(l-(it-2)-1)-1 ] = int(i)
        it=it+1
    return tab


"""****************************************************************************
PART 4 : ANALYSIS OF THE HYPOTHESIS
****************************************************************************"""

"""-------------------------------------------
Check whether the hypothesis is true for given
values of n, stretch and k:
knowing k*n bits of the seed, we can recover
the full seed with the quadratic system.
:param n: the seed size
:param stretch: the stretch of the PRG
:param k: the fraction of seed to give to the attacker
:param test: the number of test, 1 by default
-------------------------------------------"""
def hypothesis_try_k(n,stretch,k,test = 1):
    m=int(round(n**stretch))
    print("__________________________________________________")
    print("Stretch = "+str(stretch)+"   n="+str(n)+"\n k="+str(k))
    print("__________________________________________________")
    for nb_tests  in range(test):
        """Randomly generate the system"""
        secret_seed = generate_seed(n)   ###Private seed 
        initial_system = generate_system(n,m,False)   ###Public system (should not be modified)
        #initial_system = generate_system_blockwise_local(n,m)   ###Public system (should not be modified)
        output = evaluate_system(n,m,secret_seed,initial_system) ###Public output
        
        """Construct a partial seed of kn bits"""
		
        partial_seed = [-1]*n
        for i in range(int(k*n)):
            partial_seed[i] = secret_seed[i]
        print("System built.")

        """Reconstruct the entire key"""
        partial_seed = complete_and_verify_seed(partial_seed,output,initial_system)
        
        if secret_seed == partial_seed:
            print("************** We found the correct seed ***************")
        elif evaluate_system(n,m,partial_seed,initial_system) == output:
            print("************** We found another seed ***************")
        else:
            print("!!!!!!!!!!!!!! Could not recover the entire seed !!!!!!!!!!!!!!!")
    return (secret_seed,partial_seed)
