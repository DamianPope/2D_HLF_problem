'''
IBM Qiskit version of the code that implements the quantum circuit in: 

    Quantum Advantage with Shallow Circuits 
    Sergey Bravyi, David Gosset, Robert Koenig. Science 362 (6412) pp. 308-311, 2018
    https://arxiv.org/abs/1704.00690

The quantum circuit solves the 2D Hidden Linear Function problem using a *constant* depth circuit.
Classically, we need a circuit whose depth scales *logarithmically* with the number of bits that the function acts on.
Note that the quantum circuit is a non-oracular version of the Bernstein-Vazirani algorithm.

Why is the quantum circuit significant? It highlights the existence of quantum advantage for the problem. Note that, as the authors say,:
1. the circuit is concrete and so the quantum advantange exists for a concrete, physically realizable circuit. It is *not* relative to an abstract oracle.
2. the quantum advantage is rigourously provable (as shown in the paper). The quantum circuit is more efficient than any possible classical circuit that solves the problem. 
It's not just better than the best known classical circuit. 

Note: The circuit below uses classical bits to control the controlled Z (CZ) gates and the controlled S gates in the circuit. 
This follows Footnote 3 on page 10 and can be done as the control qubits are always either 0 or 1. 
Using classical bits as controls minimizes the number of qubits, making it easier to implement the circuit physically.
'''    

#from qiskit import *
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister,ClassicalRegister
import numpy as np

#sample b vector for n = 4 case
#b = [1,0,0,1]

b = [0,0,1,0,0,1,1,1,1]

#sample b vector for n = 16 case
#b = [0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,0]

#define the parameter n
n = len(b)

N = int(pow(n,0.5))

#initialize the 2D list that stores elements of the A matrix
A = [[0]*n for i in range(n)]

#generate A
for K in range(n):
    for L in range(K+1,n):
        #For each matrix element, calculate which two vertices in the associated NxN grid graph that the element is associated with.
        #The first argument of A corresponds to one of the two vertices. The second argument corresponds to the other vertex.
        firstGridColumn =  K % N
        firstGridRow = int(K/N)

        secondGridColumn =  L % N
        secondGridRow = int(L/N)

        #test if the two vertices are joined by a HORIZONTAL edge; if so, then set the A element to 1
        if (firstGridRow==secondGridRow and (abs(firstGridColumn-secondGridColumn)==1)):    
            A[K][L] = 1 

        #test if the two vertices are joined by a VERTICAL edge; if so, then set the A element to 1
        elif firstGridColumn==secondGridColumn and abs(firstGridRow-secondGridRow)==1:
            A[K][L] = 1
        
        #if the two vertices aren’t joined by any edge, set the A matrix element to zero.
        else: 
            A[K][L] = 0
        
print("PARAMETERS")
print("n=",n)
print("b=",b)
print("A matrix=",A)

#create a circuit with n qubits & n classical bits
q = QuantumRegister(n)
c = ClassicalRegister(n)
circuit = QuantumCircuit(q, c)

#specify the gates in the quantum circuit

#initial H gates
for i in range(n):
    circuit.h(i)

'''controlled Z gates. Note that this part of the circuit has a constant depth of four---no matter how large n is

     First, implement the gates in FIRST layer of the CZ gates

     Note that that each gate in this layer acts on separate qubits.
     So, in principle, we could implement all of the gates simultaneously (the same property holds for all the gates in the other three layers)'''

for gridRow in range(N):
    #don't include the rightmost grid column as there are no vertices to the right of it to create edges to
    for gridColumn in range(0,N-1):    
        #execute CZ gates in all the EVEN columns (left vertices of horizontal edges)
        if gridColumn%2==0:
            circuit.cz(gridColumn+gridRow*N,gridColumn+gridRow*N+1)

#the gates in SECOND layer of the CZ gates

for gridRow in range(N):
    for gridColumn in range(0,N-1):            
        if gridColumn%2==1:
            circuit.cz(gridColumn+gridRow*N,gridColumn+gridRow*N+1)    

#the gates in THIRD layer of the CZ gates

for gridColumn in range(N):
    for gridRow in range(0,N-1):
        if gridRow%2==0:
            circuit.cz(gridColumn+gridRow*N,gridColumn+(gridRow+1)*N)

#the gates in FOURTH layer of the CZ gates

for gridColumn in range(N):
    for gridRow in range(0,N-1):    
        if gridRow%2==1:     
            circuit.cz(gridColumn+gridRow*N,gridColumn+(gridRow+1)*N)

#S gates
for i in range(n):
    if (b[i]==1):
        circuit.p(-3.1415926/2, i)

#final set of H gates
for i in range(n):
    circuit.h(i)

#Measure each qubit in the computational basis & store the results in the classical register
circuit.measure(q,c)

#Execute the circuit
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=1)
result = job.result()
counts = result.get_counts(circuit)

#store the value of the measured bit string, z, in result_bit_string
result_bit_string = list(counts.keys())[0]

#Convert result_bit_string to a list & then an array
z_list = []

for i in range(n):
     z_list.append(int(result_bit_string[i]))

#reverse the order of bits in z_list to convert from little endian format to big endian
z_list.reverse()

z = np.array(z_list)

print("\nHidden Bit String, z = ",z, "\n")

#
#The code below verifies that z defines the hidden linear function.
#

def convert_to_binary_string(i):
    '''    
    Converts the decimal integer i into an equivalent string of bits
    
    Arg: 
        i (int): the integer that we want to convert into a string

    returns:
        tempTwo (str): the bit string that's equivalent to i 
    '''  
    
    temp = bin(i)

    #remove the leading ‘0b’ at the start of the string
    tempTwo = temp[2:]

    #add leading zeroes if (len(tempTwo) < n) to make len(tempTwo) = n
    if len(tempTwo) < n:
        for i in range(n - len(tempTwo)):
            tempTwo = '0' + tempTwo 

    return tempTwo

def calculate_inner_product_between_z_and_x(k):
    '''    
    Calculates the value of z^{T} x in the hidden linear function q(x) = (2 z^{T} x) mod 4
    
    Arg: 
        k (int): an integer that specifies that the x vector used is the k^{th} x vector

    returns:
        temp (int): the value of z^{T} x
    '''
    
    temp = 0

    for i in range(n):
        temp += z[i]*int(x_vectors_list[k][i])

    return temp

def calculateBitwiseAdditionModulo2(a,b):
    '''    
    Adds two x vectors together in a bitwise fashion, modulo 2
    
    Args: 
        a,b (int): indices that specify the x vectors that are added together

    returns:
        tempString (str): bitwise sum of x_{a} & x_{b}, modulo 2
    '''

    #initialize tempString
    tempString = ""
    
    for i in range(n):
     	tempString = tempString + str((int(x_vectors_list[a][i]) + int(x_vectors_list[b][i])) % 2)
    return tempString

def calculate_q_using_quadratic_form(i):
    '''    
    Calculates q for a given x vector (specified by i) using the quadratic form
    
    Args: 
        i (int): specifies the i^{th} x vector that's used as the argument in q(x)

    returns:
        (int) the value of q(x_{i})
    '''
    quadratic_term = 0
    linear_term = 0

    for alpha in range(0,n):
        linear_term += b[alpha]*int(x_vectors_list[i][alpha])

    for beta in range(1,n):
        for alpha in range(0,beta):    
            quadratic_term += A[alpha][beta]*int(x_vectors_list[i][alpha])*int(x_vectors_list[i][beta])
	    
    return ((2*quadratic_term + linear_term)%4)

def findMembersOfScriptLsubspace(potentialIndicesList):    
    '''    
    Returns the indices of the x values that are members of the subspace script L_{q}
    
    Arg: 
        potentialIndicesList (list(int)): list of the indices of all the x vectors that *might* be members of script L_{q}

    returns:
        membersOfScriptLsubspace (list(int)): list of the indices of all the x vectors in script L_{q}
    '''
    
    #initialize list that contains members of script L_{q}
    membersOfScriptLsubspace = []
    
    #Loop through all the x’s that might be in script L_{q} & check if they’re actually in it
    for j in range(len(potentialIndicesList)):

        #loop through all 2**n x’ strings and, for each string, test to see if q(x(J)) + q(x’) = q(x(J) ⊕ x’), where J denotes an element of potentialIndicesList
        #The operator '⊕' denotes bitwise addition modulo 2.
        #If the equality above holds for all x’, then x(J) is a member of script L_{q}
        isAMemberOfScriptLsubspaceFlag=True
        
        for i in range(2**n):
            if ((q[potentialIndicesList[j]] + q[i]) % 4) != q[int(calculateBitwiseAdditionModulo2(potentialIndicesList[j],i),2)]:
                isAMemberOfScriptLsubspaceFlag=False

        if isAMemberOfScriptLsubspaceFlag:
            #Add the index potentialIndicesList[j] to a list of indices that contains all the x’s that are actually elements of script L_{q}
            membersOfScriptLsubspace.append(potentialIndicesList[j])

    return membersOfScriptLsubspace


#initialize x_vectors_list, a variable that stores all 2**n x vectors
x_vectors_list = []

#initialize list that contains all q values
q = []

potentialElementsOfScriptLSubspaceIndices = []

#create all the 2^n possible x binary strings
for i in range(2**n):
    x_vectors_list.append(convert_to_binary_string(i))

    #calculate q(x(i)) using the quadratic form & store it in q
    q.append(calculate_q_using_quadratic_form(i))

	#
	#do a preliminary check to see if x(i) *might* be an element of the script L_{q} subspace
	#

    #potentialElementsOfScriptLSubspaceIndices is a list that stores the index of each x that *might* be a member of the script L_{q} subspace because q(x) = 0 or 2
    if q[i] == 0 or q[i] == 2:
        potentialElementsOfScriptLSubspaceIndices.append(i) 

#determine which x’s are actually in script L_{q}
globalMembersOfScriptLsubspace = findMembersOfScriptLsubspace(potentialElementsOfScriptLSubspaceIndices)

print("Members of script L_{q} subspace")
for k in range(len(globalMembersOfScriptLsubspace)):
    print("x=",x_vectors_list[globalMembersOfScriptLsubspace[k]])

#for every x in script L_{q}, calculate the difference between the output of the quadratic form & the output of linear function. 
#They should both give the same answer & so the difference should be zero.

#differenceList is a list of the differences between q(x) and (2 z^T x) mod 4 for all members of script L_{q}
differenceList = []

print("\nCOMPARING THE VALUES OF q CALCULATED USING 1) THE QUADRATIC FORM & 2) THE HIDDEN LINEAR FUNCTION (z)")

for k in range(len(globalMembersOfScriptLsubspace)):
    difference = q[globalMembersOfScriptLsubspace[k]] - ((2*calculate_inner_product_between_z_and_x(globalMembersOfScriptLsubspace[k]))%4)
    print("x=",x_vectors_list[globalMembersOfScriptLsubspace[k]],"   q[quadratic form](x)=",q[globalMembersOfScriptLsubspace[k]],"   q(z,x)=",(2*calculate_inner_product_between_z_and_x(globalMembersOfScriptLsubspace[k]))%4)
    differenceList.append(difference)

for i in range(len(differenceList)):
    differenceList[i] = float(differenceList[i])

print("\nDifferences between calculating q using 1) the quadratic form and 2) the hidden linear function (2 z^{T} x mod 4) for all elements of script L_{q} are shown below. They should all be zero, indicating that the hidden linear function actually calculates q(x):")
#If the quantum circuit is working correctly, the line below should only print zeroes.
print(differenceList)
