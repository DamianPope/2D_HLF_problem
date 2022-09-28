import pennylane as qml

"""
For a given value of the parameter n (that the user specifies), this code implements the quantum circuits associated with all the 2**n possible values of the vector b. (This vector is part of the quadratic form q(x).)
For each b value, the code checks if the difference between q(x) and the hidden linear function, (2 z^{T}x) mod 4, is zero for every x value within the subspace script L_{q}.
If the difference is always zero for every b vector, then the circuit is actually solving the 2D Hidden Linear Function problem defined in the paper.
"""

#User sets the value of the parameter n.
#Note that n = N**2, where the grid graph associated with the problem is an NxN graph.
#The two values below are the two simplest cases of the problem.
#n = 4
n = 9

#variable that stores all the possible b vectors for a given n value 
b_vectors_list=[]

#generate all possible b vectors
for i in range(2**n):
    binaryString = bin(i)
    
    #remove the leading "0b"
    binaryString = binaryString[2:]
    
    #if necessary, add leading zeroes to make the length of binaryString equal to n
    if len(binaryString) < n:
        for ii in range(n - len(binaryString)):
            binaryString = '0' + binaryString 
    
    b_vectors_list.append(binaryString)

#define N, the number of vertices on each side of the grid graph associated with the problem
N = int(pow(n,0.5))

dev = qml.device("default.qubit",wires = n,shots=1)

#initialize the 2D list that stores elements of the matrix A
A = [[0]*n for i in range(n)]

#generate A
for K in range(n):
    for L in range(n):
        #for each matrix element, calculate which two vertices in the NxN grid graph that the matrix element is associated with.
        #The first argument of A corresponds to one of the two vertices. The second argument corresponds to the other vertex.
        firstGridColumn =  K % N
        firstGridRow = int(K/N)
    
        secondGridColumn =  L % N
        secondGridRow = int(L/N)

        #test to see if the two vertices are joined by a HORIZONTAL edge; if so, set the A element to 1
        if (firstGridRow==secondGridRow and (abs(firstGridColumn-secondGridColumn)==1)):    
            A[K][L] = 1 

        #test to see if the two vertices are joined by a VERTICAL edge; if so, set the A element to 1
        elif firstGridColumn==secondGridColumn and abs(firstGridRow-secondGridRow)==1:
            A[K][L] = 1

        #if the two vertices aren’t joined by any edge, set the A matrix element to zero
        else: 
            A[K][L] = 0

#define the quantum circuit that solves the problem by generating z, the hidden bit string that's part of the hidden linear function
@qml.qnode(dev)
def circuit(b):
                      
    #implement Hadamard gates on the data register
    for i in range(n):
        qml.Hadamard(wires=i)

    '''implement the controlled Z gates using a circuit with a constant depth of four---no matter how large n is

    First, implement the gates in FIRST layer of the CZ gates
 
    Note that that each gate in this layer acts on separate qubits.
    So, in principle, we could implement all of the gates simultaneously (the same property holds for all the gates in the other three layers)'''
    
    for gridRow in range(N):
        #don't include the rightmost grid column as there are no vertices to the right of it to draw edges to
        for gridColumn in range(0,N-1):    
            #for EVEN grid rows, execute CZ gates in all the EVEN columns (left vertices of horizontal edges)
            #for ODD grid rows, execute CZ gates in all the ODD columns (left vertices of horizontal edges)
            if ((gridRow%2 == 0 and gridColumn%2==0) or (gridRow%2 == 1 and gridColumn%2==1)):
                qml.CZ(wires=[gridColumn+gridRow*N,gridColumn+gridRow*N+1])
    
    #implement the gates in SECOND layer of the CZ gates
    for gridRow in range(N):
        for gridColumn in range(0,N-1):
            if ((gridRow%2 == 0 and gridColumn%2==1) or (gridRow%2 == 1 and gridColumn%2==0)):
                qml.CZ(wires=[gridColumn+gridRow*N,gridColumn+gridRow*N+1])

    #implement the gates in THIRD layer of the CZ gates
    for gridColumn in range(N):
        for gridRow in range(0,N-1):
            if ((gridColumn%2 == 0 and gridRow%2==0) or (gridColumn%2 == 1 and gridRow%2==1)):
                qml.CZ(wires=[gridColumn+gridRow*N,gridColumn+(gridRow+1)*N])

    #implement the gates in FOURTH layer of the CZ gates
    for gridColumn in range(N):
        for gridRow in range(0,N-1):
            if ((gridColumn%2 == 0 and gridRow%2==1) or (gridColumn%2 == 1 and gridRow%2==0)):    
                qml.CZ(wires=[gridColumn+gridRow*N,gridColumn+(gridRow+1)*N])

    #implement controlled S gates on the data register
    for i in range(n):
        #use classical control for the controlled S gates---this reduces the number of qubits needed for the circuit
        if (int(b[i])==1):
            qml.PhaseShift(4.7123889,wires=i) 
            
    #implement the second set of Hadamard gates on the data register
    for i in range(n):
        qml.Hadamard(wires=i)

    #read out the data register to get z, the hidden n-bit string that defines the hidden linear function
    return qml.sample(wires=range(n))

#
#Below are four functions used in the process of verifying that the circuit is working correctly
#

#This function converts a decimal integer into an equivalent string of bits
def convert_to_binary_string(i):
    temp = bin(i)

    #remove the leading ‘0b’ at the start of the string
    tempTwo = temp[2:]

    #add leading zeroes if len(tempTwo) < n in order to make len(tempTwo) = n
    if len(tempTwo) < n:
        for i in range(n - len(tempTwo)):
            tempTwo = '0' + tempTwo 
    return tempTwo

#Function that calculates the inner product between z^{T} and x
def calculate_inner_product_between_z_and_x(k):
    temp = 0

    for i in range(n):
        temp += z[i]*int(x_vectors_list[k][i])
    return temp

#Function that adds two x vectors together in a bitwise fashion, modulo 2
def calculateBitwiseAdditionModulo2(a,b):
    #initialize tempString
    tempString = ""
    
    for i in range(n):
     	tempString = tempString + str((int(x_vectors_list[a][i]) + int(x_vectors_list[b][i])) % 2)
    return tempString

#Function that calculates q for a given x vector (specified by i) using the quadratic form
def calculate_q_using_quadratic_form(i,b):
#i specifies the i^th element of {x}. That is, i has a binary representation that’s equal to x[i]
    quadratic_term = 0
    linear_term = 0

    for alpha in range(0,n):
        linear_term += int(b[alpha])*int(x_vectors_list[i][alpha])

    for beta in range(1,n):
        for alpha in range(0,beta):    
            quadratic_term += A[alpha][beta]*int(x_vectors_list[i][alpha])*int(x_vectors_list[i][beta])    
    return ((2*quadratic_term + linear_term)%4)

#Function that finds which values of x are actually members of the script L_{q} subspace
def findMembersOfScriptLsubspace(potentialIndicesList):
    #initialize list
    membersOfScriptLsubspace = []
    
    #
    #go through all the x’s that MIGHT be in script L_{q} & check if they’re actually in it or not
    #
    for j in range(len(potentialIndicesList)):
        #loop through all 2**n x’ strings and, for each x’ string, test to see if q(x(J)) + q (x’) = q (x(J) ⊕ x’), where J denotes an element of potentialIndicesList
        #Note that ⊕ denotes bitwise addition modulo 2.
        #If the above equality holds for all x’, then x(J) is a member of script L_{q}
        isAMemberOfScriptLsubspaceFlag=True

        for i in range(2**n):
            if ((q[potentialIndicesList[j]] + q[i]) % 4) != q[int(calculateBitwiseAdditionModulo2(potentialIndicesList[j],i),2)]:
                isAMemberOfScriptLsubspaceFlag=False

        if isAMemberOfScriptLsubspaceFlag:
            membersOfScriptLsubspace.append(potentialIndicesList[j])

    return membersOfScriptLsubspace

#
#Execute the quantum circuit for all b possible b vectors & check that the 2 z^{T}x mod 4 is always equal to q(x)
#

#Boolean variable that tracks if all the differences between q(x) & 2 z^{T}x mod 4 calculated are zero
overallDifferenceListFlag=True

#Initialize x_vectors_list, a list that stores all 2**n x vectors
x_vectors_list = []

for j in range(2**n):
    x_vectors_list.append(convert_to_binary_string(j))

for i in range(len(b_vectors_list)):    

    #execute the quantum circuit & store the output in z
    z = circuit(b_vectors_list[i])

    #verify that z actually defines the hidden linear function

    #initialize the list that stores the q values
    q = []

    #initialize the list below
    potentialElementsOfScriptLSubspaceIndices = []

    #Calculate all q(x) values
    for ii in range(2**n):
        #calculate q(x(ii)) using the quadratic form & store it in the list q
        q.append(calculate_q_using_quadratic_form(ii,b_vectors_list[i]))
    
    	#
    	#do a preliminary check to see if x(ii) *might* be an element of the script L_{q} subspace
    	#
    
        #potentialElementsOfScriptLSubspaceIndices is a list that stores the indices of each x that *might* be a member of the script L_{q} subspace because q(x) = 0 or 2
        if q[ii] == 0 or q[ii] == 2:
            potentialElementsOfScriptLSubspaceIndices.append(ii) 
    
    #determine which x’s are actually in script  L_{q}
    globalMembersOfScriptLsubspace = findMembersOfScriptLsubspace(potentialElementsOfScriptLSubspaceIndices)

    print("\n***b =",b_vectors_list[i])
    print("Members of script L_{q} subspace=",globalMembersOfScriptLsubspace)
    print("Hidden string (z) =",z)

    #
    #for every x in script L_{q}, calculate the difference between the quadratic form & the hidden linear function. It should be zero.
    #
    differenceList = []
        
    for k in range(len(globalMembersOfScriptLsubspace)):
        difference = q[globalMembersOfScriptLsubspace[k]] - ((2*calculate_inner_product_between_z_and_x(globalMembersOfScriptLsubspace[k]))%4)
        differenceList.append(difference)

    differenceListFlag=True

    for iii in range(len(differenceList)):
        differenceList[iii] = float(differenceList[iii])
        
        if differenceList[iii]!=0.0:
            differenceListFlag==False
            overallDifferenceListFlag=False

    print("***List of differences between q(x) & (2 z^{T}x) mod 4 =",differenceList)

if overallDifferenceListFlag==True:
    print("\n***All the differences between q(x) & (2 z^{T}x) mod 4 for every b vector is zero.***")
