# Starter code for CS 165B HW2 Winter 2023

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    D = training_input[0][0]   # D value
    N1 = training_input[0][1]  # N1 value
    N2 = training_input[0][2]  # N2 value
    N3 = training_input[0][3]  # N3 value
  
    C1 = [[0 for x in range(D)] for y in range(N1)]    # Array for Only class 1 points
    C2 = [[0 for x in range(D)] for y in range(N2)]    # Array for Only class 2 points
    C3 = [[0 for x in range(D)] for y in range(N3)]    # Array for Only class 3 points
    
    Centroid1 = [0 for x in range(D)]      # Array for centroid of class 1
    Centroid2 = [0 for x in range(D)]      # Array for centroid of class 2
    Centroid3 = [0 for x in range(D)]      # Array for centroid of class 3
    
    MCentroid12 = [0 for x in range(D)]      # Array for centroid middle of class 1/2
    MCentroid13 = [0 for x in range(D)]      # Array for centroid middle of class 1/3
    MCentroid23 = [0 for x in range(D)]      # Array for centroid middle of class 2/3
    
    Vector12 = [0 for x in range(D)]      # Array for vector of class 1/2
    Vector13 = [0 for x in range(D)]      # Array for vector of class 1/3
    Vector23 = [0 for x in range(D)]      # Array for vector of class 2/3
    
    o12 = 0      # int for orthogonal equation class 1/2
    o13 = 0      # int for orthogonal equation class 1/3 
    o23 = 0      # int for orthogonal equation class 2/3
    
    for x in range(1, N1+1):               # Filling Array for Only class 1 points
        C1[x-1] = training_input[x] 
    
    for x in range(N1+1, N1+N2+1):         # Filling Array for Only class 2 points
        C2[x-N1-1] = training_input[x]
        
    for x in range(N1+N2+1, N1+N2+N3+1):   # Filling Array for Only class 3 points
        C3[x-N2-N1-1] = training_input[x]
        
    for x in range(0, N1):                 # sum of each data point C1             
        for y in range(0, D):
            Centroid1[y] += C1[x][y]
    
    for x in range(0, N2):                 # sum of each data point C2 
        for y in range(0, D):
            Centroid2[y] += C2[x][y]
            
    for x in range(0, N3):                 # sum of each data point C3 
        for y in range(0, D):
            Centroid3[y] += C3[x][y]
    
    for y in range(0, D):
        Centroid1[y] = Centroid1[y] / N1   # mean of each data point C1 for centroid
        Centroid2[y] = Centroid2[y] / N2   # mean of each data point C2 for centroid
        Centroid3[y] = Centroid3[y] / N3   # mean of each data point C3 for centroid
    
    
    for y in range(0, D): 
        MCentroid12[y] = (Centroid1[y] + Centroid2[y]) / 2   # centroid middle of class 1/2
        MCentroid13[y] = (Centroid1[y] + Centroid3[y]) / 2   # centroid middle of class 1/3
        MCentroid23[y] = (Centroid2[y] + Centroid3[y]) / 2   # centroid middle of class 2/3

    for y in range(0, D): 
        Vector12[y] = Centroid1[y] - Centroid2[y]   # vector of class 1/2
        Vector13[y] = Centroid1[y] - Centroid3[y]   # vector middle of class 1/3
        Vector23[y] = Centroid2[y] - Centroid3[y]   # vector middle of class 2/3
    '''    
    for y in range(0, D): 
        Vector12[y] = Centroid2[y] - Centroid1[y]   # vector of class 1/2
        Vector13[y] = Centroid3[y] - Centroid1[y]   # vector middle of class 1/3
        Vector23[y] = Centroid3[y] - Centroid2[y]   # vector middle of class 2/3
    '''
    for y in range(0, D): 
        o12 += Vector12[y] * MCentroid12[y]   # orthogonal of class 1/2
        o13 += Vector13[y] * MCentroid13[y]   # orthogonal of class 1/3
        o23 += Vector23[y] * MCentroid23[y]   # orthogonal of class 2/3
    '''    
    print(MCentroid12)
    print(MCentroid13) 
    print(MCentroid23)
    
    print(Vector12)
    print(Vector13) 
    print(Vector23) 
    
    print(o12)
    print(o13)
    print(o23)
    '''
    # Done with training data
    # Linear classifier equations are 
    # Vector12[0]*x, [1]*y, ...[D-1]*point = o12. <o12 means negative  >o12 means positiveA78
    
    testD = testing_input[0][0]   # D value
    testN1 = testing_input[0][1]  # N1 value
    testN2 = testing_input[0][2]  # N2 value
    testN3 = testing_input[0][3]  # N3 value
    temp = 0
    Estimated = [0 for x in range(testN1 + testN2 + testN3)]
    spot = 1
    for x in range(1, testN1+testN2+testN3+1):   #Loop that estimates testing datas class and puts it in array estimated
        temp = 0
        for y in range(0, testD):
            temp += Vector12[y]*testing_input[x][y]
            
        if temp >= o12: # A or C
            temp = 0
            for y in range(0, testD):
                temp += Vector13[y]*testing_input[x][y]
                
            if temp >= o13: #A
                #print("1 ", temp)
                #spot+= 1
                temp = 0
                Estimated[x-1] = 1
                
            else:           #C
               # print("3 ", temp)
               # spot += 1
                temp = 0
                Estimated[x-1] = 3
            
        else:            # B or C
            temp = 0
            for y in range(0, testD):
                temp += Vector23[y]*testing_input[x][y]
                
            if temp >= o23: #B
              #  print("2 ", temp)
               # spot+= 1
                temp = 0
                Estimated[x-1] = 2
                    
            else:           #C
              #  print("3 ", temp)
                #spot+= 1
                temp = 0
                Estimated[x-1] = 3
            
    
    #print(Estimated)
    
    tpr1 = tpr2 = tpr3 = 0
    fpr1 = fpr2 = fpr3 = 0
    tnr1 = tnr2 = tnr3 = 0
    fnr1 = fnr2 = fnr3 = 0
    tprate = tprate1 = tprate2 = tprate3 = 0
    fprate = fprate1 = fprate2 = fprate3 = 0      
    error_rate = error_rate1 = error_rate2 = error_rate3 = 0
    accuracy = accuracy1 = accuracy2 = accuracy3 = 0
    precision = precision1 = precision2 = precision3 = 0
    total = testN1+testN2+testN3
    

    
    
    for x in range(0, total):  #tpr tnr fpr fnr for class 1
        if Estimated[x] == 1:
            if x < testN1:
                tpr1 += 1
            else:
                fpr1 += 1
           
        else:
            if x < testN1:
                fnr1 += 1
            else:
                tnr1 += 1 
    
    error_rate1 = (fpr1 + fnr1) / total
    accuracy1 = (tpr1 + tnr1) / total
    precision1 = tpr1 / (tpr1 + fpr1)
    tprate1 = tpr1 / (tpr1 + fnr1)
    fprate1 = fpr1 / (fpr1 + tnr1)
    
            
    for x in range(0, total):  #tpr tnr fpr fnr for class 2
        if Estimated[x] == 2:
            if testN1 <= x < (testN2 + testN1):
                tpr2 += 1
            else:
                fpr2 += 1
           
        else:
            if testN1 <= x < (testN2 + testN1):
                fnr2 += 1
            else:
                tnr2 += 1
                
    error_rate2 = (fpr2 + fnr2) / total
    accuracy2 = (tpr2 + tnr2) / total
    precision2 = tpr2 / (tpr2 + fpr2)
    tprate2 = tpr2 / (tpr2 + fnr2)
    fprate2 = fpr2 / (fpr2 + tnr2)            
                
    for x in range(0, total):  #tpr tnr fpr fnr for class 3
        if Estimated[x] == 3:
            if (testN2 + testN1) <= x:
                tpr3 += 1
            else:
                fpr3 += 1
           
        else:
            if (testN2 + testN1) <= x:
                fnr3 += 1
            else:
                tnr3 += 1      
    
    error_rate3 = (fpr3 + fnr3) / total
    accuracy3 = (tpr3 + tnr3) / total
    precision3 = tpr3 / (tpr3 + fpr3)
    tprate3 = tpr3 / (tpr3 + fnr3)
    fprate3 = fpr3 / (fpr3 + tnr3)
    
   # print (tpr1, " ", tnr1, " ", fpr1, " ", fnr1,)
   # print (tpr2, " ", tnr2, " ", fpr2, " ", fnr2,)
   # print (tpr3, " ", tnr3, " ", fpr3, " ", fnr3,)
    
    tprate = (tprate1 + tprate2 + tprate3)/3
    fprate = (fprate1 + fprate2 + fprate3)/3 #CORRRECT
    error_rate = (error_rate1 + error_rate2 + error_rate3)/3
    accuracy = (accuracy1 + accuracy2 + accuracy3)/3
    precision = (precision1 + precision2 + precision3)/3
    
    return {
                "tpr": round(tprate, 5),
                "fpr": round(fprate, 5),
                "error_rate": round(error_rate, 5),
                "accuracy": round(accuracy, 5),
                "precision": round(precision, 5)
            }
    
    
    
    # TODO: IMPLEMENT
    pass

#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    print(run_train_test(training_input, testing_input))

