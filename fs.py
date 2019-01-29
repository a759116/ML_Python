def forwardSelection(x, sl):
    row, column = x.shape
    minIndex = -2
    k = column

    #loop through k-1 size combinations starting at 0 (i)
    for i in range(0, k+1):
        #print("i: " + str(i) + " | X" + str(minIndex) + " added to model" +  " | " + str(x[0]))
        if i > 1 and minIndex != -1:
            selected = np.column_stack((selected[:,], x[:,minIndex]))
            x = np.delete(x, minIndex, 1)
        elif i == 1 and minIndex != -1:
            selected = x[:,minIndex]
            x = np.delete(x, minIndex, 1)
        
        minPval = 1
        minIndex = -1
        #loop through columns k-i times (j) starting at 0
        for j in range(0, k-i):
            if i > 0:
                obj_OLS = sm.OLS(y, np.column_stack((selected[:,], x[:,j]))).fit()
            else:
                obj_OLS = sm.OLS(y, x[:,j]).fit()
 
            pVal = obj_OLS.pvalues[-1].astype(float)
            print(obj_OLS.pvalues)
            #print("----X" + str(j) + ": p-value = " + str(pVal) + " | Min P-Value = " + str(minPval) + "-----")
            if pVal < sl:
                if pVal < minPval:
                    minPval = pVal
                    minIndex = j
    return selected
    
SL = 0.05
X_sig = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
X_ModeledForward = forwardSelection(X_sig, SL)