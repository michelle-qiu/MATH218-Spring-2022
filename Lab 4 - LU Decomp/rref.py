import numpy as np

def rref(B,tol=10**-10):
    A = np.array(B)
    nrow,ncol=A.shape
    
    col = 0
    pivots = []
    for row in range(nrow):
    
        pivotfound = False
    
        while (not pivotfound and col < ncol):
            # The next line finds the row with the max abs val in the column
            maxrow = np.argmax(np.abs(A[row:,col])) + row
            if np.abs(A[maxrow,col])>tol:
                A[row],A[maxrow]=A[maxrow],A[row].copy()
                pivotfound = True
            else: # column is entirely zero
                A[row:,col] = 0 # So make it actually so
                col += 1 # and move on to the next column

        if col == ncol: # we're at the end of the row (i.e. we have a zero row)
            continue    # so move on to the next row
            
        # (row,col) is now a pivot
        pivots.append((row,col))
        A[row] /= A[row,col]
        
        for currow in range(row+1,nrow):
            factor = A[currow,col]
            A[currow] -= factor*A[row]
        
        col += 1
        if col == ncol:
            break
            
    for pivot in reversed(pivots):
        row,col = pivot
        for currow in range(row):
            factor = A[currow,col]/A[row,col]
            A[currow] -= factor * A[row]
        
    return A

