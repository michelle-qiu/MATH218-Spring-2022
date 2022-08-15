import numpy as np
def swaprows(A,i,j):
    A[[i, j],:] = A[[j, i],:] 
def rowmult(A,i,c):
    A[i] = A[i]*c
def rowaddmult(A,i,j,c):
    A[j] = A[i]*c + A[j]
def rowred(A):
    rows,cols = A.shape
    copy = A.copy()
    pivotcol = 0
    pivotrow = 0
    i = 1
    while((pivotcol<cols) & (pivotrow<rows)):
        while(i<rows):
            rowaddmult(copy,pivotrow,i,((-1*copy[i,pivotcol])/(copy[pivotrow,pivotcol])))
            i+=1
        pivotcol+=1
        pivotrow+=1
        i = pivotrow+1
    return copy;
def backsub(U,v):
    rows,cols = U.shape
    x = np.zeros(cols)
    for i in range (rows-1,-1,-1):
        x[i] += (U[i, i+1:rows]@x[i+1:rows]) # dot product version 
        x[i] = v[i] - x[i]
        x[i] = x[i]/U[i,i]
    return x;

def fwdsub(L,v):
    rows,cols = L.shape
    x = np.zeros(cols)
    for i in range (0,rows,1):
        x[i] += (L[i, :i]@x[:i]) # dot product version 
        x[i] = v[i] - x[i]
        x[i] = x[i]/L[i,i]
    return x;

def gauss(a,v):
    rows,cols = a.shape #dimensions of the original matrix yet to be augmented
    A = np.zeros(rows*(cols+1)).reshape(rows,cols+1)
    A[:rows,:cols] = a[:,:]
    A[:,cols] = v
    #print(A)
    sol = rowred(A)
    #print(sol)
    newsol = backsub(sol[:cols,:cols],sol[:,cols]) #here we have to separate the REF matrix into its augmented parts.
    return (newsol)
def gausswithpivot(a,v):
    rows,cols = a.shape #dimensions of the original matrix yet to be augmented
    A = np.zeros(rows*(cols+1)).reshape(rows,cols+1)
    A[:rows,:cols] = a[:,:]
    A[:,cols] = v
    #print(A)
    sol = rowredpivot(A)
    print(sol)
    newsol = backsub(sol[:cols,:cols],sol[:,cols]) #here we have to separate the REF matrix into its augmented parts.
    return (newsol)

def rowredpivot(A):
    rows,cols = A.shape
    copy = A.copy()
    pivotcol = 0
    pivotrow = 0
    i = 1
    while((pivotcol<cols) & (pivotrow<rows)):
        while(i<rows):
            maxe = np.argmax(abs(copy[pivotrow:,pivotcol]))+pivotrow
            if (maxe > pivotrow):
                copyrow = (copy[pivotrow]).copy();
                copy[pivotrow] = (copy[maxe]).copy();
                copy[maxe] = copyrow;
            rowaddmult(copy,pivotrow,i,((-1*copy[i,pivotcol])/(copy[pivotrow,pivotcol])))
            i+=1
        pivotcol+=1
        pivotrow+=1
        i = pivotrow+1
    return copy;

def LU(A):
    rows,cols = A.shape
    copy = A.copy()
    pivotcol = 0
    pivotrow = 0
    i = 1
    zero = np.zeros((rows,cols))
    perm = np.eye(rows)
    cool = (copy,zero,perm) #U,L,P
    while((pivotcol<cols) & (pivotrow<rows)):
        while(i<rows):
            maxe = np.argmax(abs(copy[pivotrow:,pivotcol])) +pivotrow
            if (maxe > pivotrow):
                swaprows(perm,maxe,pivotrow)
                swaprows(zero,maxe,pivotrow)
                copyrow = (copy[pivotrow]).copy();
                copy[pivotrow] = (copy[maxe]).copy();
                copy[maxe] = copyrow;
            multval = (-1*copy[i,pivotcol])/(copy[pivotrow,pivotcol])
            rowaddmult(copy,pivotrow,i,(multval))
            zero[i,pivotrow] = -multval
            i+=1
        pivotcol+=1
        pivotrow+=1
        i = pivotrow+1
    np.fill_diagonal(zero,1.)
    return cool

def LUdet(A):
    rows,cols = A.shape
    copy = A.copy()
    pivotcol = 0
    pivotrow = 0
    i = 1
    numswaps =0
    zero = np.zeros((rows,cols))
    perm = np.eye(rows)
    cool = (copy,zero,perm) #U,L,P, 
    while((pivotcol<cols) & (pivotrow<rows)):
        while(i<rows):
            maxe = np.argmax(abs(copy[pivotrow:,pivotcol])) +pivotrow
            if (maxe > pivotrow):
                swaprows(perm,maxe,pivotrow)
                swaprows(zero,maxe,pivotrow)
                numswaps+=1
                copyrow = (copy[pivotrow]).copy();
                copy[pivotrow] = (copy[maxe]).copy();
                copy[maxe] = copyrow;
            multval = (-1*copy[i,pivotcol])/(copy[pivotrow,pivotcol])
            rowaddmult(copy,pivotrow,i,(multval))
            zero[i,pivotrow] = -multval
            i+=1
        pivotcol+=1
        pivotrow+=1
        i = pivotrow+1
    np.fill_diagonal(zero,1.)
    Udet = np.prod(np.diag(copy))
    numswaps = numswaps%2
    Pdet = (-1)**numswaps
    return Udet*Pdet

def det(A):
    rows,cols = A.shape
    i = 0
    totaldet = 0
    if (rows!=cols):
        raise ValueError('The matrix is not square!')
    if (rows == cols ==2):
        return det2(A)
    for j in range (rows):
        m = minor(A,i,j)
        totaldet+= ((-1)**(i+j))*A[i,j]* det(m)
    return totaldet

def minor(A,i,j):
    B = np.delete(A,i, axis=0)
    B = np.delete(B,j, axis=1)
    return B


def LUSolve(A,v):
    U,L,P = LU(A)
    v = P@v
    y = fwdsub(L,v)
    x = backsub(U,y)
    return x