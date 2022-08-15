import numpy as np

def plotimagelist(l):
    
    import matplotlib.pylab as plt    
    
    numimages = float(len(l))
    cols = np.ceil(np.sqrt(numimages))
    rows = np.ceil(numimages/cols)
    
    fig=plt.figure(figsize=(15,(15./cols)*rows));
    
    for imgnum in range(1,int(numimages)+1):
        _ = fig.add_subplot(rows,cols,imgnum)
        _ = plt.imshow(l[imgnum-1])

def plotgrayimagelist(l):
    
    import matplotlib.pylab as plt    
    
    numimages = float(len(l))
    cols = np.ceil(np.sqrt(numimages))
    rows = np.ceil(numimages/cols)
    
    fig=plt.figure(figsize=(15,(15./cols)*rows));
    
    for imgnum in range(1,int(numimages)+1):
        _ = fig.add_subplot(rows,cols,imgnum)
        _ = plt.imshow(l[imgnum-1],cmap='gray')
        
def showimgRGB(img):
    
    import matplotlib.pylab as plt    
    
    R,G,B=(img[:,:,i] for i in range(3))

    fig=plt.figure(figsize=(15,15));

    _=fig.add_subplot(2,2,1)
    _=plt.imshow(img)
    _=fig.add_subplot(2,2,2)
    _=plt.imshow(R,cmap='Reds')
    _=fig.add_subplot(2,2,3)
    _=plt.imshow(G,cmap='Greens')
    _=fig.add_subplot(2,2,4)
    _=plt.imshow(B,cmap='Blues')
    

def showblockgray(l,height=8,width=8):
    numimages = float(len(l))
    cols = int(np.ceil(np.sqrt(numimages)))
    rows = int(np.ceil(numimages/cols))

    ht,wd = l[0].shape

    totalht = rows*ht
    totalwd = cols*wd
    
    blockmat = np.zeros((totalht,totalwd))
    i = 0
    for row in range(rows):
        for col in range(cols):
            if i < numimages:
                blockmat[row*ht:(row+1)*ht,col*wd:(col+1)*wd] = l[i]
                i += 1
    
    import matplotlib.pylab as plt    

    fig=plt.figure();

    plt.imshow(blockmat,cmap='gray')
    plt.show()
    
    
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    Code from https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    """
    import re
    import numpy as np

    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return (np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width))),int(height),int(width))

def readORLDatabase(folder):
    import glob
    import re

    gotdimension = False
    
    numimages = len(list(glob.iglob(folder + '**/*.pgm', recursive=True)))
    labels = np.zeros(numimages)
    i = 0

    for filename in glob.iglob(folder + '**/*.pgm', recursive=True):
        face,ht,wd = read_pgm(filename)

        if not(gotdimension):
            faces = np.zeros((numimages,ht,wd))
            gotdimension = True
        
        faces[i] = face
        m = re.search('.+(s)([0-9]{1,2}).+',filename)
        labels[i] = int(m.group(2))
        i += 1
    
    return (faces,labels,ht,wd,numimages)


def readYaleDatabase(folder):
    import glob
    import re

    gotdimension = False
    
    numimages = len(list(glob.iglob(folder + '**/*P00A*.pgm', recursive=True)))
    labels = np.zeros(numimages)
    i = 0
    
    for filename in glob.iglob(folder + '**/*P00A*.pgm', recursive=True):

        try:
            face,ht,wd = read_pgm(filename)
            
        except:
            continue

        if not(gotdimension):                
            faces = np.zeros((ht*wd,numimages))
            gotdimension = True

        faces[:,i] = face.flatten()

        m = re.search('.+(yaleB)([0-9][0-9]).+',filename)
        labels[i] = int(m.group(2))
        i +=1

    return (faces,labels,ht,wd,numimages)
