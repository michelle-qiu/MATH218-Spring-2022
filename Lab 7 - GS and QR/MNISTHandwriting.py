
# coding: utf-8

# In[ ]:


import numpy as np

def hexarraytodecimal(inp):
    # This is used to convert hex arrays to decimal
    # Array with powers of 256 in reverse
    hexconv = [256**i for i in reversed(range(len(inp)))]
    return np.dot(inp,hexconv)

def readimgs(filename):
    # Read MNIST files into numpy arrays
    # See http://yann.lecun.com/exdb/mnist/

    rows = 0
    cols = 0
    
    f = open(filename,'rb')

    # The first four bytes determine file type: 2051 means images, 2049 means labels
    magicnum = int.from_bytes(f.read(4),byteorder='big')
    
    if ( magicnum != 2051 and magicnum != 2049 ):
        print("Filename provided does not contain image data in the required format")
        return
    
    # The next four bytes give the number of images
    numimages = int.from_bytes(f.read(4),byteorder='big')    
    
    data = np.fromfile(f,dtype=np.uint8)
    
    if magicnum==2051:
        # For images files, the next two sets of four bytes give the numbers 
        # of rows and column. So convert the first 8 bytes back to decimal...
        rows = hexarraytodecimal(data[:4])
        cols = hexarraytodecimal(data[4:8])

        # ...and format the rest of the data into images.
        data = data[8:].reshape((numimages,cols,rows))
    
    # otherwise magicnum = 2049, so we have label data, which we don't need
    # to process further.
    
    f.close()
    
    return (data,rows,cols)

