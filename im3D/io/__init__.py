from __future__ import absolute_import

def PIL_read_tiff(fn):
    """
    USEAGE
    ======
        >>> fn = '/path/to/tiff_image.tiff'
        >>> im = read_tiff(fn)
        >>> ... do stuff with im ...
    
    INPUTS
    ======
        fn ---> string; required
                Filename of the image that you want to read
    
    OUTPUTS
    =======
        im ---> numpy array
                The image, formatted as an array
    
    NOTES
    =====
        This requires both Numpy and Python Image Library (PIL)
        to be installed.
    
    """
    # Test to see if the file exists:
    import os
    if not os.path.exists(fn):
        raise IOError("File '%s' does not exist" % fn)
    # Load numpy for access to arrays
    import numpy as np
    # Load the necessary module
    from PIL import Image
    # 
    # Open the file.  This doesn't read the data, just gives you
    # access to the file and its contents
    im_file = Image.open(fn)
    # You can now do things like im_file.size to have it print
    # the dimensions of the image.  You can also use im_file.crop
    # to select a smaller portion of the file to read, which 
    # makes the actual reading from the hard drive faster
    # 
    # Check to see if it is an 6-bit of 16-bit tiff:
    if im_file.mode == 'L':
        nbits = 8
    elif im_file.mode == 'I;16':
        nbits = 16
    # 
    # Go straight to an array if it's 8-bit:
    if nbits == 8:
        im = np.asarray(im_file, dtype=np.uint8)
    # A little more work if it's 16-bit:
    elif nbits == 16:
        # Read the data that is in the file
        im_data = im_file.getdata()
        # Convert the not-very-useful PIL format to an array
        im = np.asarray(im_data, dtype=np.uint16)
        # The data will be in a 1D array so you need to reshape it.
        # You can use the information in im_data.size, but be 
        # because it is in the opposite order of what you want (i.e.
        # it is in (y_size, x_size) and you want (x_size, y_size) so
        # you have to index it with [::-1] to reverse the order
        im_shape = im_data.size[::-1]
        im = im.reshape(im_shape)
    # Return the array:
    return im

def read_tiff(fn):
    """
    USEAGE
    ======
        >>> fn = '/path/to/tiff_image.tiff'
        >>> im = read_tiff(fn)
        >>> ... do stuff with im ...
    
    INPUTS
    ======
        fn ---> string; required
                Filename of the image that you want to read
    
    OUTPUTS
    =======
        im ---> numpy array
                The image, formatted as an array
    
    NOTES
    =====
        This requires both Numpy and Python Image Library (PIL)
        to be installed.
    
    """
    # Test to see if the file exists:
    import os
    if not os.path.exists(fn):
        raise IOError("File '%s' does not exist" % fn)
    # Load the necessary module
    import skimage.io
    skimage.io.use_plugin('freeimage')
    im = skimage.io.imread(fn, plugin='freeimage')
    # Return the array:
    return im


def read(fn):
    """
    USEAGE
    ======
    >>> fn = '/path/to/tiff_image.png'
    >>> im = read(fn)
    >>> ... do stuff with im ...
    
    INPUTS
    ======
    fn : string; required
        Filename of the image that you want to read
    
    OUTPUTS
    =======
    im : numpy array
        The image, formatted as an array
    
    NOTES
    =====
    This requires Scikits-image (aka skimage) and freeimage to be installed.
    
    """
    # Test to see if the file exists:
    import os
    if not os.path.exists(fn):
        raise IOError("File '%s' does not exist" % fn)
    # Load the necessary module
    import skimage.io
    skimage.io.use_plugin('freeimage')
    im = skimage.io.imread(fn, plugin='freeimage')
    # Return the array:
    return im

def write(fn, im, vmin=None, vmax=None, nbits=8):
    """
    USEAGE
    ======
    >>> 
    >>> 
    >>> 
    
    INPUTS
    ======
    fn : string; required
        Filename of the image that you want to read
    
    OUTPUTS
    =======
    im : numpy array
        The image, formatted as an array
    
    NOTES
    =====
    This requires Scikits-image (aka skimage) and freeimage to be installed.
    
    """
    import numpy as np
    # Test to see if the file exists:
    import os
    if not os.path.exists(os.path.dirname(fn)):
        raise IOError("File '%s' does not exist" % os.path.dirname(fn))
    # Load the necessary module
    import skimage.io
    skimage.io.use_plugin('freeimage')
    # Scale the image to either 8 or 16 bits
    scaled_im = np.clip((im - vmin) / (vmax - vmin), 0.0, 1.0)
    if nbits == 16:
        scaled_im = scaled_im * (2**16 - 1)
        scaled_im = scaled_im.astype('uint16')
        #scaled_im = skimage.img_as_uint(scaled_im)
    elif nbits ==8:
        scaled_im = scaled_im * (2**8 - 1)
        scaled_im = scaled_im.astype('uint8')
        #scaled_im = skimage.img_as_ubyte(scaled_im)
    else:
        print("Can't use nbits=%i. Using nbits=8 instead" % nbits)
        scaled_im = scaled_im * (2**8 - 1)
        scaled_im = scaled_im.astype('uint8')
        #scaled_im = skimage.img_as_ubyte(scaled_im)
    # Save the image
    skimage.io.imsave(fn, scaled_im, plugin='freeimage')

