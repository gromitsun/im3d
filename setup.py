import sys, os, stat, site
from distutils.core import setup
from distutils.extension import Extension
# Check for Cython
try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
except:
    print('Cython is required for this install')
    sys.exit(1)

# ==============================================================
# scan a directory for extension files, converting them to
# extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

# ==============================================================
# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [site.PREFIXES[0]+'/include', '.'],   # adding the '.' to include_dirs is CRUCIAL!!
        library_dirs = [site.PREFIXES[0]+'/lib'],
        extra_compile_args = ["-O3", "-fopenmp", "-Wno-maybe-uninitialized"],
        extra_link_args = ["-fopenmp"],
        # libraries = ['/Users/yue/anaconda/lib/libpython2.7.dylib'],
        # libraries = ['python2.7'],
        )

# ==============================================================
# get the list of extensions
extNames = scandir("im3D")
# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]
for ext in extensions:
    ext.cython_directives = {}
    ext.cython_directives["boundscheck"] = False
    ext.cython_directives["wraparound"] = False
    ext.cython_directives["cdivision"] = True
    ext.cython_directives["embedsignature"] = True
    ext.cython_directives["profile"] = False
# ==============================================================
setup(
    name="im3D",
    version='0.1.0',
    author='John Gibbs',
    author_email='jwgibbs@u.northwestern.edu',
    packages=['im3D', 
              'im3D.curvature', 
              'im3D.gradient', 
              'im3D.histogram', 
              'im3D.io', 
              'im3D.isd', 
              'im3D.isd.k1_k2', 
              'im3D.isd.s_c', 
              'im3D.metrics', 
              'im3D.sdf', 
              'im3D.sdf.inplace',
              'im3D.shapes',
              'im3D.smoothing',
              'im3D.smoothing.inplace',
              'im3D.smoothing.inplace.__lessmem',
              'im3D.transform'],
    url=None,
    license='LICENSE.txt',
    description='3D image processing and visualization',
    long_description=open('README.md').read(),
    ext_modules=extensions,
    cmdclass = {'build_ext': build_ext},
)
