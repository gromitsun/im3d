#!/opt/local/bin py27-MacPorts
from distutils.core import setup
import glob
# ==============================================================
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# 
# hist_module = Extension(
#     'im3D.histogram', 
#     ['im3D/pyx_src/histogram/histogram.pyx',], 
#     extra_compile_args=['-O3', '-fPIC', '-fopenmp',]
# )
# 
# metrics_module = Extension(
#     'im3D.metrics', 
#     ['im3D/pyx_src/metrics/metrics.pyx',], 
#     extra_compile_args=['-O3', '-fPIC', '-fopenmp',]
# )
# 
# cmdclass = { 'build_ext': build_ext }
# ext_modules = [hist_module, metrics_module]
# ==============================================================
# from Cython.Build import cythonize
# 
# pyx_files = ['im3D/histogram/pyx_src/_cy_hist_.pyx',
#              'im3D/metrics/pyx_src/_cy_metrics_.pyx']
# ext_modules = cythonize(pyx_files)
# ==============================================================
setup(
    name='im3D',
    version='0.1.0',
    author='John Gibbs',
    author_email='jwgibbs@u.northwestern.edu',
    packages=['im3D', 'im3D.histogram', 'im3D.metrics', 'im3D.smoothing'], #
    scripts=glob.glob('bin/*.so'),
    url=None,
    license='LICENSE.txt',
    description='3D image processing and visualization',
    long_description=open('README.txt').read(),
    #cmdclass = cmdclass,
    #ext_modules=ext_modules,
)

