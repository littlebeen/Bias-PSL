from distutils.core import  setup
from Cython.Build import cythonize
import numpy as np
#python setup.py install  本机
#python setup.py build_ext --inplace 计算环境
setup(
    name='Bias_PSL',
    ext_modules=cythonize('Bias_PSL.pyx'),
    include_dirs=[np.get_include()]
)