from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'The package for Home Assistant Integration'
LONG_DESCRIPTION = 'Some long description sazing the same'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="recsys", 
        version=VERSION,
        author="SIS RS 6",
        author_email="<sonyaarom@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["interpret==0.3.0",
                   "lime==0.2.0.1",
                   "matplotlib==3.6.2",
                   "meteostat==1.6.5",
                   "numpy==1.24.1",
                   "pandas==1.5.2",
                   "pytz==2022.7",
                   "scikit_learn==1.2.0",
                   "shap==0.41.0",
                   "statsmodels==0.13.5",
                   "tqdm==4.64.1",
                   "xgboost==1.7.2"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['recsys', 'home assistant'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)