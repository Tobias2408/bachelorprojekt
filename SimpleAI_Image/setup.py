# Update setup.py to version 0.1.8
from setuptools import setup, find_packages

setup(
    name='SimpleAI_Image',
    version='0.1.8',  # Incremented version number
    description='A package for processing and storing image vectors in PostgreSQL',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tobias Poulsen',
    author_email='your.email@example.com',
    url='https://github.com/Tobias2408/bachelorprojekt',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'sqlalchemy',
        'matplotlib',
        'tensorflow-macos',
        'tensorflow-datasets',
        'tensorflow-metal',
        'scikit-learn',
        'requests',
        'keras',
        'tensorboard',
        'pgvector',
        'psycopg2-binary'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

