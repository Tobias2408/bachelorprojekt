from setuptools import setup, find_packages

setup(
    name="BigFlowPrototypes",
    version="0.1",
    author="Your Name",
    author_email="your_email@example.com",
    description="A package for preprocessing data for AI models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BigFlowPrototypes",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'openai',
        'python-dotenv',
        'scipy',
        'tensorflow',
    ],
)
