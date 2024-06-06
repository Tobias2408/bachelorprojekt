from setuptools import setup, find_packages

setup(
    name="preprocess_ai",

version="0.2.8",
   author="Your Name",
    author_email="your_email@example.com",
    description="A package for preprocessing data for AI models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/preprocess_ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'openai==0.28.0',
        'python-dotenv==1.0.1',
        'scipy==1.13.0',
    ],
)
