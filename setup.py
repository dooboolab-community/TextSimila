import setuptools


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="TextSimila",
    version="0.0.6",
    author="dooboolab",
    author_email="support@dooboolab.com",
    description="Text Similarity Recommendation System",
    long_description = long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/dooboolab/TextSimila",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'ipykernel',
        'gensim==4.2.0',
        'soynlp',
        'nltk',
        'torch',
        'jupyter',
        'pyyaml',
        'argparse'
    ]
)   