"""
Setup configuration for CHiPSAL Multimodal Hate Speech Detection

This package provides tools for detecting hate speech in Devanagari-scripted
multimodal memes using cross-modal attention fusion.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='chipsal-multimodal-hate-detection',
    version='1.0.0',
    author='Samir Thapa',
    author_email='samir@example.com',
    description='Cross-Modal Attention Fusion for Nepali Hate Speech Detection in Multimodal Memes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/chipsal-multimodal-hate-detection',
    project_urls={
        'Paper': 'https://github.com/yourusername/chipsal-multimodal-hate-detection/blob/main/PaperPortion/FINAL_paper_report.md',
        'Bug Tracker': 'https://github.com/yourusername/chipsal-multimodal-hate-detection/issues',
        'Documentation': 'https://github.com/yourusername/chipsal-multimodal-hate-detection/tree/main/docs',
        'Source Code': 'https://github.com/yourusername/chipsal-multimodal-hate-detection',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'notebooks', 'scripts', 'docs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Natural Language :: Hindi',  # Closest to Nepali
    ],
    keywords=[
        'hate-speech-detection',
        'multimodal-learning',
        'cross-modal-attention',
        'devanagari',
        'nepali-nlp',
        'low-resource-languages',
        'meme-analysis',
        'social-media-moderation',
        'clip',
        'bge-m3',
        'fusion-networks',
        'dynamic-gating',
        'transformer',
        'deep-learning',
        'pytorch',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'isort>=5.12.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'ipywidgets>=8.1.0',
            'matplotlib>=3.7.2',
            'seaborn>=0.12.2',
        ],
        'viz': [
            'matplotlib>=3.7.2',
            'seaborn>=0.12.2',
            'plotly>=5.16.1',
            'tensorboard>=2.14.0',
        ],
        'all': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'isort>=5.12.0',
            'jupyter>=1.0.0',
            'ipywidgets>=8.1.0',
            'matplotlib>=3.7.2',
            'seaborn>=0.12.2',
            'plotly>=5.16.1',
            'tensorboard>=2.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'chipsal-train=abiral.train:main',
            'chipsal-infer=abiral.inference:main',
            'chipsal-generate-submission=abiral.generate_submission:main',
            'chipsal-experiments=abiral.experiments:main',
        ],
    },
    include_package_data=True,
    package_data={
        'abiral': ['*.yaml', '*.json'],
    },
    zip_safe=False,
    license='MIT',
    platforms='any',
)
