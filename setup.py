from setuptools import setup, find_packages 

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name = 'tradetestlib',
    description = 'A backtesting library for MetaTrader5',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    version = '1.1.0',
    author = 'Jay Alfaras',
    author_email='alfarasjb@gmail.com',
    url = 'https://github.com/alfarasjb/TradeTestLib',
    packages = find_packages(),
    license='MIT',
    install_requires = [
        'numpy>=1.26.0',
        'pandas>=2.0.0',
        'matplotlib>=3.5.2',
        'seaborn>=0.11.2',
        'tqdm>=4.64.1',
        'MetaTrader5>=5.0.45'
    ],
    include_package_data=True,
    python_requires = '>=3.8'
)

