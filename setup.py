from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name = 'tradetestlib',
    description = 'A backtesting library for MetaTrader5',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    version = '1.0.0',
    author = 'Jay Alfaras',
    author_email='alfarasjb@gmail.com',
    url = 'https://github.com/alfarasjb/TradeTestLib',
    packages = find_packages(),
    install_requires = [
        'numpy>=1.21.2',
        'pandas>=1.4.4',
        'matplotlib',
        'seaborn',
        'tqdm',
        'MetaTrader5'
    ],
    include_package_data=True,
    python_requires = '>=3.8'
)