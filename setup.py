from setuptools import setup, find_namespace_packages

setup(
    name='myfinrl',
    version='0.0.1',
    packages=find_namespace_packages(include=["biostrand.*"]),
    #packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List your project dependencies here
    ],
    entry_points={
        'console_scripts': [
            'your_script_name = your_package.module:main_function',
        ],
    },
    author='Dieter Devlaminck',
    author_email='ddvlamin@gmail.com',
    description='',
    url='',
)