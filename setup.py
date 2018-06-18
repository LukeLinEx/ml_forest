from setuptools import setup

setup(
    name='ml_forest',
    packages=['ml_forest'],
    include_package_data=True,
    install_requires=[
        "numpy",
        "bson"
    ]
)