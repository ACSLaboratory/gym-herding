"""
Setup for the Herding Environment
"""
from setuptools import setup

setup(name='gym_herding',
      version='0.1.1',
      description='Leader-agent herding OpenAI gym environment',
      url='https://github.com/acslaboratory/gym-herding',
      author='Zahi Kakish',
      author_email='zkakish@gmail.com',
      python_requires='>=3',
      install_requires=['gym', 'numpy', 'matplotlib']
)