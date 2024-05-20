from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='RAGChatbot',
    version='0.1.0',
    description='RAG chatbot system to interact with odf documents in a conversational manner',
    long_description=readme,
    author='João Mata',
    author_email='joao.m.mata@tecnico.ulisboa.pt',
    url='https://github.com/joaommata/Project2024',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')) #??
)