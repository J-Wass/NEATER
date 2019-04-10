import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='NEATER',  

     version='1.0',

     scripts=['neat.py','gene.py','genome.py'] ,

     author="Jonah Wasserman",

     author_email="jonahwasserman97@gmail.com",

     description="NeuroEvolution of Augment Topologies Extended Revision (NEAT ER)",

     long_description=long_description,

   long_description_content_type="text/markdown",

     url="https://github.com/lechosenone/NEATER",

     packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

 )
