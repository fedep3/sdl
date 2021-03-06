# Predicting Adverse Thermal Events in a Smart Building

18-697 Statistical Discovery & Learning
Team 2, Project C3
Host: Rodney Martin

Team Members:
* [Jenna MacCarley](https://github.com/jmaccarl)
* [Federico Ponte](https://github.com/fedep3)
* [Victor Hu](https://github.com/vhu)
* [Dan Yang] (https://github.com/buptyangdan)

## Background

NASA’s Sustainability Base is a green building located in the Ames Research Center at Moffet Field, CA and among the greenest in the federal government. It is LEED platinum certified, achieving the highest rating for sustainability through its innovative design of architecture, resource recycling systems, renewable power generation and advanced sensing systems. The advanced sensing systems in the building are advertised as “anticipating and reacting to changes in sunlight, temperature, wind, and occupancy and will be able to optimize its performance automatically, in real time, in response to internal and external changes”. However, despite these advancements in technology there have been numerous complaints that rooms in the building are “too cold”. Since the building incorporates thousands of sensors, including 2632 thermal sensors specifically, it is difficult to narrow down where this issue of cold complaints may be coming from. Our goal for this research is to use machine learning models to understand what events may be triggering an abnormally cold temperature in the building and be able to predict instances of it. Our research is based off of a project out of Cornell university, which had the same goal of predicting adverse thermal events in this building. Our goal for this project is to expand upon their research, addressing potential limitations in feature selection and incorporating additional log data to improve prediction accuracy.

## Installation 

To install virtual env and the requirements on most linux machines, execute the following commands:

```bash
sudo pip install virtualenvwrapper

#Add to .bashrc
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=/path/to/sdl
source /usr/local/bin/virtualenvwrapper.sh
###

#If only Python 2.7 is installed run:
mkvirtualenv sdl
#Else
mkvirtualenv --python=/path/to/python/2.7 sdl

pip install -r requirements.txt
```

Everytime you open the terminal and wanna work on the project run the following command:

```bash
workon sdl
```

# Running the Software

The software at the time of this writing has 3 modes: regression toolbox mode, detection toolbox mode, and best run mode. The '-t' command line option is used to specify what mode to run, and the optional argument '-r' is used to specify what regression algorithm to use. Use the '-O' python option to suppress debugging output and graph plotting, which is usually necessary when many different configurations are being run:

python [-O] main.py [-t type] [-r reg_alg]

## Regression Toolbox Mode
To run the regression algorithm comparison and generate the corresponding box plot, use the '-t reg' option, which will run the data preprocessing and regression algorithm comparison and display a box plot representing the statistics on the negative mean squared error for observation.

## Detection Toolbox Mode
The detection toolbox mode is designed to search for the ideal parameters for a given regression algorithm, and output the best findings ranked by the AUC value. To run this mode, use the '-t det' option, as well as the '-r' option used to specify which regression algorithm to use in the detection toolbox. For instance, to run the detection toolbox using SVM, use the '-r SVM' option.

## Best Run Mode
This mode is designed to generate final results using testing data. To run it, use the '-t best' option. The best runs are currently hardcoded into this function, including the top results from the top three regression algorithms. The results are ranked based on lowest combined false alarm and missed detection rate


