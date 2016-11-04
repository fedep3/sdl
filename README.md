# sdl

## Installation 

Install virtual env and the requirements, executes this commands:

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
