#!/bin/bash
cd CheckPoints
convert */img/$1.png +append $2/$1.png
cd ..


# RUN THE FOLLOWING CODE IN AN IPYTHON SHELL
#import os
#from features import *
#for att in attributes:
#    os.system('./combine_images.sh {} {}'.format(att, filename))
