#!/bin/bash
cd CheckPoints
montage -resize 1000x1000 -quality 100 -geometry +4+4 */img/$1.png $2/$1.png
cd ..

#convert */img/$1.png -adjoin $2/$1.png
# RUN THE FOLLOWING CODE IN AN IPYTHON SHELL
#import os
#from features import *
#for att in plots:
#    os.system('./combine_images.sh {} {}'.format(att, filename))
