#!/bin/bash
cd CheckPoints
convert */img/$1.png +append All/$1.png
cd ..


# RUN THE FOLLOWING CODE IN AN IPYTHON SHELL
#import os
#for att in attributes:
#    os.system('./combine_images.sh {}'.format(att))
