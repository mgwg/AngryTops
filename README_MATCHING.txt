Matching Code README:
====================================================================================================================================================================================================================
====================================================================================================================================================================================================================
TO DOWNLOAD OFF GIT & CHANGE TO THE MATCHING CODE BRANCH:
====================================================================================================================================================================================================================
0. Create a directory somewhere in your directory, mkdir YOUR_FOLDER_NAME
1. Enter that directory, cd YOUR_FOLDER_NAME/
2. Initialize a(n empty) git repo, git init
3. Clone the repository, git clone https://github.com/IMFardz/AngryTops.git
4. Fetch all remote branches, git fetch https://github.com/IMFardz/AngryTops.git
5. Change to the matching code branch, git checkout matching_Kuunal
NOTE:
	-The Matching code requires training data, and needs to be run in the ..CheckPoints/SOME_SUB_DIRECTORY/ directory
	-the CheckPoints/ directory is not available on the git, one should create this oneself before running the training code, or alternativelly to copy it from the kmathani directory
		-to copy from one directory into another from terminal, use the scp -r command
		i.e. scp -r /home/kmahtani/AngryTops/CheckPoints/ /home/YOURNAME/AngryTops/
		-the -r prefix is required for copying directories, and is optional if copying individual files
====================================================================================================================================================================================================================
TO USE:
====================================================================================================================================================================================================================
0. Open up a Screen
1. Change to the following directory:
	home/YOURNAME/AngryTops/AngryTops/
2. Run the following example code:
	bash make_comparisons28.sh /home/YOURNAME/AngryTops/CheckPoints/SOME_SUB_DIRECTORY/ "pxpypzEM" > /home/YOURNAME/AngryTops/CheckPoints/SOME_SUB_DIRECTORY/NAME_OF_LOG.txt
this means:
	"Run the BASH script make_comparisons28.sh"
	"The first input into the BASH script is the path "/home/YOURNAME/CheckPoints/SOME_SUB_DIRECTORY/" "
	"The second input into the BASH script is the representation "pxpypzEM" "
	"Save the Results Printed to the Terminal into the file NAME_OF_LOG.txt in the directory /home/YOURNAME/CheckPoints/SOME_SUB_DIRECTORY/"	
====================================================================================================================================================================================================================
NOTES:
====================================================================================================================================================================================================================
	-The only differences between all the make_comparisons*.sh files in my (kmahtani's) directory are the versions of the "bW_comparisons*.py" matching scripts called upon
	-As of May 2021, the latest version of the matching script is "bW_comparisons9_7.py"
	-It is reccomended to store the image outputs into the subdirectory where the image outputs of the training is located (within the "../CheckPoints/" directory)
	-It is reccomended to ensure that the lines printed to terminal are stored in a .txt file, as the lines printed in terminal contain data about the percentage of fully reconstructable jets, fully
         reconstructed jets, etc.
        -It is reccomended to run the script in a screen
	-The potting code takes about 70 minutes to run (On data trained on a CNN which has been augmented by a factor of 10, ~4*10^6 events)
====================================================================================================================================================================================================================
Details about parts of code:
====================================================================================================================================================================================================================
Lines 30-59:
- Defining function to make TLorentzVector Objects
Lines 62 - 91 (and - 99):
- Importing data and re-shaping array into appropriate size
- reccomended to play around with this in a seperate python2 window
Lines 101 - 126:
- Seperating data into appropriate arrays
Lines 129 - 196:
- Defining Histograms 
Lines 198 - 262:
- Defining constants and objects for following loop
Lines 269 - 300:
- Making TLorentzVector objects for each jet and for Observed Jets
Lines 302 - 304:
- Calculating Missing Transverse Energy for Missing Transverse Energy Distribution Plots
Lines 306 - 327:
- Calculating eta-phi distances for true vs predicted jets
Lines 329 - 404:
- Jet selection, determining whether each jet in the event has been reconstructed, refer to lines 466 - 557
Lines 406 - 446:
- Calculating eta-phi distances for true vs observed jets
Lines 448 - 463:
- Calculating phi-distances for true vs observed W-leptonic jets
Lines 465 - 487:
- Determining how many jets were reconstructable in the event
Lines 489 - 557:
	Lines 489 - 511:
	- For fully reconstructable events (events with 4/4 truth vs observed jets reconstructed), filling the histograms for each jet and determining whether the event was reconstructed, partially
          reconstructed, or not reconstructed/unreconstructed
	Lines 512 - 534:
	- For partially reconstructable events (events with 3/4 truth vs observed jets reconstructed), filling the histograms for each jet and determining whether the event was reconstructed, partially
          reconstructed, or not reconstructed/unreconstructed
	Lines 535 - 557:
	- For partially reconstructable events (events with 3/4 truth vs observed jets reconstructed), filling the histograms for each jet and determining whether the event was reconstructed, partially
          reconstructed, or not reconstructed/unreconstructed
Lines 558 - 567:
- Filling the Missing E_T and jets values into appropriate arrays
Lines 581 - 597:
- Determining percentage of jets reconstructable for each b hadronic, b leptonic, W hadronic and W leptonic jets
- This step shows that the issue lies within the W_leptonic jets at the moment
Lines 598 - 607:
- Filling in histograms for each jet and for missing et values
Lines 609 - 639:
- Calculating the numbers and percentages of jets fully reconstructable, partially reconstructable, fully reconstructed, partially reconstructed, ... etc.
Lines 640 - 783:
- Plotting data and saving plots













====================================================================================================================================================================================================================
====================================================================================================================================================================================================================

MASTER README BASICS (pretty much identical to pekka_readme.txt in Pekka's folder, just copied and pasted here for reference & convenience):
====================================================================================================================================================================================================================
Setting up

0. Log onto Huron.
1. Clone the AngryTop project in the appropriate branch(four_vector_alt is the latest) from https://github.com/IMFardz/AngryTops
2. Add the following to the .bashrc file in the home directory:
        source /usr/local/packages/root/bin/thisroot.sh
        export PYTHONPATH=$PYTHONPATH:'/home/YOURNAME/AngryTops/'
3. Run the 2 lines of command or log out and log in.
====================================================================================================================================================================================================================
Training the network:

1. Run ipython
2. Change the directory to YOURNAME/AngryTops/AngryTops
3. Run the following example code:
        from ModelTraining import train_simple_model
        train_simple_model.train_model("MODEL_NAME", "PLOT_TITLES", "CSV_DATA_FILENAME.csv", scaling='SCALING_TYPE', rep='REPRESENTATION', EPOCHES=INTEGER, sort_jets=False, load_model=False, log_training=True)
this means:
	"Run the train_simple_model script, training on MODEL_NAME, with plot titles PLOT_TITLES, using data from CSV_DATA_FILENAME.csv"
        "Ensure that the training is run with SCALING_TYPE, and using REPRESENTATION, and train to INTEGER number of epoches."
        (i.e.)
        train_simple_model.train_model("BDLSTM_model", "May1", "Feb9.csv", scaling='minmax', rep='pxpypzEM', EPOCHES=25, sort_jets=False, load_model=False, log_training=True)
4. Depending on the version of the code, the output may be directed to a log file, instead of showing in the terminal.
5. The csv file must be in /home/YOURNAME/AngryTops/csv/
6. Results are saved at /home/YOURNAME/AngryTops/CheckPoints/
====================================================================================================================================================================================================================
Plotting after the training:

1. Change the directory to YOURNAME/AngryTops/AngryTops
2. Run the following example code:
        bash make_plots.sh /home/YOURNAME/AngryTops/CheckPoints/March13_cnn25_mm_SJ ptetaphiEM "March15"

this means:
	"Run the BASH script make_plots.sh"
	"The first input into the BASH script is the path "/home/YOURNAME/AngryTops/CheckPoints/SOME_SUBDIRECTORY/" "
	"The second input into the BASH script is the representation "REPRESENTATION" "
	"The third input into the BASH script is the label for the plots "PLOT_DATE" "
====================================================================================================================================================================================================================
Converting Root to csv file:

1. Change the directory to YOURNAME/AngryTops/AngryTops
2. Run the following example code:
        python2 EventGeneration/root2csv.py filelistname.txt test_May.csv
3. Depending on the version of the code, the output may be directed to a log file, instead of showing in the terminal.
4. If the csv file is used for training, make sure to relocate the them to /home/YOURNAME/AngryTops/csv/
====================================================================================================================================================================================================================
Other tips:

1. Use "tmux" to run time-consuming scripts, in case of losing connection.
2. Use "scp" to transfer files/folder to the local machine. Example:
        scp -r xxu@huron:AngryTops/AngryTops/plots_Jan29 ~/Google\ Drive/UT/PHY\ 479/
3. Use "rsync" to transfer files/folder from the local machine. Example:
        rsync -av ~/Documents/GitHub/AngryTops/AngryTops/ModelTraining/train_simple_model.py xxu@huron:AngryTops/AngryTops/ModelTraining/.