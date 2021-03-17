echo "DO NOT RUN LOCALLY! YOU COMPUTER MIGHT BREAK DOWN."

git clone https://github.com/circulosmeos/gdown.pl.git
mkdir train

cd train
./../gdown.pl/gdown.pl https://drive.google.com/file/d/1CScamR90Oc47nn7T8a7lFt14BRdLJYIN/view train.zip

sudo apt install unzip -y
unzip train.zip
cd ..

rm -rf gdown.pl
