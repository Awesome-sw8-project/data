echo "DO NOT RUN LOCALLY, OR YOU COMPUTER WILL BE SENT TO DESTINATION FUCKED!"

git clone https://github.com/circulosmeos/gdown.pl.git
mkdir train
mkdir test

cd train
./../gdown.pl/gdown.pl https://drive.google.com/file/d/1CScamR90Oc47nn7T8a7lFt14BRdLJYIN/view train.zip
cd ../test
./../gdown.pl/gdown.pl https://drive.google.com/drive/folders/13hPbiE_WInmTu3Za5s9utuHJ34rYi2LE test_reformatted.7z
cd ..

sudo apt install unzip -y
sudo apt install p7zip-full -y

cd train
unzip train.zip
cd ../test
7z x test_reformatted.7z
cd ..

rm -rf gdown.pl
