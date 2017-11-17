#!/bin/sh
dir="nlp"

if [ "$1" != "" ]; then

	echo "Launching Program $1"
	gcloud compute instances start instance-1

	if [ "$2" != "s" ] && [ "$2" != "S" ]; then
		echo "Waiting 30s so that the instance can start properly."
		sleep 30
		echo "\a"
	fi

	# To let you know that we're done waiting
	echo "Attempting to SCP file to server."
	gcloud compute scp ./$1 instance-1:~/$dir

	echo "SSH and Running File"
	gcloud compute ssh instance-1 --command="cd ./$dir && export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}} && export PATH=/usr/local/cuda-8.0/bin\${PATH:+:\${PATH}} && screen -d -m autorun.sh $1"

	echo "DONE!"
else
	echo "Provide at least the filename of the program you want to run as first argument."
fi