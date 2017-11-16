#!/bin/sh
dir="nlp"

if [[ "$1" != "" ]]; then
	echo "Launching Program $1"
	gcloud compute instances start instance-1

	counter=0
	loaded=0
	echo "Attempting to SCP file to server."
	gcloud compute scp ./$1 instance-1:~/$dir/

	while [[ $? -ne 0 ]]; do
		echo "Attempting to SCP file to server."
		gcloud compute scp ./$1 instance-1:~/dir
		if [[ counter -gt 10 ]]; then
			echo "Failed to push file to server."
			loaded=1
			break
		else
			echo "Sleeping... to try again in 5 seconds."
			sleep 5
		fi
		counter+=1
	done

	if [[ loaded -ne 1 ]]; then
		gcloud compute ssh instance-1 --command="cd ./$dir && export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}} && export PATH=/usr/local/cuda-8.0/bin\${PATH:+:\${PATH}} && export PATH=~/bin:\$PATH && autorun.sh $1"
		gcloud compute instances stop instance-1 
	fi

else
	echo "Provide at least the filename of the program you want to run as first argument."
fi
