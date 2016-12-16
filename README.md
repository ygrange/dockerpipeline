# dockerpipeline
Docker files to create a running copy of the LOFAR software, and a script to run a demonstration pipeline.

To run a command in one of the containers, use the syntax as listed on the lofar dockerhub page: 

docker run --rm -u $(id -u) -e USER=$USER -e HOME=$HOME -v $HOME:$HOME lofar-pipeline '<your-command> <arguments>'

