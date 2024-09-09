#! /bin/bash
service ssh start

ulimit -c unlimited

echo "Please leave this terminal open, go to a new terminal window and run ./start_docker.sh"
pip install dill
#chmod -R  777 /tmp/ray/
tail -f /dev/null

