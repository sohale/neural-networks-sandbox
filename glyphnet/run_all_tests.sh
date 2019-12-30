
scriptDir1=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
scriptDir2="`dirname $0`"
echo 1: $scriptDir1  # absolute resolved
echo 2: $scriptDir2  # relative dir

#source $scriptDir1/../source_tf1.sh

#cd $scriptDir1
python --version
echo "ls>"
ls
echo "<ls"
#python -m unittest -v tests/mln_topology_tests
python tests/mln_topology_tests.py

