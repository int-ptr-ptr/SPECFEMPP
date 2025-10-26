#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}/provenance

# trace back to root dir, which should have a .git folder
ROOT_DIR=${SCRIPT_DIR}
has_git=1

while [ $has_git -ne 0 ]
do
# one folder up
ROOT_DIR=$( cd -- ${ROOT_DIR}/.. &> /dev/null && pwd )

# another termination condition: got to /
if [ ${#ROOT_DIR} -le 1 ]
then
echo "Could not find project root!"
exit 1
fi

# see if .git folder is in this directory
ls ${ROOT_DIR}/.git &> /dev/null
has_git=$?
done


# see if gmshlayerbuilder script is there
ls ${ROOT_DIR}/scripts/gmshlayerbuilder &> /dev/null
if [ $? -ne 0 ]
then
echo 'Could not find gmshlayerbuilder in "'${ROOT_DIR}'/scripts"'
exit 1
fi

python ${ROOT_DIR}/scripts/gmshlayerbuilder simple_dg_topo.dat MESH
if [ $? -ne 0 ]
then
echo -e "\n"
echo 'gmshlayerbuilder failed. Have you activated the correct virtual environment?'
exit 1
fi


${ROOT_DIR}/bin/xmeshfem2D -p Par_file
if [ $? -ne 0 ]
then
echo -e "\n"
echo 'xmeshfem2d failed. Is it not present in "'"${ROOT_DIR}"'/bin/xmehsfem2d"?'
exit 1
fi
