#! /usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)


PROVENANCE_DIR=${SCRIPT_DIR}/provenance

# generate external mesh files
GMSHLB_EXE=${SCRIPT_DIR}/../../../../../../../SPECFEMPP/scripts/gmshlayerbuilder
cd ${GMSHLB_EXE}/../..
uv run scripts/gmshlayerbuilder 3d --depth_block_km 0 "${PROVENANCE_DIR}/interface_files/interfaces.txt" "${PROVENANCE_DIR}/meshfiles"

# decompose mesh
XDECOMPOSE_EXE=${SCRIPT_DIR}/../../../../../../bin/xdecompose_mesh
cd "${PROVENANCE_DIR}"
mkdir -p OUTPUT_FILES
${XDECOMPOSE_EXE} -p xdecompose_par_file
mv "${PROVENANCE_DIR}/OUTPUT_FILES/Database.bin" "${SCRIPT_DIR}/database.bin"
