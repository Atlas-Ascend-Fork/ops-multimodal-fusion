#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

LOG_HEAD() {
    echo ""
    echo "[INFO] $(date +%Y%m%d-%H%M%S) $*"
}

LOG_ERROR() {
    echo "[ERROR] $(date +%Y%m%d-%H%M%S) $*" >&2
}

LOG_INFO() {
    echo "[INFO] $(date +%Y%m%d-%H%M%S) $*"
}

LOG_DO() {
    echo "[Command] $(date +%Y%m%d-%H%M%S) $*"
    "$@"
}

DP_ASSERT_CHECK_SKIP() {
    local actual_value=${1}
    local assert_msg=${2}
    if [ "${actual_value}" -eq 0 ] || [ "${actual_value}" -eq 200 ]; then
        LOG_HEAD "${assert_msg} is success"
    else
        LOG_ERROR "${assert_msg} is failed. Exit code: ${actual_value}"
        exit 1
    fi
}

CHECK_ENV_VAR() {
    local var_name=${1}
    local var_value=${!var_name}
    if [[ -z "${var_value}" ]]; then
        LOG_ERROR "Environment variable ${var_name} is not set"
        exit 1
    fi
}

CHECK_ENV_VAR task_name
CHECK_ENV_VAR WORKSPACE
CHECK_ENV_VAR GIT_TARGET_BRANCH

LOG_HEAD "package_name: ${package_name}"
LOG_HEAD "task_name: ${task_name}"
LOG_HEAD "WORKSPACE: ${WORKSPACE}"
LOG_HEAD "GIT_TARGET_BRANCH: ${GIT_TARGET_BRANCH}"
LOG_HEAD "GE_ST_RT2: ${GE_ST_RT2}"

ASCEND_3RD_LIB_PATH="/home/jenkins/opensource"

if [ -f "/home/jenkins/Ascend/cann/bin/setenv.bash" ]; then
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
elif [ -f "/home/jenkins/Ascend/latest/bin/setenv.bash" ]; then
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/latest
else
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/ascend-toolkit/latest
fi

if [ -d "/home/jenkins/Ascend/ascend-toolkit/latest" ]; then
    export ASCEND_INSTALL_PATH="/home/jenkins/Ascend/ascend-toolkit/latest"
elif [ -d "/home/jenkins/Ascend/cann" ]; then
    export ASCEND_INSTALL_PATH="/home/jenkins/Ascend/cann"
else
    export ASCEND_INSTALL_PATH="/home/jenkins/Ascend/latest"
fi

source ${ASCEND_HOME_PATH}/bin/setenv.bash

if [[ "${task_name}" == *ubuntu24* ]]; then
    if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
        echo "Switched to gcc-16"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
        echo "Switched to gcc-15"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
        echo "gcc-16/15 not available, fell back to gcc-14"
    fi
    export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]]; then
    if [[ -f "CMakeLists.txt" ]]; then
        sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    fi
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT:-/dev/null}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT:-/dev/null}"
fi

cd "${WORKSPACE}/" || exit
python3 --version
pip3 --version
pip3 list
set +e
LOG_DO bash build.sh --filelist="${WORKSPACE}/pr_filelist.txt"
BUILD_EXIT_CODE=$?
set -e
DP_ASSERT_CHECK_SKIP "${BUILD_EXIT_CODE}" "build"

if [[ -d "dist" ]]; then
    cd dist || exit
    ls
    cd "${WORKSPACE}/" || exit
    mkdir -p build_out
    cp dist/*.whl build_out/ 2>/dev/null || true
fi
