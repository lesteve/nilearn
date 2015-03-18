#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

create_new_venv() {
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to be in control of the numpy version
    # we are using for example through apt-get install
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install nose
}

print_conda_requirements() {
    # Echo a conda requirement string for example
    # "pip nose python='2.7.3' scikit-learn='*'". It has a hardcoded
    # list of possible packages to install and looks at _VERSION
    # environment variables to know whether to install the package and
    # if yes which version to install. For example:
    #   - for numpy, NUMPY_VERSION is used
    #   - for scikit-learn, SCIKIT_LEARN_VERSION is used
    TO_INSTALL_ALWAYS="pip nose"
    REQUIREMENTS="$TO_INSTALL_ALWAYS"
    TO_INSTALL_MAYBE="python numpy scipy matplotlib scikit-learn"
    for PACKAGE in $TO_INSTALL_MAYBE; do
        PACKAGE_VERSION_VARNAME="${PACKAGE^^}_VERSION"
        # replace - by _, needed for scikit-learn for example
        PACKAGE_VERSION_VARNAME="${PACKAGE_VERSION_VARNAME//-/_}"
        # dereference $PACKAGE_VERSION_VARNAME to figure out the
        # version to install
        PACKAGE_VERSION="${!PACKAGE_VERSION_VARNAME}"
        if [ -n "$PACKAGE_VERSION" ]; then
            REQUIREMENTS="$REQUIREMENTS $PACKAGE=$PACKAGE_VERSION"
        fi
    done
    echo $REQUIREMENTS
}

create_new_conda_env() {
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    REQUIREMENTS=$(print_conda_requirements)
    echo "conda requirements string: $REQUIREMENTS"
    conda create -n testenv --yes $REQUIREMENTS
    source activate testenv

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi
}

if [[ "$DISTRIB" == "ubuntu" ]]; then
    create_new_venv
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-scipy python-nose python-pip python-sklearn

elif [[ "$DISTRIB" == "ubuntu-no-matplotlib" ]]; then
    create_new_venv
    # --no-install-recommends only installs explictly mentioned
    # packages. By default apt-get installs recommended packages and
    # python-matplotlib is recommended by python-sklearn
    # Note python-joblib needs to be added explicity because in 12.04
    # it is marked 'recommends' rather than 'depends' by python-sklearn
    sudo apt-get install --no-install-recommends -qq python-scipy python-nose python-pip python-sklearn python-joblib

elif [[ "$DISTRIB" == "neurodebian" ]]; then
    create_new_venv
    bash <(wget -q -O- http://neuro.debian.net/_files/neurodebian-travis.sh)
    sudo apt-get install -qq python-scipy python-nose python-nibabel python-sklearn

elif [[ "$DISTRIB" == "conda" ]]; then
    create_new_conda_env

else
    echo "Unrecognized distribution ($DISTRIB); cannot setup travis environment."
    exit 1
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

python setup.py install
