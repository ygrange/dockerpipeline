# 
#     Dockerfile to generate the lofar-base Docker image for use with the pipeline demonstrator
#     Copyright (C) 2016  ASTRON (Netherlands Institute for Radio Astronomy)
#     P.O.Box 2, 7990 AA Dwingeloo, The Netherlands
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
# 

#
# base
#
FROM ubuntu:16.04

#
# common-environment
#
ENV USER=lofar
ENV INSTALLDIR=/opt

#
# environment
#
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=2.7

#
# versions
#
ENV CASACORE_VERSION=latest \
    CASAREST_VERSION=latest \
    PYTHON_CASACORE_VERSION=2.1.2 \
    BOOST_VERSION=1.58

#
# set-uid
#
ENV UID=1024

#
# set-build-options
#
# ENV J=58

#
# Base and runtime dependencies
#
#RUN sed -i 's/archive.ubuntu.com/osmirror.rug.nl/' /etc/apt/sources.list 
RUN apt-get update && \
    apt-get install -y python2.7 libpython2.7 && \
    apt-get install -y libopenblas-base libcfitsio-bin libwcs5 libfftw3-bin libhdf5-10 libboost-python${BOOST_VERSION}.0 && \
    apt-get install -y python-pip && \
    pip install numpy && \
    apt-get purge -y python-pip && \
    apt-get autoremove --purge -y && \
    apt-get install -y nano sudo

#
# open security holes (allow smooth user switching, allow sudo)
#
RUN echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    sed -i 's/requiretty/!requiretty/g' /etc/sudoers && \
    chmod a+rw /etc/group /etc/passwd /home

#
# setup install dir
#
RUN mkdir -p ${INSTALLDIR}

#
# *******************
#   Casacore
# *******************
#
RUN apt-get update && apt-get install -y wget git cmake g++ gfortran flex bison libopenblas-dev libfftw3-dev libhdf5-dev libboost-python-dev libcfitsio-dev wcslib-dev && \
    mkdir -p ${INSTALLDIR}/casacore/build && \
    mkdir -p ${INSTALLDIR}/casacore/data && \
    cd ${INSTALLDIR}/casacore && git clone https://github.com/casacore/casacore.git src && \
    if [ "${CASACORE_VERSION}" != "latest" ]; then cd ${INSTALLDIR}/casacore/src && git checkout tags/v${CASACORE_VERSION}; fi && \
    cd ${INSTALLDIR}/casacore/data && wget --retry-connrefused ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && \
    cd ${INSTALLDIR}/casacore/data && tar xf WSRT_Measures.ztar  && rm -f WSRT_Measures.ztar && \
    cd ${INSTALLDIR}/casacore/build && cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/casacore/ -DDATA_DIR=${INSTALLDIR}/casacore/data -DBUILD_PYTHON=True -DENABLE_TABLELOCKING=ON -DUSE_OPENMP=ON -DUSE_FFTW3=TRUE -DUSE_HDF5=ON -DCXX11=YES -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fsigned-char -O2 -DNDEBUG -march=native" ../src/ && \
    cd ${INSTALLDIR}/casacore/build && make -j  && \
    cd ${INSTALLDIR}/casacore/build && make install && \
    bash -c "strip ${INSTALLDIR}/casacore/{lib,bin}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/casacore/{build,src}" && \
    apt-get purge -y wget git cmake g++ gfortran flex bison libopenblas-dev libfftw3-dev libhdf5-dev libboost-python-dev libcfitsio-dev wcslib-dev && \
    apt-get autoremove --purge -y

#
# *******************
#   Casarest
# *******************
#
RUN apt-get update && apt-get install -y git cmake g++ gfortran libboost-system-dev libboost-thread-dev libhdf5-dev libcfitsio3-dev wcslib-dev libopenblas-dev && \
    mkdir -p ${INSTALLDIR}/casarest/build && \
    cd ${INSTALLDIR}/casarest && git clone https://github.com/casacore/casarest.git src && \
    if [ "${CASAREST_VERSION}" != "latest" ]; then cd ${INSTALLDIR}/casarest/src && git checkout tags/v${CASAREST_VERSION}; fi && \
    cd ${INSTALLDIR}/casarest/build && cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/casarest -DCASACORE_ROOT_DIR=${INSTALLDIR}/casacore -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-std=c++11 -O2 -march=native -DNDEBUG" ../src/ && \
    cd ${INSTALLDIR}/casarest/build && make -j  && \
    cd ${INSTALLDIR}/casarest/build && make install && \
    bash -c "strip ${INSTALLDIR}/casarest/{lib,bin}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/casarest/{build,src}" && \
    apt-get purge -y git cmake g++ gfortran libboost-system-dev libboost-thread-dev libhdf5-dev libcfitsio3-dev wcslib-dev libopenblas-dev && \
    apt-get autoremove --purge -y

#
# *******************
#   Pyrap
# *******************
#
RUN apt-get update && apt-get install -y git make g++ python-setuptools libboost-python-dev libcfitsio3-dev wcslib-dev && \
    mkdir ${INSTALLDIR}/python-casacore && \
    cd ${INSTALLDIR}/python-casacore && git clone https://github.com/casacore/python-casacore && \
    if [ "$PYTHON_CASACORE_VERSION" != "latest" ]; then cd ${INSTALLDIR}/python-casacore/python-casacore && git checkout tags/v${PYTHON_CASACORE_VERSION}; fi && \
    cd ${INSTALLDIR}/python-casacore/python-casacore && ./setup.py build_ext -I${INSTALLDIR}/casacore/include/ -L${INSTALLDIR}/casacore/lib/ && \
    mkdir -p ${INSTALLDIR}/python-casacore/lib/python${PYTHON_VERSION}/site-packages/ && \
    mkdir -p ${INSTALLDIR}/python-casacore/lib64/python${PYTHON_VERSION}/site-packages/ && \
    export PYTHONPATH=${INSTALLDIR}/python-casacore/lib/python${PYTHON_VERSION}/site-packages:${INSTALLDIR}/python-casacore/lib64/python${PYTHON_VERSION}/site-packages:$PYTHONPATH && cd ${INSTALLDIR}/python-casacore/python-casacore && ./setup.py install --prefix=${INSTALLDIR}/python-casacore/ && \
    bash -c "find ${INSTALLDIR}/python-casacore/lib -name '*.so' | xargs strip || true" && \
    bash -c "rm -rf ${INSTALLDIR}/python-casacore/python-casacore" && \
    apt-get purge -y git make g++ python-setuptools libboost-python-dev libcfitsio3-dev wcslib-dev && \
    apt-get autoremove --purge -y

#
# *******************
#   QPID client
# *******************
#

# Run-time dependencies
# QPID daemon legacy store would require: libaio1 libdb5.1++
RUN apt-get update && apt-get install -y sasl2-bin libuuid1 libnss3 libnspr4 xqilla libboost-program-options${BOOST_VERSION}.0 libboost-filesystem${BOOST_VERSION}.0

# Install
# QPID daemon legacy store would require: libaio-dev libdb5.1++-dev
RUN apt-get update && apt-get install -y subversion swig ruby ruby-dev python-dev libsasl2-dev pkg-config cmake libtool uuid-dev libxerces-c-dev libnss3-dev libnspr4-dev help2man fakeroot build-essential debhelper libsslcommon2-dev libxqilla-dev python-setuptools libboost-program-options${BOOST_VERSION}-dev libboost-filesystem${BOOST_VERSION}-dev && \
    mkdir /opt/qpid && \
    svn --non-interactive -q co https://svn.astron.nl/LOFAR/branches/LOFAR-Release-2_18/LCS/MessageBus/qpid/ /opt/qpid; \
    /opt/qpid/local/sbin/build_qpid && \
    bash -c "strip /opt/qpid/{bin,lib}/* || true" && \
    bash -c "rm -rf ~/sources" && \
    apt-get purge -y subversion swig ruby ruby-dev python-dev libsasl2-dev pkg-config cmake libtool uuid-dev libxerces-c-dev libnss3-dev libnspr4-dev help2man fakeroot build-essential debhelper libsslcommon2-dev libxqilla-dev python-setuptools libboost-program-options${BOOST_VERSION}-dev libboost-filesystem${BOOST_VERSION}-dev && \
    apt-get autoremove --purge -y
#
# *******************
#   DAL
# *******************
#
RUN  apt-get update && apt-get install -y git cmake g++ swig python-dev libhdf5-dev && \
    mkdir ${INSTALLDIR}/DAL && \
    cd ${INSTALLDIR}/DAL && git clone https://github.com/nextgen-astrodata/DAL.git src && \
    mkdir ${INSTALLDIR}/DAL/build && cd ${INSTALLDIR}/DAL/build && cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/DAL ../src && \
    make -j  && \
    make install && \
    bash -c "find ${INSTALLDIR}/DAL/lib -name '*.so' | xargs strip || true" && \
    bash -c "rm -rf ${INSTALLDIR}/DAL/{src,build}" && \
    apt-get purge -y git cmake g++ swig python-dev libhdf5-dev && \
    apt-get autoremove --purge -y

#
# entry
#
COPY ["bashrc",    "${INSTALLDIR}/"]
COPY ["bashrc.d",  "${INSTALLDIR}/bashrc.d/"]
COPY ["chuser.sh", "/usr/local/bin"]
ENTRYPOINT ["/usr/local/bin/chuser.sh"]

