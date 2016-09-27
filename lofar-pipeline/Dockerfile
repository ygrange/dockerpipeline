#
#     Dockerfile to generate the lofar-pipeline image that contains the the pipeline demonstrator. Just add data.
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
FROM lofar-base 

ENV AOFLAGGER_VERSION=2.8.0

# Run-time dependencies
RUN apt-get update && apt-get install -y python-xmlrunner python-scipy liblog4cplus-1.1-9 libxml2 libboost-thread${BOOST_VERSION}.0 libboost-filesystem${BOOST_VERSION}.0 libboost-date-time${BOOST_VERSION}.0 libboost-signals${BOOST_VERSION}.0 libpng12-0 libsigc++-2.0-dev libxml++2.6-2v5 libgsl2 openssh-client libboost-regex${BOOST_VERSION}.0 gettext-base rsync python-matplotlib slurm-client && \
    apt-get -y install python-pip python-dev && \
    pip install pyfits pywcs python-monetdb && \
    apt-get -y purge python-pip python-dev  && \
    apt-get -y autoremove --purge

#
# *******************
#   AOFlagger
# *******************
#

RUN apt-get update && apt-get install -y wget cmake g++ libxml++2.6-dev libpng12-dev libfftw3-dev libboost-filesystem${BOOST_VERSION}-dev libboost-date-time${BOOST_VERSION}-dev libboost-signals${BOOST_VERSION}-dev libboost-thread${BOOST_VERSION}-dev libcfitsio3-dev libopenblas-dev && \
    mkdir -p ${INSTALLDIR}/aoflagger/build && \
    bash -c "cd ${INSTALLDIR}/aoflagger && wget --retry-connrefused http://downloads.sourceforge.net/project/aoflagger/aoflagger-${AOFLAGGER_VERSION%%.?}.0/aoflagger-${AOFLAGGER_VERSION}.tar.bz2" && \
    cd ${INSTALLDIR}/aoflagger && tar xf aoflagger-${AOFLAGGER_VERSION}.tar.bz2 && \
    cd ${INSTALLDIR}/aoflagger/build && cmake -DCASACORE_ROOT_DIR=${INSTALLDIR}/casacore/ -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS="--std=c++11 -O2 -DNDEBUG" -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/aoflagger ../aoflagger-${AOFLAGGER_VERSION} && \
    cd ${INSTALLDIR}/aoflagger/build && make -j  && \
    cd ${INSTALLDIR}/aoflagger/build && make install && \
    bash -c "strip ${INSTALLDIR}/aoflagger/{lib,bin}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/aoflagger/{build,aoflagger-${AOFLAGGER_VERSION}}" && \
    bash -c "rm -rf ${INSTALLDIR}/aoflagger/aoflagger-${AOFLAGGER_VERSION}.tar.bz2" && \
    apt-get -y purge wget cmake g++ libxml++2.6-dev libpng12-dev libfftw3-dev libboost-filesystem${BOOST_VERSION}-dev libboost-date-time${BOOST_VERSION}-dev libboost-signals${BOOST_VERSION}-dev libboost-thread${BOOST_VERSION}-dev libcfitsio3-dev libopenblas-dev && \
    apt-get -y autoremove --purge

#
# *******************
#   LOFAR
# *******************
#

# Tell image build information
ENV LOFAR_BRANCH=branches/LOFAR-Release-2_18 \
    LOFAR_REVISION=35323 \
    LOFAR_BUILDVARIANT=gnucxx11_optarch


# Install
RUN apt-get update && apt-get install -y subversion cmake g++ gfortran bison flex liblog4cplus-dev libhdf5-dev libblitz0-dev libboost-dev libboost-python-dev python-dev libxml2-dev pkg-config libpng12-dev libfftw3-dev libunittest++-dev libxml++2.6-dev libgsl-dev libboost-filesystem${BOOST_VERSION}-dev libboost-date-time${BOOST_VERSION}-dev libboost-thread${BOOST_VERSION}-dev libboost-regex${BOOST_VERSION} binutils-dev libcfitsio3-dev wcslib-dev libopenblas-dev && \
    mkdir -p ${INSTALLDIR}/lofar/build/${LOFAR_BUILDVARIANT} && \
    cd ${INSTALLDIR}/lofar && \
    svn --non-interactive -q co -r 35323 -N https://svn.astron.nl/LOFAR/branches/LOFAR-Release-2_18 src; \
    svn --non-interactive -q up src/CMake && \
    cd ${INSTALLDIR}/lofar/build/${LOFAR_BUILDVARIANT} && cmake -DBUILD_PACKAGES=Offline -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/lofar/ -DCASAREST_ROOT_DIR=${INSTALLDIR}/casarest/ -DCASACORE_ROOT_DIR=${INSTALLDIR}/casacore/ -DAOFLAGGER_ROOT_DIR=${INSTALLDIR}/aoflagger/ -DQPID_ROOT_DIR=/opt/qpid/ -DUSE_OPENMP=True ${INSTALLDIR}/lofar/src/ && \
    cd ${INSTALLDIR}/lofar/build/${LOFAR_BUILDVARIANT} && sed -i '29,31d' include/ApplCommon/PosixTime.h && \
    cd ${INSTALLDIR}/lofar/build/${LOFAR_BUILDVARIANT} && make -j && \
    cd ${INSTALLDIR}/lofar/build/${LOFAR_BUILDVARIANT} && make install && \
    bash -c "mkdir -p ${INSTALLDIR}/lofar/var/{log,run}" && \
    bash -c "chmod a+rwx  ${INSTALLDIR}/lofar/var/{log,run}" && \
    bash -c "strip ${INSTALLDIR}/lofar/{bin,sbin,lib64}/* || true" && \
    bash -c "rm -rf ${INSTALLDIR}/lofar/{build,src}" && \
    apt-get purge -y subversion cmake g++ gfortran bison flex liblog4cplus-dev libhdf5-dev libblitz0-dev libboost-dev libboost-python-dev python-dev libxml2-dev pkg-config libpng12-dev libfftw3-dev libunittest++-dev libxml++2.6-dev libgsl-dev libboost-filesystem${BOOST_VERSION}-dev libboost-date-time${BOOST_VERSION}-dev libboost-thread${BOOST_VERSION}-dev binutils-dev libcfitsio3-dev wcslib-dev libopenblas-dev && \
    apt-get autoremove --purge -y

