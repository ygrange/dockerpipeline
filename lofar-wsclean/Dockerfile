FROM lofar-pipeline

ENV WSCLEAN_VERSION 1.12

# install wsclean
# Note: install with LOFAR station response correction
# Note: set CPATH as otherwise, if compiled with LOFAR support, in LOFAR header files 
# #include <measures/Measures/MeasFrame.h> from casacore is unknown
RUN apt-get update && apt-get install -y bzip2 wget subversion cmake g++ gfortran bison flex libhdf5-dev libfftw3-dev libcfitsio3-dev libgsl-dev libboost-date-time${BOOST_VERSION}-dev libboost-filesystem${BOOST_VERSION}-dev libboost-thread${BOOST_VERSION}-dev libboost-system${BOOST_VERSION}-dev && \
    CPATH=${INSTALLDIR}/casacore/include/casacore:$CPATH && \
    mkdir -p ${INSTALLDIR}/wsclean/build && \
    cd ${INSTALLDIR}/wsclean && \
    wget http://downloads.sourceforge.net/project/wsclean/wsclean-${WSCLEAN_VERSION}/wsclean-${WSCLEAN_VERSION}.tar.bz2 && \
    tar xf wsclean-${WSCLEAN_VERSION}.tar.bz2 && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/wsclean -DCMAKE_PREFIX_PATH="${INSTALLDIR}/lofar;${INSTALLDIR}/casacore;${CFITSIO_ROOT_DIR}" -DBUILD_SHARED_LIBS=TRUE ../wsclean-${WSCLEAN_VERSION} && \
    make -j  && \
    make install && \
    rm ${INSTALLDIR}/wsclean/wsclean-${WSCLEAN_VERSION}.tar.bz2 && \
    apt-get purge -y bzip2 wget subversion cmake g++ gfortran bison flex libhdf5-dev libfftw3-dev libcfitsio3-dev libgsl-dev libboost-date-time${BOOST_VERSION}-dev libboost-filesystem${BOOST_VERSION}-dev libboost-thread${BOOST_VERSION}-dev libboost-system${BOOST_VERSION}-dev && \
    apt-get autoremove --purge -y

COPY ["70-wsclean",  "${INSTALLDIR}/bashrc.d/70-wsclean"]
