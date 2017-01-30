FROM  lofar-wsclean


## Copy in CALIB script
COPY ["CALIB.py", "${INSTALLDIR}/pipeline_script"]

RUN sudo apt-get -y update && \
    sudo apt-get -y install wget bzip2 time && \
    echo deb http://dev.monetdb.org/downloads/deb/ trusty monetdb >  /etc/apt/sources.list.d/monetdb.list && \
    echo deb-src http://dev.monetdb.org/downloads/deb/ trusty monetdb >>  /etc/apt/sources.list.d/monetdb.list && \
    wget --output-document=- https://www.monetdb.org/downloads/MonetDB-GPG-KEY | sudo apt-key add - && \
    sudo apt-get -y update && \
    sudo apt-get -y install monetdb5-sql monetdb-client  

RUN echo user=monetdb > ${HOME}/.monetdb && \
    echo password=monetdb >> ${HOME}/.monetdb && \
    monetdbd create ${INSTALLDIR}/monetdb/monetdb-lofar-farm && \
    monetdbd start ${INSTALLDIR}/monetdb/monetdb-lofar-farm && monetdb create gsm && \
    monetdb release gsm && \
    mclient -d gsm -s "CREATE user gsm WITH PASSWORD 'msss' NAME 'GlobalSkyModel2' SCHEMA sys" && \
    mclient -d gsm -s 'CREATE SCHEMA "gsm" AUTHORIZATION "gsm"' && \
    mclient -d gsm -s 'ALTER USER "gsm" SET SCHEMA "gsm"' && \
    monetdbd set port=51000 ${INSTALLDIR}/monetdb/monetdb-lofar-farm 

COPY ./gsm.sql.bz2 /root/gsm.sql.bz2

RUN cd /root/ && bunzip2 /root/gsm.sql.bz2 && \
    echo user=gsm > ${HOME}/.monetdb && \
    echo password=msss >> ${HOME}/.monetdb && \
    monetdbd start ${INSTALLDIR}/monetdb/monetdb-lofar-farm && monetdb start gsm && mclient -lsql -p 51000 --database=gsm ${HOME}/gsm.sql && \
    echo \#\! /usr/bin/env bash > ${INSTALLDIR}/monetdb/start_db_script && \
    echo monetdbd start ${INSTALLDIR}/monetdb/monetdb-lofar-farm >> ${INSTALLDIR}/monetdb/start_db_script && \
    echo monetdb start gsm >> ${INSTALLDIR}/monetdb/start_db_script && \
    rm -f gsm.sql && \
    sudo sed -i.orig 's/^\( *\)db_host = .*/\1db_host = "localhost"/g' ${INSTALLDIR}/lofar/bin/gsm.py && \
    echo \#\! /usr/bin/env bash > ${INSTALLDIR}/monetdb/stop_db_script && \
    echo monetdb stop gsm >> ${INSTALLDIR}/monetdb/stop_db_script && \
    echo  monetdbd stop ${INSTALLDIR}/monetdb/monetdb-lofar-farm >> ${INSTALLDIR}/monetdb/stop_db_script && \
    echo sleep 10 >> ${INSTALLDIR}/monetdb/stop_db_script && \
    # want to make sure the farm stopped after this script
    echo \#\! /usr/bin/env bash > ${INSTALLDIR}/lofar/bin/gsm_wrapper && \
    echo sudo ${INSTALLDIR}/monetdb/start_db_script >> ${INSTALLDIR}/lofar/bin/gsm_wrapper && \
    echo gsm.py \$\* >> ${INSTALLDIR}/lofar/bin/gsm_wrapper && \
    echo  sudo ${INSTALLDIR}/monetdb/stop_db_script >> ${INSTALLDIR}/lofar/bin/gsm_wrapper && \
    chmod 744 ${INSTALLDIR}/monetdb/start_db_script && \
    chmod 744 ${INSTALLDIR}/monetdb/stop_db_script && \
    chmod 755 ${INSTALLDIR}/lofar/bin/gsm_wrapper

    
# No write access for lock file in ${INSTALLDIR}/monetdb/. Need to check if we can relocate Check
# Then the next step is to see if we can make a stop script
# last step is to write a wrapper around gsm

