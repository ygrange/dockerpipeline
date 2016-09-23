#!/usr/bin/env bash
#
#     Entry point for the lofar-base Docker image for use with the pipeline demonstrator
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
# Correct UID
export UID=`id -u`

# Configure user
if [ -z "${USER}" ]; then
  export USER=${UID}
fi

# Create home directory
if [ -n "${HOME}" ]; then
  export HOME=/home/${USER}
  mkdir -p $HOME && cd $HOME
fi

# Add user to system
fgrep -q ":x:${UID}:" /etc/passwd || echo "${USER}:x:${UID}:${UID}::${HOME}:/bin/bash" >> /etc/passwd
fgrep -q ":x:${UID}:" /etc/group  || echo "${USER}:x:${UID}:" >> /etc/group

# Set the environment
[ -e /opt/bashrc ] && source /opt/bashrc

# Run the requested command
if [ -z "$*" ]; then
  echo HALLO
  exec sg ${USER} "/bin/bash --rcfile ${INSTALLDIR}/bashrc"
else
  exec sg ${USER} "$@"
fi
