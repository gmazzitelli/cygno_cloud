#!/bin/bash

git config --global user.name "Giovanni Mazzitelli"
git config --global user.email "giovanni.mazzitelli@lnf.infn.it"
git config --global core.excludesfile "/jupyter-workspace/private/cygno_cloud/.gitignore"
git config --global credential.helper store
git pull
