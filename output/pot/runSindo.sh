#!/bin/bash

. /home/shun/SINDO/sindo/sindovars.sh
export POTDIR=./

sindo < vscf.inp   > vscf.out   2>&1

