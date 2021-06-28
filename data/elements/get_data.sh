#!/bin/bash

for Z in $(seq 118) 
do
	Z=$(printf "%03d" $Z)
	#wget https://periodictable.com/Elements/$Z/data.html -O data/$Z.html
	tidy -c --ascii-chars yes --numeric-entities yes -asxhtml -o data/$Z.html data/$Z.html
done

