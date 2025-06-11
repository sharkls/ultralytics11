#! /bin/bash
rm ../data/*
rm ../param/*

# mkdir ../data/
# mkdir ../param/

fastddsgen CDataBase.idl -d ../data -replace
fastddsgen CMultiModalSrcData.idl -d ../data -replace
fastddsgen CRadarSrcData.idl -d ../data -replace
fastddsgen CAlgResult.idl -d ../data -replace

