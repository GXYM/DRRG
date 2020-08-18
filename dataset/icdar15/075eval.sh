rm submit/*
cp $1/*.txt submit
cd submit/;zip -r  submit.zip *;mv submit.zip ../; cd ../
python2 script075.py -g=gt.zip -s=submit.zip
