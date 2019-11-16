all:
    # install pycocotools locally
	python setup.py build_ext --inplace
	mv ./csc/*.so ./
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	python setup.py build_ext build develop
	rm -rf build
