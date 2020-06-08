# GNU -*- makefile -*-

#VERSION := ${shell python -c "import DLStudio;print DLStudio.__version__"}

VERSION := ${shell python -c "import os; os.chdir('DLStudio'); import DLStudio; print(DLStudio.__version__)"}

default:
	@echo
	@echo "  *** Welcome to DLStudio ${VERSION} ***"
	@echo
	@echo "  docs   -  Build documentation (html)"
	@echo "  help   -  Open the documentation"
	@echo
	@echo "  clean  -  Remove temporary files"
	@echo "  test   -  Run the unittests"
	@echo "  check  -  Look for rough spots"
	@echo "  sdist  -  Build a source distribution tar ball"

docs:
	pydoc -w DLStudio/DLStudio.py

help:
	open DLStudio-${VERSION}.html

clean:
	rm -f *.pyc *~

real-clean: clean
	rm -f MANIFEST  *.html DLStudio-py.info
	rm -rf build dist

# Run the unittest
test:
	@echo
	@echo Testing...
	@echo
	./TestDLStudio/Test.py 

sdist: test
	@echo
	@echo Building a source distribution...
	@echo
	./setup.py sdist --formats=gztar,bztar

# Look for rough spots
check:
	@grep -n FIX *.py *.in PKG-INFO Makefile | grep -v grep
	@echo
	pychecker DLStudio
