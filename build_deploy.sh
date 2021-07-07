if [ "$1" = "--clean" ] ; then
  echo "Deleting old directories and files..."
  rm -r dist ;
  rm -r build ;
  rm -r submodlib.egg-info ;
  rm submodlib_cpp.cpython-39-x86_64-linux-gnu.so
  echo "Uninstalling submodlib..."
  pip uninstall submodlib ;
fi
if [ "$1" = "--build" ] ; then
  echo "Building submodlib and creating distribution..."
  python setup.py sdist bdist_wheel ;
fi
if [ "$1" = "--install" ] ; then
  echo "Installing submodlib from whl file..."
  pip install dist/*.whl
fi
if [ "$1" = "--devinstall" ] ; then
  echo "Installing submodlib in editable mode..."
  pip install -e .
fi
if [ "$1" = "--deploy" ] ; then
  if twine check dist/* ; then
    #if [ "$1" = "--test" ] ; then
    echo "Deploying to TestPyPi..."
    #python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    python3 -m twine upload --repository testpypi dist/*.tar.gz --verbose
    #python3 -m twine upload --repository testpypi dist/* --verbose
    #else
    #twine upload dist/* ;
  fi
fi
