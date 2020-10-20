if [ "$1" = "--cleanbuild" ] ; then
  echo "Deleting old directories..."
  rm -r dist ;
  rm -r build ;
  rm -r submodlib.egg-info ;
  echo "Uninstalling submodlib..."
  pip uninstall submodlib ;
  echo "Building submodlib..."
  python setup.py sdist bdist_wheel ;
  echo "Installing submodlib..."
  pip install dist/*.whl
elif [ "$1" = "--deploy" ] ; then
  if twine check dist/* ; then
    #if [ "$1" = "--test" ] ; then
    #twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    echo "Deploying to TestPyPi..."
    python3 -m twine upload --repository testpypi dist/* --verbose
    #else
    #twine upload dist/* ;
  fi
fi
