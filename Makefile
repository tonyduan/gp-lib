pkg:
	python3 setup.py sdist bdist_wheel
clean:
	rm -r build dist gp_lib.egg-info
upload:
	twine upload dist/*


