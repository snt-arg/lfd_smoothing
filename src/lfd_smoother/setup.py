from setuptools import setup, find_packages

setup(name='lfd_dadayeh',
      version='1.0',
      author='Alireza Barekatain',
      description='lfd smoothing',
      url='https://github.com/snt-arg/lfd_smoothing',
      packages=find_packages(
          include=['lfd_smoother']),
      package_dir={'lfd_smoother':'..'})