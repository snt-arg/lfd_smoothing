from setuptools import setup, find_packages

setup(name='lfd_smoother',
      version='1.0',
      author='Alireza Barekatain',
      description='Smoothing motion trajectories learned from demonstration',
      url='https://github.com/snt-arg/lfd_smoothing',
      packages=find_packages(include=['lfd_smoother', 'lfd_smoother.*']),
      )
