from glob import glob
from setuptools import find_packages, setup
setup(name='scout_global_traversability',version='0.1.0',
      packages=find_packages(exclude=['test','tests']),
      data_files=[('share/ament_index/resource_index/packages',['resource/scout_global_traversability']),
                  ('share/scout_global_traversability',['package.xml','LICENSE','HT_MODEL_LICENSE.txt','HT_MODEL_SOURCES.json']),
                  ('share/scout_global_traversability/models',glob('models/*')),
                  ('share/scout_global_traversability/config',glob('config/*.yaml')),
                  ('share/scout_global_traversability/launch',glob('launch/*.py'))],
      install_requires=['setuptools','numpy'],zip_safe=True,
      maintainer='Unmanned Systems Lab',maintainer_email='maintainers@unmanned-systems-lab.invalid',
      description='Incremental tiled HT inference and bounded global/local visualization',
      license='Apache-2.0; vendored HT model: Proprietary research code',
      entry_points={'console_scripts':['global_traversability = scout_global_traversability.node:main']})
