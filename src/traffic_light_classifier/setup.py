import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'traffic_light_classifier'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Meng shi',
    maintainer_email='mengshi2022@ia.ac.com',
    description='Traffic light classification package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cam_node = traffic_light_classifier.cam_node:main',
            'classification_node = traffic_light_classifier.classification_node:main',
        ],
    },
)
