from setuptools import setup
import os
from glob import glob

package_name = 'complete_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A simple laser scan publisher and subscriber',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'complete_publisher = complete_node.complete_publisher:main',
            'complete_subscriber = complete_node.complete_subscriber:main',
        ],
    },
)

