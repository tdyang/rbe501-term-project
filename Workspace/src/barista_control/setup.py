from setuptools import find_packages, setup

package_name = 'barista_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RBE501 Barista Team',
    maintainer_email='selmansouri@wpi.com',
    description='Barista robot pouring planning and control.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'ur5_kinematics_node = barista_control.ur5_kinematics_node:main',
        ],
    },
)
