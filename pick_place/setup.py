from setuptools import setup

package_name = 'pick_place'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='UR5e ACT imitation learning: demo collection and policy execution',
    license='TODO',
    entry_points={
        'console_scripts': [
            'data_collector = pick_place.data_collector:main',
            'act_inference = pick_place.act_inference:main',
        ],
    },
)
