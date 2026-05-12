from setuptools import setup, find_packages

setup(
    name="gym_torcs",
    version="0.0.1",
    description="OpenAI Gym wrapper for TORCS racing simulator",
    author="gym_torcs contributors",
    packages=find_packages(),
    install_requires=[
        "gym==0.15.4",
        "numpy>=1.16,<1.20",
        "opencv-python==4.5.5.64",
    ],
    python_requires=">=3.6,<3.9",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)