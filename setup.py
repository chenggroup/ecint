import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecint",
    version="0.0.1",
    author="Jingfang Xiong, Yunpei Liu, Yongbin Zhuang",
    author_email="jingfangxiong@gmail.com, scottryuu@outlook.com, robinzhuang@outlook.com",
    description="Electrochemical Interficial simulation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xjf729/ecint",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'aiida-core==1.0.0b6',
        'aiida-cp2k==1.0.0b3',
        'ase'
        ]
    python_requires='>=3.6',
)
