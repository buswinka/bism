import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    description="Biomedical Image Segmentation Models (BISM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.13.0',
        'bism',
        'fastremap',
        'zarr',
        'tqdm',
        'scikit-image',
        'yacs',
        'tensorboard',

    ],
    entry_points={
        'console_scripts': ['bism-train = bism.train.__main__:main',
                            'bism-eval = bism.eval.__main__:main']
    }
)