

from setuptools import setup, find_packages

setup(
    name="hungto",
    version="0.1",
    description="My Research Journey - Cuberoot Reward",
    author="To Tien Hung",
    author_email="totienhung3112@gmail.com",
    packages=find_packages(),
)




"""  
# Installs the package
install(
    name='learn2learn',
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass=cmd_class,
    zip_safe=False,  # as per Cython docs
    version=VERSION,
    description='PyTorch Library for Meta-Learning Research',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    author='Debajyoti Datta, Ian bunner, Seb Arnold, Praateek Mahajan',
    author_email='smr.arnold@gmail.com',
    url='https://github.com/learnables/learn2learn',
    download_url='https://github.com/learnables/learn2learn/archive/' + str(VERSION) + '.zip',
    license='MIT',
    classifiers=[],
    scripts=[],
    setup_requires=['cython>=0.28.5', ],
    install_requires=[
        'numpy>=1.15.4',
        'gym>=0.14.0',
        'torch>=1.1.0',
        'torchvision>=0.3.0',
        'scipy',
        'requests',
        'gsutil',
        'tqdm',
        # 'qpth>=0.0.15',
        #  'pytorch_lightning>=1.0.2',
    ],
)


"""