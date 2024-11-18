from setuptools import setup,find_packages

setup(
    name='DIST',
    version='1.0',
    description = 'Denoise and Impute Spatial Transcriptomics',
    long_description = 'Spatially resolved transcriptomics technologies enable comprehensive measurement of gene expression patterns in the context of intact tissues. However, existing technologies suffer from either low resolution or shallow sequencing depth. So, we present DIST, a deep learning-based method that enhance spatial transcriptomics data by self-supervised learning or transfer learning. Through self-supervised learning, DIST can impute the gene expression levels at unmeasured area accurately and improve the data quality in terms of total counts and percentage of dropout. Moreover, transfer learning enables DIST improve the imputed gene expressions by borrowing information from other high-quality data.',
    classifiers = ['Development Status :: 3 - Alpha',
              'Programming Language :: Python :: 3.6'
                  ],
    url = 'https://github.com/zhaoyp1997/DIST.git',
    author = 'Yanping Zhao, Gang Hu',
    author_email = '1786548199@qq.com, huggs@nankai.edu.cn',
    python_requires='>=3.6,<=3.8',
    packages = find_packages(),
    install_requires=['tensorflow<2',
                'numpy<1.17',
                'pandas<1.2',
                #'scipy<1.5',
                #'matplotlib<3.4',
                #'scanpy<1.7',
                #'h5py<3'
                     ]
)