# python
This is a code hub abaut python with  Webservice and AI and DeepLearning and CNN and Image Processing

本项目全部是python应用脚本或者python的一些学习心得。应用领域包括web框架网络爬虫（搜索）、Webservice、人工智能（语音识别、文字识别、自然语言处理）、深度学习，神经网络，图像处理和跟踪识别算法，大数据分析与可视化，云计算等


windows下安装python库， py -m pip install 库名字 --user
  比如安装pygame， py -m pip install pygame --user

 
 
安装caffe和tensorflow

编译安装时坑太多。

========================================================== caffe安装  CPU版本

参考 http://www.linuxidc.com/Linux/2015-07/120449.htm


用root安装的

1、 基本库 
    sudo apt-get install build-essential 
    sudo apt-get install vim cmake git
    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
     
2、安装cmake
    cmake-3.9.1-Linux-x86_64.tar.gz
    tar -zxvf cmake-3.9.1-Linux-x86_64.tar.gz
    cd cmake-3.9.1-Linux-x86_64
    ./bootstrap 
    make -j16
    make install
    
    将export PATH=/home/xuyong/cmake-3.9.1-Linux-x86_64/bin:$PATH 加入/etc/profile 
    source /ete/profile
    
3、安装opencv2.4.10
    git clone https://github.com/jayrambhia/Install-OpenCV.git
    cd Ubuntu/2.4
    ./opencv2_4_10.sh
    如果网速慢下载不了opencv2.4.10，可以手动下载opencv2_4_10.tar 放到OpenCV目录下，这个看安装脚本可以知道。
    
    安装atlas
    sudo apt-get install libatlas-base-dev 
    
4、编译caffe
    git clone git://github.com/BVLC/caffe.git 直接clone master分支
    cd caffe-master
    cp Makefile.config.example Makefile.config
    修改Makefile.config
        CPU_ONLY := 1 #放开
        PYTHON_INCLUDE := /usr/include/python2.7 \
            /usr/local/lib/python2.7/dist-packages/numpy-1.13.1-py2.7-linux-x86_64.egg/numpy/core/include   # include 这个看你的python是哪个，有的是anaconda
    cd python 
    for req in $(cat requirements.txt); do pip install $req; done # 看下是否所有的库都满足要求
    
    make all -j16
    make test
    make runtest
    make pycaffe
    测试 python -c "import caffe"
    
    然后添加环境变量到 /etc/profile
    
    export PYTHONPATH=/home/xuyong/caffe-master/python:$PYTHONPATH 
    source /etc/profile
    

    ========================================================= tensorflow CPU版本   非root用户安装
    1、先安装anaconda， 不建议用root用户安装
        conda info
        conda -e
        conda create
        conda install
        
    2、去github下载tensorflow对应的版本
    
    3、pip install tensorflow-1.3.0-cp27-none-linux_x86_64.whl 
       
    4、出现的问题   
        OSError: [Errno 1] Operation not permitted: '/usr/local/lib/python2.7/dist-packages/easy_install.pyc'
            It's probably a bad idea to have 777 permissions on any file. Also, since you have pip in ~/.local, try using the --user option to the last pip command
            加上 --user 就好了， 就是anaconda用root用户安装造成的