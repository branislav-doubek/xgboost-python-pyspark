FROM centos:7

# Install gcc 4.8
RUN yum -y install libmpc-devel mpfr-devel gmp-devel zlib-devel bzip2 curl
RUN yum -y  install gcc openssl-devel bzip2-devel libffi-devel zlib-devel xz-devel 
RUN yum -y groupinstall "Development Tools"

RUN curl https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz --output /tmp/Python-3.7.0.tgz
WORKDIR /tmp
RUN tar xzf Python-3.7.0.tgz 
WORKDIR /tmp/Python-3.7.0
RUN ./configure --enable-optimizations 
RUN yum install make -y
RUN make altinstall
RUN yum install which -y
WORKDIR /tmp
RUN rm -r Python-3.7.0.tgz
RUN yum -y install epel-release

# Install cmake 3.8
WORKDIR /home
RUN curl -O https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz
RUN tar zxvf cmake-3.8.0.tar.gz \
	&& cd cmake-3.8.0 \
	&& ./bootstrap --prefix=/usr/local \
	&& make -j4 \
	&& make install

# Install maven and git
RUN yum -y install maven
RUN yum -y install http://opensource.wandisco.com/centos/7/git/x86_64/wandisco-git-release-7-2.noarch.rpm && \
	yum -y install git

# clone and build xgboost
WORKDIR /home
#RUN git clone --recursive https://github.com/dmlc/xgboost && \
#	cd xgboost && \
#	git checkout tags/v0.90 && \
#	git submodule update --init --recursive && \
#	make -j4 && \
#	cd jvm-packages && \
#    mvn package

# clone this project, cp jar files
#WORKDIR /home
#RUN git clone https://github.com/ShandiLin/xgb_demo.git
#RUN mkdir /home/xgb_demo/pyspark_xgb/jars
#WORKDIR /home/xgb_demo/pyspark_xgb/jars
#RUN cp /home/xgboost/jvm-packages/xgboost4j/target/xgboost4j-0.90*.jar ./xgboost4j-0.90.jar
#RUN cp /home/xgboost/jvm-packages/xgboost4j-spark/target/xgboost4j-spark-0.90*.jar ./xgboost4j-spark-0.90.jar
RUN git clone --recursive https://github.com/dmlc/xgboost && \
	cd xgboost && \
	git checkout tags/v0.82 && \
	git submodule update --init --recursive && \
	make -j4 &&\
	cd jvm-packages && \
    mvn package -DskipTests

WORKDIR /home
RUN git clone https://github.com/branislav-doubek/xgboost-python-pyspark.git
RUN mkdir /home/xgboost-python-pyspark/pyspark_xgb/jars
WORKDIR /home/xgboost-python-pyspark/pyspark_xgb/jars
RUN cp /home/xgboost/jvm-packages/xgboost4j/target/xgboost4j-0.82*.jar ./xgboost4j-0.82.jar
RUN cp /home/xgboost/jvm-packages/xgboost4j-spark/target/xgboost4j-spark-0.82*.jar ./xgboost4j-spark-0.82.jar


# spark
WORKDIR /home
RUN curl -O https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.6.tgz
RUN tar zxvf spark-2.4.0-bin-hadoop2.6.tgz
RUN mv spark-2.4.0-bin-hadoop2.6 /usr/local/
 
ENV SPARK_HOME /usr/local/spark-2.4.0-bin-hadoop2.6
ENV PATH "$PATH:$SPARK_HOME/bin"
ENV JAVA_HOME /usr/lib/jvm/java
ENV PYSPARK_PYTHON=python3.7

# install pip & python packages
WORKDIR /home
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py
RUN python3.7 -m pip install --upgrade pip
RUN pip install pandas===0.23.4 xgboost===0.82 numpy sklearn optuna pyyaml
ENTRYPOINT [ "/home/xgboost-python-pyspark/pyspark_xgb/start.sh"]
CMD ["score.py"]