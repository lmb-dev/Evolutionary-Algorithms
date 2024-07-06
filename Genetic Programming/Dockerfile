# Do not change the following line . It specifies the base image which
# will be downloaded when you build your image .

FROM pklehre/ec2024-lab2

# Add all the files you need for your submission into the Docker image ,
# e . g . source code , Java bytecode , etc . In this example , we assume your
# program is the Python code in the file abc123 . py . For simplicity , we
# copy the file to the / bin directory in the Docker image . You can add
# multiple files if needed .

ADD lxb150.py /bin

# Install all the software required to run your code . The Docker image
# is derived from the Debian Linux distribution . You therefore need to
# use the apt - get package manager to install software . You can install
# e . g . java , python , ghc or whatever you need . You can also
# compile your code if needed .

# Note that Java and Python are already installed in the base image .

RUN apt-get update
RUN apt-get -y install python-pip
RUN pip install sexpdata==1.0.0

# The final line specifies your username and how to start your program .
# Replace abc123 with your real username and python / bin / abc123 . py
# with what is required to start your program .

CMD ["-username" , "lxb150" , "-submission" , "python /bin/lxb150.py"]
