FROM julia

RUN apt-get update \
 && apt-get install -y sudo

RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo 
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker
RUN sudo apt-get update && \
    sudo apt-get install -y xorg-dev mesa-utils xvfb libgl1 freeglut3-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev
ADD *.toml /TabularRL.jl/

RUN cd /TabularRL.jl && DISPLAY=:0 xvfb-run -s '-screen 0 1024x768x24' julia --color=yes --project=/TabularRL.jl -e "using Pkg; Pkg.precompile()"
#RUN  DISPLAY=:0 xvfb-run -s '-screen 0 1024x768x24' julia --color=yes --project=/TabularRL.jl -e 'using Pkg; Pkg.instantiate()' && echo "BUILD_SUCCESSFUL=true" >> ~/buildlog.txt

#RUN  DISPLAY=:0 xvfb-run -s '-screen 0 1024x768x24' julia --color=yes --project=/TabularRL.jl -e 'using Pkg; Pkg.test()' && echo "TESTS_SUCCESSFUL=true" >> ~/buildlog.txt
CMD DISPLAY=:0 xvfb-run -a -s '-screen 0 1024x768x24' julia --color=yes --project=/TabularRL.jl -e "using Pkg; Pkg.test()" &&  echo TESTS_SUCCESSFUL >> ~/testlog.txt