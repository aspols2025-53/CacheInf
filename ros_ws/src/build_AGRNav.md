sudo apt-get install -y libarmadillo-dev ros-noetic-nlopt libsdl-image1.2-dev libsdl-dev ros-noetic-derived-object-msgs ros-noetic-ackermann-msgs

git clone  https://github.com/fmtlib/fmt.git
cd fmt
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..

git clone https://github.com/strasdat/Sophus.git
cd Sophus/
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..


# To remove installed
cd build && xargs sudo rm < install_manifest.txt
