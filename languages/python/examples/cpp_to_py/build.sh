g++ -g -o hello.so -shared -fPIC -arch x86_64 $(python3-config --includes) hello.cpp $(python3-config --ldflags) -L/Users/yuqi/anaconda3/lib -lpython3.12 -lpthread -ldl -lutil -lm -undefined dynamic_lookup

nm -g hello.so

otool -L hello.so  # macOS
ldd hello.so       # Linux

