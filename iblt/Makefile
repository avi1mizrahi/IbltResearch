

SOURCES = iblt.cpp murmurhash3.cpp utilstrencodings.cpp pyblt.cpp
OBJECTS = $(SOURCES:.cpp=.o)

TARGET = pyblt

CC = g++ -O3 -std=c++20

# -I/usr/local/Cellar/boost/1.60.0_2/include
FLAGS := $(shell python-config --includes)

all: $(TARGET) search

test: pyblt
	$(CC) $(FLAGS) -fPIC libpyblt.so iblt_test.cpp -o iblt_test

search: search.cpp
	$(CC) $(FLAGS) search.cpp -Wno-c++11-extensions -o search

clean:
	rm -f $(OBJECTS) $(TARGET) libpyblt.so search iblt_test

%.o: %.cpp iblt.h
	$(CC)  $(FLAGS) -fPIC -c  $<

pyblt: $(OBJECTS)
	g++ $(OBJECTS) $(FLAGS) -fPIC -shared -o libpyblt.so

