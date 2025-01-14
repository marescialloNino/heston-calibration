CC = g++
CFLAGS = -O3 -march=native -Wall
CXXFLAGS += -std=c++17

SRCS = main.cpp metropolis.cpp heston.cpp pso.cpp heston_fft.cpp
OBJS = $(SRCS:.cpp=.o)

all: heston_project

heston_project: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o heston_project $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o heston_project
