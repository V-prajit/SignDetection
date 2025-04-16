CC = clang
CXX = clang++

TARGET = vid_extractor

ZLIB_DIR = zlib/zlib-1.2.3
INC_DIRS = -I. -I$(ZLIB_DIR)

CPPFLAGS = $(INC_DIRS)
CFLAGS = -O2 -Wall -Wextra
CXXFLAGS = -O2 -Wall -Wextra -std=c++11 -Wno-writable-strings
LDFLAGS =
LDLIBS =

CXX_SRCS = vid_extractor.cpp VidHeader.cpp read-frame.cpp bayer.cpp showErrMsgBox.cpp
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

C_SRCS = \
	$(ZLIB_DIR)/adler32.c \
	$(ZLIB_DIR)/compress.c \
	$(ZLIB_DIR)/crc32.c \
	$(ZLIB_DIR)/deflate.c \
	$(ZLIB_DIR)/gzio.c \
	$(ZLIB_DIR)/infback.c \
	$(ZLIB_DIR)/inffast.c \
	$(ZLIB_DIR)/inflate.c \
	$(ZLIB_DIR)/inftrees.c \
	$(ZLIB_DIR)/trees.c \
	$(ZLIB_DIR)/uncompr.c \
	$(ZLIB_DIR)/zutil.c

C_OBJS = $(notdir $(C_SRCS:.c=.o))

# All Object Files
OBJS = $(CXX_OBJS) $(C_OBJS)

# Default Target: Build the executable
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)
	@echo "Build complete: $(TARGET)"

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(C_OBJS): %.o: $(ZLIB_DIR)/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Target to clean up build files
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) $(OBJS)
	@echo "Cleanup complete."

.PHONY: all clean