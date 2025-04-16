#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include "VidHeader.h"

void print_error(const std::string& msg) {
    fprintf(stderr, "Error: %s\n", msg.c_str());
}

void print_usage() {
    fprintf(stderr, "Usage: vid_extractor <input_vid_file> <frame_number | properties>\n");
    fprintf(stderr, "  <frame_number>: Outputs raw RGB pixel data for the specified frame (0-indexed) to stdout.\n");
    fprintf(stderr, "  properties: Outputs 'width height num_frames' to stdout.\n");
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage();
        return 1;
    }

    char* vid_filename = argv[1];
    std::string command = argv[2];

    VidHeader* vid = nullptr;
    try {
        vid = new VidHeader(vid_filename, stderr);
        if (vid->videoFile == NULL) {
             return 1;
        }
         if (vid->width <= 0 || vid->height <= 0 || vid->num_frames <= 0) {
              print_error("VidHeader initialization failed (invalid dimensions/frames). Check previous errors.");
              delete vid;
              return 1;
         }
    } catch (const std::exception& e) {
        print_error("Exception during VidHeader creation: " + std::string(e.what()));
        if (vid) delete vid;
        return 1;
    } catch (...) {
        print_error("Unknown exception during VidHeader creation.");
         if (vid) delete vid;
        return 1;
    }


    if (command == "properties") {
        fprintf(stdout, "%d %d %d\n", vid->width, vid->height, vid->num_frames);
    } else {
        int frame_number = -1;
        try {
             frame_number = std::stoi(command);
        } catch (const std::invalid_argument& e) {
             print_error("Invalid frame number provided.");
             print_usage();
             delete vid;
             return 1;
        } catch (const std::out_of_range& e ) {
             print_error("Frame number out of range for integer.");
             print_usage();
             delete vid;
             return 1;
        }


        if (frame_number < 0 || frame_number >= vid->num_frames) {
            print_error("Frame number " + std::to_string(frame_number) + " is out of range (0 to " + std::to_string(vid->num_frames - 1) + ").");
            delete vid;
            return 1;
        }

        if (!vid->readFrameFromVideoFile(frame_number)) {
            print_error("Failed to read frame " + std::to_string(frame_number) + ".");
            delete vid;
            return 1;
        }

        size_t bytes_to_write = (size_t)vid->width * vid->height * 3;
        size_t bytes_written = fwrite(vid->pixmap, 1, bytes_to_write, stdout);

        if (bytes_written != bytes_to_write) {
            print_error("Failed to write all pixel data to stdout.");
            delete vid;
            return 1;
        }
    }

    delete vid;
    return 0;
}