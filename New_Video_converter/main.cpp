#include "VidHeader.h"

int main(int argc, char * argv[])
{
	VidHeader * vid = new VidHeader(argv[1], NULL);

	char outfile[1000];
	//V:\PortableCapture_project\compressed_video\08-18-2010-test\scene1-camera1.vid
	//V:\ASL_lexicon_project\asl-data3\ASL_2010_08_10_Liz_additional_lexicon_signs\scene1-camera1.vid
	int frame_number = 100;
	while( frame_number < vid -> num_frames )
	{
		vid -> readFrameFromVideoFile( frame_number );
		sprintf(outfile,"./frames/%05d.ppm", frame_number);
		//vid -> writePPMimage(outfile, vid -> pixmap);
		frame_number += 100;
		break;
	}

	delete vid;
	return 0;
}

