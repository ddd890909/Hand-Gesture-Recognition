
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <Windows.h>
#include "CompressiveTracker.h"
#include "handDetect.h"

using namespace cv;
using namespace std;

//#define DEP

Rect box_RGB,box_DEP; // tracking box
bool drawing_box = false;
bool gotBB = false;	// got tracking box or not
bool fromfile = true; //bool fromfile = false;
string video;

/*
void read_options(int argc, char** argv, VideoCapture& capture)
{
	for (int i=0; i<argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)	// read tracking box from file
		{
			if (argc>i)
			{
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
			{
				print_help();
			}
		}
		if (strcmp(argv[i], "-v") == 0)	// read video from file
		{
			if (argc > i)
			{
				video = string(argv[i+1]);
				capture.open(video);
				fromfile = true;
			}
			else
			{
				print_help();
			}
		}
	}
}
*/

/*
void print_help(void)
{
	printf("welcome to use HandTrack_ddd\n");
	printf("-v source video\n-b tracking box file\n");
}
*/

void draw_frameCount(Mat Frame,int framecount)//Frameframecount
{
    char strFrame[10];
    sprintf_s(strFrame, "#%0d ",framecount);
    putText(Frame,strFrame,cvPoint(30,30),2,1,CV_RGB(25,200,25));
}

void write_featureValue(Mat detectFeatureValue_ddd);

int main(int argc, char * argv[])
{
	VideoCapture capture_RGB, capture_DEP;
	capture_RGB.open("D:\\lab\\KinectRecord\\ddd\\Record_ddd\\data\\ddd_occlusion_RGB.avi"); //capture.open(0);
#ifdef DEP
	capture_DEP.open("D:\\lab\\KinectRecord\\ddd\\Record_ddd\\HandTrack_ddd\\data\\segmentation3\\ddd_demo_DEP_6.avi");
#endif

	// Read options
	//read_options(argc, argv, capture);
	// Init camera
	if (!capture_RGB.isOpened())
	{
		cout << "capture_RGB failed to open!" << endl;
		return 1;
	}

#ifdef DEP
	if (!capture_DEP.isOpened())
	{
		cout << "capture_EDP failed to open!" << endl;
		return 1;
	}
#endif

	namedWindow("CT_RGB", CV_WINDOW_AUTOSIZE);
#ifdef DEP
	namedWindow("CT_DEP", CV_WINDOW_AUTOSIZE);
#endif
	// Register mouse callback to draw the tracking box	
	//setMouseCallback("CT", mouseHandler, NULL);

	// CT framework
	CompressiveTracker ct;

	Mat frame_GRB,frame_DEP;
	Mat last_gray;
	Mat first_GRB;
	Mat detectFeatureValue_ddd;

	if (fromfile)
	{
		capture_RGB >> frame_GRB;
		cvtColor(frame_GRB, last_gray, CV_RGB2GRAY);
		frame_GRB.copyTo(first_GRB);
#ifdef DEP
		capture_DEP >> frame_DEP;
#endif
	}
	else
	{
		capture_RGB.set(CV_CAP_PROP_FRAME_WIDTH, 640);
		capture_RGB.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
#ifdef DEP
		capture_DEP.set(CV_CAP_PROP_FRAME_WIDTH, 640);
		capture_DEP.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
#endif
	}

	// Initialization
	//hand detect, get ROI box
	handDetect(first_GRB, box_RGB);

	rectangle(frame_GRB, box_RGB, Scalar(255,255,255),2);
	imshow("CT_RGB", frame_GRB);
#ifdef DEP
	box_DEP=box_RGB;
	rectangle(frame_DEP, box_DEP, Scalar(255,255,255),2);
	imshow("CT_DEP", frame_DEP);
#endif
	
	cvWaitKey(0);
		
	// Remove callback
	//setMouseCallback("CT", NULL, NULL);
	
	printf("Initial Tracking Box = x:%d y:%d h:%d w:%d\n", box_RGB.x, box_RGB.y, box_RGB.width, box_RGB.height);
	// CT initialization
	ct.init(last_gray, box_RGB);

	// Run-time
	Mat current_gray_GRB, current_gray_DEP;
	long currentFrame = 1;

	//VideoWriter
	string CTresultVideo_RGB="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\data\\ddd_occlusion_RGB_CT.avi";
	VideoWriter CTresult_RGB(CTresultVideo_RGB, CV_FOURCC('M','J','P','G'), 30, Size(640,480), true );
#ifdef DEP
	string CTresultVideo_DEP="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\HandTrack_ddd\\data\\segmentation3\\ddd_demo_DEP_6_CT.avi";
	VideoWriter CTresult_DEP(CTresultVideo_DEP, CV_FOURCC('M','J','P','G'), 30, Size(640,480), true );
#endif

	//FeatureValueWriter
	string HaarFeatureValue_RGB="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\data\\ddd_occlusion_RGB_Haar.dat";
	ofstream fout_RGB(HaarFeatureValue_RGB);
#ifdef DEP
	string HaarFeatureValue_DEP="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\HandTrack_ddd\\data\\segmentation3\\ddd_demo_DEP_6_Haar.dat";
	ofstream fout_DEP(HaarFeatureValue_DEP);
#endif

	while(capture_RGB.read(frame_GRB))
	{
#ifdef DEP
		box_DEP=box_RGB;
#endif
		// get frame
		cvtColor(frame_GRB, current_gray_GRB, CV_RGB2GRAY);
		// Process Frame
		ct.processFrame(current_gray_GRB, box_RGB, detectFeatureValue_ddd);
		//writer featureValue
		fout_RGB<< detectFeatureValue_ddd << "\n";	
		// Draw Points
		rectangle(frame_GRB, box_RGB, Scalar(255,255,255),2);
		//draw framecount
		draw_frameCount(frame_GRB,currentFrame);
		// Display
		imshow("CT_RGB", frame_GRB);
		//printf("Current Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);	
		//writer video
		CTresult_RGB.write( frame_GRB);
		
#ifdef DEP
		capture_DEP.read(frame_DEP);		
		cvtColor(frame_DEP, current_gray_DEP, CV_RGB2GRAY);
		ct.processFrame(current_gray_DEP, box_DEP, detectFeatureValue_ddd);
		fout_DEP<< detectFeatureValue_ddd << "\n";
		rectangle(frame_DEP, box_RGB, Scalar(255,255,255),2);
		draw_frameCount(frame_DEP,currentFrame);
		imshow("CT_DEP", frame_DEP);
		CTresult_DEP.write( frame_DEP);
#endif
		
		currentFrame++;

		if (cvWaitKey(33) == 'q') {	break; }
	}

	return 0;
}