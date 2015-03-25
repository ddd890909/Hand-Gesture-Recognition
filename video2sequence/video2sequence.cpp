// STL Header
#include <iostream>
#include <stdio.h>
#include <string.h>
  
// OpenCV Header
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
  
// namespace
using namespace std;
using namespace cv;

//#define RECORD
#define WRITE_IMG
//#define DEP

//default width and height
//int video_size_width = 640;
//int video_size_height = 480;
//int video_size_fps = 30;

//save path
string save_path = "D:\\lab\\KinectRecord\\ddd\\Record_ddd\\data_train\\Class_1\\RGB\\ddd_RGB_1_1";

void draw_framecount(Mat Frame,int framecount)//Frameframecount
{
    char strFrame[10];
    sprintf_s(strFrame, "#%0d ",framecount);
    putText(Frame,strFrame,cvPoint(30,30),2,1,CV_RGB(25,200,25));
}

int main( int argc, char **argv )
{

	VideoCapture capture;
	capture.open("D:\\lab\\KinectRecord\\ddd\\Record_ddd\\data_train\\Class_1\\RGB\\ddd_RGB_1_1.avi"); //capture.open(0);
	// Init camera
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}

	//get total frame number
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout<<"total: "<<totalFrameNumber<<" frames"<<endl;

	//set frameToStart
	long frameToStart = 1;
	capture.set( CV_CAP_PROP_POS_FRAMES,frameToStart);
	cout<<"the start frame is the "<<frameToStart<<" frame"<<endl;

	//set frameToStop
	int frameToStop = totalFrameNumber-1;

	if(frameToStop < frameToStart)
	{
		cout<<"frameToStop is smaller than frameToStart¡I"<<endl;
		return -1;
	}
	else
	{
		cout<<"the stop frame is the "<<frameToStop<<" frame"<<endl;
	}

	double rate = capture.get(CV_CAP_PROP_FPS);
	cout<<"FPS: "<<rate<<endl;

	bool stop = false; 

	Mat frameRGB,frameBGR;

	namedWindow("frameRGB", CV_WINDOW_AUTOSIZE);
	namedWindow("frameBGR", CV_WINDOW_AUTOSIZE);

	//delay time between two frames
	int delay = 1000/rate;
	
	long currentFrame = frameToStart;
	
	char frameChar[8];

	while (!stop)
	{
		if(!capture.read(frameRGB))
		{
			cout<<"read video fail"<<endl;
			return -1;	
		}

		cout<<"reading frame "<<currentFrame<<endl;

		#ifdef DEP

			//Mat frameScaled;
			frameRGB.convertTo( frameRGB, CV_8U, 1 );

		#endif // DEP

		//draw framecount
		draw_framecount(frameRGB,currentFrame);

		//capture >> frameRGB;
		imshow("frameRGB",frameRGB);
		cvtColor(frameRGB, frameBGR, CV_RGB2BGR);
		imshow("frameBGR",frameBGR);
		
		//write images
		//frameChar=(char)currentFrame;
		sprintf_s(frameChar,"%ld",currentFrame);
		#ifdef WRITE_IMG
		
		cout<<"writing frame "<<currentFrame<<endl;

		imwrite(save_path+"\\ddd_RGB_1_1_"+frameChar+".jpg", frameRGB);			

		#endif

		//waitKey
		int c=waitKey(delay);

		//stop when ESC or frameToStop
		if((char) c == 27 || currentFrame > frameToStop)
		{
			stop = true;
		}

		//wait for next key
		if( c >= 0)
		{
			waitKey(0);
		}

		currentFrame++;
    }

	capture.release();
	waitKey(0);
	return 0;
  
}
  