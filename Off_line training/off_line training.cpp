//
//off_line training.cpp
//Hand gesture recognition
//Haar-like feature extraction from IHG dataset for matlab to train HMM model
//
//Destin Liu
//2015.3.22

//#include <afxwin.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string> 
#include <cstdio>
#include <Windows.h>
#include <math.h>
#include <io.h>
#include "CT_ddd.h"
#include "HaarFeature_ddd.h"

using namespace std;
using namespace cv;

#define RGB_ddd
//#define DEP_ddd

int featureNumber=90;  //number of Haar-like features selected in one frame
CompressiveTracker ct;

int main(int argc, char * argv[])
{
#ifdef RGB_ddd
	//init Haar
	ct.HaarFeature_ddd(featureNumber);
	
	for (int i = 0; i < 1; i++) //each video
	{
		string path_video_RGB;
		path_video_RGB = "D:\\lab\\KinectRecord\\ddd\\Record_ddd\\Off_line training\\Class_2\\RGB\\ddd_RGB_1_2.avi";
		string name_window_RGB;
		name_window_RGB = "Hand_tracking_RGB";
		string path_video_RGB_CT;
		path_video_RGB_CT = "D:\\lab\\KinectRecord\\ddd\\Record_ddd\\Off_line training\\Class_2\\RGB\\CT\\ddd_RGB_1_2_CT.avi";
		string path_featureValue_RGB;
		path_featureValue_RGB = "D:\\lab\\KinectRecord\\ddd\\Record_ddd\\Off_line training\\Class_2\\RGB\\featureValue\\ddd_RGB_1_2.dat";
		
		//load video, track hand, compute feature
		
		Mat hand_track(path_video_RGB, name_window_RGB, path_video_RGB_CT, path_featureValue_RGB, ct);



	}
#endif

#ifdef DEP_ddd
	Mat frame_DEP;
	Mat frame_DEP_gray;
	Mat featureValue_DEP_ddd;

	//init
	HaarFeature_ddd(featureNum);

	for(int currentSubject=1; currentSubject<=10; currentSubject++) //each subject, no more than 10
	{
		string folderName_img="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\FeatureExtraction\\train_sequence\\gestureData\\ddd\\Sequence";
		string folderName_dat="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\FeatureExtraction\\FeatureData\\gestureData\\ddd\\HaarFeatureValue\\matlab";
		
		for(int currentSequence=1;currentSequence<=36;currentSequence++) //each sequence, no more than 36
		{		
			char fileName_seq[1000];
			char fileName_dat[1000];
			if(currentSubject<10)
			{
				if(currentSequence<10)
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_0%d_0%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_0%d_0%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
				else
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_0%d_%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_0%d_%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
			}
			else //currentSubject=10
			{
				if(currentSequence<10)
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_%d_0%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_%d_0%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
				else
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_%d_%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_%d_%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
			}
	
			if( _access(fileName_seq,0)== -1 )
			{
				continue;
			}

			//FeatureValueWriter
			string HaarFeatureValue_DEP=fileName_dat;
			ofstream fout_DEP(HaarFeatureValue_DEP);

			for(int currentFrame=1;;currentFrame++) //each frame
			{
				//cout<< currentFrame <<endl;
				
				char fileName_img[1000];
				if(currentSubject<10)
				{
					if(currentSequence<10)
						sprintf_s(fileName_img,"%s\\sub_depth_0%d_0%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
					else
						sprintf_s(fileName_img,"%s\\sub_depth_0%d_%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
				}
				else //currentSubject=10
				{
					if(currentSequence<10)
						sprintf_s(fileName_img,"%s\\sub_depth_%d_0%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
					else
						sprintf_s(fileName_img,"%s\\sub_depth_%d_%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
				}
				
				//load path
				frame_DEP=imread(fileName_img,0);
				if(frame_DEP.empty())
				{
				  cout<< "only "<< currentFrame-1<< " frames"<< endl;
				  frameNumber=currentFrame;
				  break;
				}
				//draw framecount
			    //draw_frameCount(frame_DEP,currentFrame);

				//Draw haar_boxes
			    //for (int i=0; i<10; i++) //draw 10 boxes
				//{  
				//	rectangle(frame_DEP, features[i][1], Scalar(255,255,255),1);
				//}
				
				//display
				//imshow("frame_DEP",frame_DEP);
				//waitKey(0);

				// change to gray
				//cvtColor(frame_DEP, frame_DEP_gray, CV_RGB2GRAY);
				frame_DEP_gray=frame_DEP;
				// Process Frame
				processFrame_ddd(frame_DEP_gray, featureValue_DEP_ddd);
				
				//writer featureValue
				fout_DEP<< format(featureValue_DEP_ddd.t(),"csv")<<endl;
			}
		}
	}

#endif

	return 0;
}