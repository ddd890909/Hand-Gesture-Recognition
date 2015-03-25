//
//HaarFeature_ddd.cpp
//Hand gesture recognition
//Haar-like feature extraction from IHG dataset for matlab to train HMM model
//
//Destin Liu
//2015.3.22

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

//vector<vector<Rect>> features;
//vector<vector<float>> featuresWeight;
//RNG rng;
//int	featureMinNumRect = 2;
//int	featureMaxNumRect = 4;	// number of rectangle in a Haar is [2, 4)
//int imageWidth=120;
//int imageHeight=120;
////bool fromfile = true;
int videoWidth = 320;
int videoHeight = 240;
int videoFPS = 30;

void draw_frameCount(Mat Frame,int framecount)//Frameframecount
{
    char strFrame[10];
    sprintf_s(strFrame, "#%0d ",framecount);
    putText(Frame,strFrame,cvPoint(10,10),2,0.5,CV_RGB(200,25,200));
}

void skinExtract(const Mat &frame, Mat &skinArea)    
{    
    Mat YCbCr;    
    vector<Mat> planes;    
    
    //转换为YCrCb颜色空间    
    cvtColor(frame, YCbCr, CV_RGB2YCrCb);    
    //将多通道图像分离为多个单通道图像    
    split(YCbCr, planes);     
    
    //运用迭代器访问矩阵元素    
    MatIterator_<uchar> it_Cb = planes[1].begin<uchar>(),    
                        it_Cb_end = planes[1].end<uchar>();    
    MatIterator_<uchar> it_Cr = planes[2].begin<uchar>();    
    MatIterator_<uchar> it_skin = skinArea.begin<uchar>();    
    
    //人的皮肤颜色在YCbCr色度空间的分布范围:100<=Cb<=127, 138<=Cr<=170    
    for( ; it_Cb != it_Cb_end; ++it_Cr, ++it_Cb, ++it_skin)    
    {    
        if (138 <= *it_Cr &&  *it_Cr <= 170 && 100 <= *it_Cb &&  *it_Cb <= 127) 
            *it_skin = 255;    
        else    
            *it_skin = 0;    
    }    
    
	for (int i = 1; i < skinArea.cols; i++)
	{		
		for (int j = 1; j < skinArea.rows; j++)
		{
			if (i<180 || i>400 || j<230 || j>400)
			{
				skinArea.at<uchar>(j,i)=0;
			}
		}		
	}

    //膨胀和腐蚀，膨胀可以填补凹洞（将裂缝桥接），腐蚀可以消除细的凸起（“斑点”噪声）    
    dilate(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));    
    erode(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));    
}

void hand_detect(Mat& frameImage, Rect& rect)
{    
    Mat show_img, skinArea;
	//Mat skinAreaErode, skinAreaDilate, skinAreaOpen, skinAreaClose; 

    skinArea.create(frameImage.rows, frameImage.cols, CV_8UC1);    
    skinExtract(frameImage, skinArea);
	  
    //frameImage.copyTo(show_img, skinArea);    
	frameImage.copyTo(show_img); 

    vector< vector<Point> > contours;    
    vector<Vec4i> hierarchy;    
    
	//erode(skinArea,skinAreaErode,Mat(5,5,CV_8U),Point(-1,-1),1);
	//dilate(skinArea, skinAreaDilate,Mat(5,5,CV_8U),Point(-1,-1),1);
	//morphologyEx(skinArea,skinAreaOpen,MORPH_OPEN,Mat(3,3,CV_8U),Point(-1,-1),1);
	//morphologyEx(skinArea,skinAreaClose,MORPH_CLOSE,Mat(3,3,CV_8U),Point(-1,-1),1);

    //寻找轮廓    
    findContours(skinArea, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    

    //找到最大的轮廓    
    int index1;    
    double area, maxArea1(0);
	//int index2;
	//double maxArea2(0);
	for (int i=0; i < contours.size(); i++)    
    {    
        area = contourArea(Mat(contours[i]));    
        if (area > maxArea1)    
        {    
            maxArea1 = area;    
            index1 = i;
        } 
		//if ( (area > maxArea2) && (area < maxArea1) )
        //{    
        //    maxArea2 = area;    
        //    index2 = i;
        //}
    }    
    
    drawContours(frameImage, contours, index1, Scalar(0, 0, 255), 2, 8, hierarchy );    
        
	//draw rectangle
	Rect rectRoI=boundingRect(contours[index1]);

	rect.x=rectRoI.x;
	rect.y=rectRoI.y;
	rect.width=rectRoI.width+10;
	rect.height=rectRoI.height+10;
	//rectangle(frameImage,rect,Scalar(255,255,255),2);
	
	printf("rect.x=%d\n",rect.x);
	printf("rect.y=%d\n",rect.y);
	printf("rect.width=%d\n",rect.width);
	printf("rect.height=%d\n",rect.height);

	//imshow("frameImage", frameImage); 
    return;    
}    

void hand_track(string path_video, string name_window, string path_video_CT, string path_featureValue, CompressiveTracker &ct)
{
	VideoCapture capture_video;
	capture_video.open(path_video);
	if (!capture_video.isOpened())
	{
		cout << "capture_RGB failed to open!" << endl;
		return;
	}
	namedWindow(name_window, CV_WINDOW_AUTOSIZE);

	Mat frame_current, frame_first, frame_first_gray;
	capture_video >> frame_current;
	frame_current.copyTo(frame_first);
	cvtColor(frame_current, frame_first_gray, CV_RGB2GRAY);		
	
	Rect RoI;  //tracking box
	hand_detect(frame_first, RoI);

	rectangle(frame_current, RoI, Scalar(255,255,255), 3);
	imshow(name_window, frame_current);
	cout << "Initial Tracking Box = x:" << RoI.x << " y:" << RoI.y << " w:" << RoI.width << " h:" << RoI.height << endl;
	//cvWaitKey(0);

	//VideoWriter
	VideoWriter CTresult(path_video_CT, CV_FOURCC('M','J','P','G'), videoFPS, Size(videoWidth, videoHeight), true );
	//FeatureValueWriter
	ofstream fout_featureValue(path_featureValue);

	// CT initialization	
	ct.init(frame_first_gray, RoI);

	// Run-time	
	long frame_current_number = 1;	
	Mat frame_current_gray;
	Mat featureValue_ddd;

	while ( capture_video.read(frame_current) )
	{
		// get frame
		cvtColor(frame_current, frame_current_gray, CV_RGB2GRAY);
		// Process Frame
		ct.processFrame(frame_current_gray, RoI, featureValue_ddd);
		// Writer featureValue
		fout_featureValue << format(featureValue_ddd.t(),"csv") << endl;
		// Draw Points
		rectangle(frame_current, RoI, Scalar(255,255,255), 3);
		// Draw framecount
		draw_frameCount(frame_current, frame_current_number);
		// Display
		imshow(name_window, frame_current);
		// Writer video
		CTresult.write( frame_current );
		
		frame_current_number++;

		if (cvWaitKey(33) == 'q') 
		{	
			break; 
		}
	}
}